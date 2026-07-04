//! The custom `wgpu` render pass embedded inside an `egui` panel via
//! `egui_wgpu::CallbackTrait` - the standard "3D viewport inside an
//! immediate-mode UI" pattern.
//!
//! Two pipelines, since they need genuinely different vertex shaders:
//!
//!   - **Line pipeline** (`LineVertex`, see `geometry.rs`): draws the
//!     toolpath and axes as actual thick lines. `wgpu`/WebGPU has no
//!     portable line-width control for `LineList` geometry, so getting
//!     real thickness means expanding each segment into a camera-facing
//!     quad in the vertex shader (a standard technique) - given each
//!     vertex's own position, its segment's other endpoint, and a
//!     `side` sign, the shader projects both endpoints, finds the
//!     segment's on-screen direction, and pushes this vertex sideways
//!     by half the desired thickness (a per-draw-call uniform, in
//!     points) perpendicular to that direction. Since this happens in
//!     the shader (not baked into vertex data), the SAME geometry can
//!     be drawn at different thicknesses for different passes - thin
//!     for the "ghost" pass, thicker for "so far"/"current"/axes.
//!   - **Solid pipeline** (`Vertex`): a plain, un-thickened
//!     position+color triangle pass with a direct MVP transform and a
//!     uniform tint - used only for the small origin-marker sphere,
//!     which needs real filled 3D geometry, not a thickened line.
//!
//! Draws happen in four (line) + one (solid) passes with a fixed
//! isometric camera fit to the toolpath's bounding box (no interactive
//! camera - see the crate root docs on why that's a deliberate v1 scope
//! decision):
//!
//!   0. X/Y/Z reference axes through the machine origin (red/green/
//!      blue), drawn first as a constant background reference.
//!   1. The ENTIRE path, dimmed - a constant "ghost" of the whole
//!      program so the viewer always has spatial context for where the
//!      current step sits in the bigger picture.
//!   2. The path resolved SO FAR (steps `0..=current`), in real
//!      move-type colors (rapid vs. feed/arc), drawn thicker than the
//!      ghost pass so progress reads clearly at a glance.
//!   3. The CURRENT step's segments, highlighted and thickest, drawn
//!      last so they show up on top regardless of 3D depth.
//!   4. The origin marker sphere (solid pipeline).
//!
//! Depth testing is deliberately not used for the line passes: 0-3 are
//! an intentional back-to-front overlay (each should be visible over
//! the previous one where they overlap on screen), not real
//! depth-sorted 3D geometry, so a depth buffer would fight the design
//! rather than help. The sphere doesn't need depth testing either since
//! it's the only solid object in the scene.
//!
//! The line pipeline's four passes need different uniform values (a
//! shared MVP matrix, but a different color-override/tint/thickness per
//! pass), and `CallbackTrait::paint` - where the actual draw calls
//! happen - has no `wgpu::Queue` access to write a uniform buffer with
//! (only `prepare` does). So its buffer holds four separate uniform
//! "slots" at `min_uniform_buffer_offset_alignment`-aligned offsets,
//! all written during `prepare`, and `paint` only ever picks a slot via
//! a dynamic bind-group offset - no writes during paint. The solid
//! pipeline only ever draws once per frame, so its uniform buffer
//! doesn't need multiple slots.

use eframe::egui_wgpu::{self, wgpu};

use crate::geometry::{LineVertex, Vertex};

/// Bare-bones column-major 4x4 matrix (matching WGSL's `mat4x4`
/// layout) - hand-rolled instead of pulling in a math crate for the
/// one matrix this example needs.
#[derive(Clone, Copy)]
pub struct Mat4(pub [[f32; 4]; 4]);

impl Mat4 {
    fn identity() -> Self {
        Self([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    fn mul(&self, rhs: &Mat4) -> Mat4 {
        let mut out = [[0.0f32; 4]; 4];
        for col in 0..4 {
            for row in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.0[k][row] * rhs.0[col][k];
                }
                out[col][row] = sum;
            }
        }
        Mat4(out)
    }

    /// Right-handed look-at view matrix.
    fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Mat4 {
        fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
            [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
        }
        fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
            a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        }
        fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        }
        fn normalize(a: [f32; 3]) -> [f32; 3] {
            let len = dot(a, a).sqrt();
            [a[0] / len, a[1] / len, a[2] / len]
        }

        let f = normalize(sub(target, eye)); // forward
        let s = normalize(cross(f, up)); // right
        let u = cross(s, f); // recomputed up

        Mat4([
            [s[0], u[0], -f[0], 0.0],
            [s[1], u[1], -f[1], 0.0],
            [s[2], u[2], -f[2], 0.0],
            [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0],
        ])
    }

    /// Orthographic projection, wgpu's [0, 1] depth range.
    #[allow(clippy::too_many_arguments)]
    fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat4 {
        let mut m = Mat4::identity();
        m.0[0][0] = 2.0 / (right - left);
        m.0[1][1] = 2.0 / (top - bottom);
        m.0[2][2] = 1.0 / (near - far);
        m.0[3][0] = (left + right) / (left - right);
        m.0[3][1] = (bottom + top) / (bottom - top);
        m.0[3][2] = near / (near - far);
        m
    }
}

/// An orbit camera: rotation (`yaw`/`pitch`, spherical angles around a
/// target) and `zoom`, a multiplier on the fitted orthographic
/// half-width/half-height. The target and base distance are NOT stored
/// here - they're always re-derived from the scene's current bounding
/// box each frame (see `view_projection`), so the camera always frames
/// the toolpath even as `zoom`/rotation are adjusted on top of that fit.
///
/// Orthographic zoom can't be done by moving the eye closer (unlike a
/// perspective camera, distance alone doesn't change apparent size
/// under orthographic projection) - `zoom` scales the projection's
/// half-extents directly instead.
#[derive(Clone, Copy)]
pub struct OrbitCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub zoom: f32,
}

/// Keeps the eye direction away from parallel-to-up, where `look_at`
/// degenerates (its `right` vector, `cross(forward, up)`, goes to
/// zero).
const PITCH_LIMIT: f32 = 1.5; // ~85.9 degrees

impl OrbitCamera {
    /// The same fixed angle this viewer used before camera controls
    /// existed: equal angles to all three axes.
    pub fn default_isometric() -> Self {
        Self {
            yaw: std::f32::consts::FRAC_PI_4,
            pitch: (1.0 / 2.0f32.sqrt()).atan(),
            zoom: 1.0,
        }
    }

    pub fn orbit(&mut self, yaw_delta: f32, pitch_delta: f32) {
        self.yaw -= yaw_delta;
        self.pitch = (self.pitch + pitch_delta).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    pub fn zoom_by(&mut self, factor: f32) {
        self.zoom = (self.zoom * factor).clamp(0.05, 20.0);
    }

    /// Straight-down top view of the XY plane. Looking exactly along Z
    /// is the one case `eye_direction`'s fixed Z-up convention can't
    /// disambiguate a `right` vector for on its own - `view_projection`
    /// switches to a Y-up basis when it detects this.
    pub fn view_xy(&mut self) {
        self.yaw = 0.0;
        self.pitch = std::f32::consts::FRAC_PI_2;
    }

    /// Front view of the XZ plane (looking along +Y).
    pub fn view_xz(&mut self) {
        self.yaw = -std::f32::consts::FRAC_PI_2;
        self.pitch = 0.0;
    }

    /// Side view of the YZ plane (looking along -X).
    pub fn view_yz(&mut self) {
        self.yaw = 0.0;
        self.pitch = 0.0;
    }

    fn eye_direction(&self) -> [f32; 3] {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        // Z-up, matching G-code convention (Z is the spindle axis).
        [cos_pitch * cos_yaw, cos_pitch * sin_yaw, sin_pitch]
    }
}

/// View + orthographic projection for `camera`, framed to fully contain
/// the given bounding box at `camera.zoom == 1.0` (values above 1.0
/// zoom out, below 1.0 zoom in).
pub fn view_projection(
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    aspect_ratio: f32,
    camera: &OrbitCamera,
) -> Mat4 {
    let center = [
        (bounds_min[0] + bounds_max[0]) / 2.0,
        (bounds_min[1] + bounds_max[1]) / 2.0,
        (bounds_min[2] + bounds_max[2]) / 2.0,
    ];
    let extent = [
        (bounds_max[0] - bounds_min[0]).max(1.0),
        (bounds_max[1] - bounds_min[1]).max(1.0),
        (bounds_max[2] - bounds_min[2]).max(1.0),
    ];
    // Radius of a sphere containing the whole box - simplest way to
    // guarantee the box fits regardless of viewing angle.
    let base_radius =
        ((extent[0].powi(2) + extent[1].powi(2) + extent[2].powi(2)).sqrt() / 2.0).max(1.0) * 1.15; // 15% margin
    let radius = base_radius * camera.zoom;

    let eye_dir = camera.eye_direction();
    let eye = [
        center[0] + eye_dir[0] * base_radius * 3.0,
        center[1] + eye_dir[1] * base_radius * 3.0,
        center[2] + eye_dir[2] * base_radius * 3.0,
    ];
    // Looking straight down/up the Z axis makes `right = cross(forward,
    // up)` degenerate for the usual Z-up basis (the top/bottom preset
    // views land exactly here) - fall back to a Y-up basis in that case.
    let up = if eye_dir[0].abs() < 1e-3 && eye_dir[1].abs() < 1e-3 {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let view = Mat4::look_at(eye, center, up);

    let (half_h, half_w) = if aspect_ratio >= 1.0 {
        (radius, radius * aspect_ratio)
    } else {
        (radius / aspect_ratio, radius)
    };
    // Near/far stay keyed to `base_radius`, not the zoomed `radius`, so
    // zooming in doesn't clip geometry that's still within the
    // originally-fitted scene depth.
    let proj = Mat4::orthographic(-half_w, half_w, -half_h, half_h, 0.01, base_radius * 10.0);

    proj.mul(&view)
}

const GHOST_SLOT: u32 = 0;
const SO_FAR_SLOT: u32 = 1;
const CURRENT_SLOT: u32 = 2;
const AXES_SLOT: u32 = 3;
const SLOT_COUNT: u64 = 4;

const GHOST_COLOR: [f32; 4] = [0.35, 0.35, 0.35, 1.0];
const CURRENT_COLOR: [f32; 4] = [1.0, 0.85, 0.1, 1.0];
const SPHERE_COLOR: [f32; 4] = [1.0, 0.55, 0.15, 1.0];

const GHOST_THICKNESS_PX: f32 = 1.4;
const SO_FAR_THICKNESS_PX: f32 = 3.0;
const CURRENT_THICKNESS_PX: f32 = 4.5;
const AXES_THICKNESS_PX: f32 = 3.0;

// --- Line pipeline (toolpath + axes): thick lines via screen-space
// quad expansion in the vertex shader - see this module's docs. ---

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LineUniforms {
    mvp: [[f32; 4]; 4],
    /// 0 = use each vertex's own color; 1 = ignore it and use `tint`.
    override_color: u32,
    thickness_px: f32,
    viewport_w: f32,
    viewport_h: f32,
    tint: [f32; 4],
}

const LINE_SHADER_SRC: &str = r#"
struct Uniforms {
    mvp: mat4x4<f32>,
    override_color: u32,
    thickness_px: f32,
    viewport_w: f32,
    viewport_h: f32,
    tint: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) other: vec3<f32>,
    @location(2) side: f32,
    @location(3) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let clip_a = u.mvp * vec4<f32>(in.position, 1.0);
    let clip_b = u.mvp * vec4<f32>(in.other, 1.0);
    let viewport = vec2<f32>(u.viewport_w, u.viewport_h);

    let screen_a = (clip_a.xy / clip_a.w) * viewport * 0.5;
    let screen_b = (clip_b.xy / clip_b.w) * viewport * 0.5;
    var dir = screen_b - screen_a;
    let len = length(dir);
    if (len > 0.0001) {
        dir = dir / len;
    } else {
        dir = vec2<f32>(1.0, 0.0);
    }
    let normal = vec2<f32>(-dir.y, dir.x);
    let offset_px = normal * in.side * (u.thickness_px * 0.5);

    var clip = clip_a;
    clip.x = clip.x + offset_px.x / viewport.x * 2.0 * clip_a.w;
    clip.y = clip.y + offset_px.y / viewport.y * 2.0 * clip_a.w;

    out.clip_position = clip;
    if (u.override_color == 1u) {
        out.color = u.tint;
    } else {
        out.color = vec4<f32>(in.color, 1.0);
    }
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

pub struct PathRenderResources {
    line_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    axis_vertex_buffer: wgpu::Buffer,
    line_uniform_buffer: wgpu::Buffer,
    line_uniform_slot_stride: u64,
    line_bind_group: wgpu::BindGroup,

    solid_pipeline: wgpu::RenderPipeline,
    sphere_vertex_buffer: wgpu::Buffer,
    sphere_vertex_count: u32,
    solid_uniform_buffer: wgpu::Buffer,
    solid_bind_group: wgpu::BindGroup,
}

// --- Solid pipeline (origin marker sphere): plain position+color
// triangles, direct MVP transform, single uniform tint. ---

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SolidUniforms {
    mvp: [[f32; 4]; 4],
    override_color: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
    tint: [f32; 4],
}

const SOLID_SHADER_SRC: &str = r#"
struct Uniforms {
    mvp: mat4x4<f32>,
    override_color: u32,
    // Three plain u32 fields, NOT `vec3<u32>` - a vec3 has the same
    // 16-byte alignment as vec4 in WGSL's uniform-address-space layout
    // rules, which would insert padding before `tint` that Rust's
    // `#[repr(C)] [u32; 3]` (4-byte aligned) doesn't produce, making
    // the two sides disagree on the struct's total size (confirmed
    // directly: wgpu's validator rejected an earlier version of this
    // pipeline over exactly this mismatch).
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
    tint: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4<f32>(in.position, 1.0);
    if (u.override_color == 1u) {
        out.color = u.tint;
    } else {
        out.color = vec4<f32>(in.color, 1.0);
    }
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

impl PathRenderResources {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        path_vertices: &[LineVertex],
        axis_vertices: &[LineVertex],
        sphere_vertices: &[Vertex],
    ) -> Self {
        use wgpu::util::DeviceExt as _;

        // --- line pipeline ---

        let line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewer.line_shader"),
            source: wgpu::ShaderSource::Wgsl(LINE_SHADER_SRC.into()),
        });

        let line_uniform_size = std::mem::size_of::<LineUniforms>() as u64;
        let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let line_uniform_slot_stride = line_uniform_size.div_ceil(alignment) * alignment;

        let line_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("viewer.line_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(line_uniform_size),
                    },
                    count: None,
                }],
            });

        let line_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewer.line_pipeline_layout"),
            bind_group_layouts: &[&line_bind_group_layout],
            push_constant_ranges: &[],
        });

        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewer.line_pipeline"),
            layout: Some(&line_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &line_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3, // position
                        1 => Float32x3, // other
                        2 => Float32,   // side
                        3 => Float32x3, // color
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &line_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("viewer.path_vertex_buffer"),
            contents: bytemuck::cast_slice(path_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let axis_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("viewer.axis_vertex_buffer"),
            contents: bytemuck::cast_slice(axis_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let line_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewer.line_uniform_buffer"),
            size: line_uniform_slot_stride * SLOT_COUNT,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let line_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer.line_bind_group"),
            layout: &line_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &line_uniform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(line_uniform_size),
                }),
            }],
        });

        // --- solid pipeline (sphere) ---

        let solid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewer.solid_shader"),
            source: wgpu::ShaderSource::Wgsl(SOLID_SHADER_SRC.into()),
        });

        let solid_uniform_size = std::mem::size_of::<SolidUniforms>() as u64;

        let solid_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("viewer.solid_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(solid_uniform_size),
                    },
                    count: None,
                }],
            });

        let solid_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("viewer.solid_pipeline_layout"),
                bind_group_layouts: &[&solid_bind_group_layout],
                push_constant_ranges: &[],
            });

        let solid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewer.solid_pipeline"),
            layout: Some(&solid_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &solid_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &solid_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sphere_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("viewer.sphere_vertex_buffer"),
            contents: bytemuck::cast_slice(sphere_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let solid_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewer.solid_uniform_buffer"),
            size: solid_uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let solid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer.solid_bind_group"),
            layout: &solid_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: solid_uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            line_pipeline,
            vertex_buffer,
            axis_vertex_buffer,
            line_uniform_buffer,
            line_uniform_slot_stride,
            line_bind_group,
            solid_pipeline,
            sphere_vertex_buffer,
            sphere_vertex_count: sphere_vertices.len() as u32,
            solid_uniform_buffer,
            solid_bind_group,
        }
    }

    fn write_line_slot(
        &self,
        queue: &wgpu::Queue,
        slot: u32,
        mvp: Mat4,
        override_color: bool,
        thickness_px: f32,
        viewport: [f32; 2],
        tint: [f32; 4],
    ) {
        let uniforms = LineUniforms {
            mvp: mvp.0,
            override_color: override_color as u32,
            thickness_px,
            viewport_w: viewport[0],
            viewport_h: viewport[1],
            tint,
        };
        let offset = slot as u64 * self.line_uniform_slot_stride;
        queue.write_buffer(
            &self.line_uniform_buffer,
            offset,
            bytemuck::bytes_of(&uniforms),
        );
    }

    fn draw_line_buffer(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        slot: u32,
        buffer: &wgpu::Buffer,
        range: std::ops::Range<u32>,
    ) {
        if range.is_empty() {
            return;
        }
        let offset = slot * self.line_uniform_slot_stride as u32;
        render_pass.set_pipeline(&self.line_pipeline);
        render_pass.set_bind_group(0, &self.line_bind_group, &[offset]);
        render_pass.set_vertex_buffer(0, buffer.slice(..));
        render_pass.draw(range, 0..1);
    }

    fn draw_path(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        slot: u32,
        range: std::ops::Range<u32>,
    ) {
        self.draw_line_buffer(render_pass, slot, &self.vertex_buffer, range);
    }

    fn draw_axes(&self, render_pass: &mut wgpu::RenderPass<'_>, axis_vertex_count: u32) {
        self.draw_line_buffer(
            render_pass,
            AXES_SLOT,
            &self.axis_vertex_buffer,
            0..axis_vertex_count,
        );
    }

    fn write_solid_uniforms(&self, queue: &wgpu::Queue, mvp: Mat4, tint: [f32; 4]) {
        let uniforms = SolidUniforms {
            mvp: mvp.0,
            override_color: 1,
            _padding0: 0,
            _padding1: 0,
            _padding2: 0,
            tint,
        };
        queue.write_buffer(&self.solid_uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    fn draw_sphere(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        if self.sphere_vertex_count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.solid_pipeline);
        render_pass.set_bind_group(0, &self.solid_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.sphere_vertex_buffer.slice(..));
        render_pass.draw(0..self.sphere_vertex_count, 0..1);
    }
}

/// One frame's worth of parameters for the paint callback.
pub struct PathPaintCallback {
    pub mvp: Mat4,
    pub viewport: [f32; 2],
    pub full_range: std::ops::Range<u32>,
    pub so_far_range: std::ops::Range<u32>,
    pub current_range: std::ops::Range<u32>,
    pub axis_vertex_count: u32,
}

impl egui_wgpu::CallbackTrait for PathPaintCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources: &PathRenderResources = callback_resources.get().unwrap();
        resources.write_line_slot(
            queue,
            GHOST_SLOT,
            self.mvp,
            true,
            GHOST_THICKNESS_PX,
            self.viewport,
            GHOST_COLOR,
        );
        resources.write_line_slot(
            queue,
            SO_FAR_SLOT,
            self.mvp,
            false,
            SO_FAR_THICKNESS_PX,
            self.viewport,
            [0.0; 4],
        );
        resources.write_line_slot(
            queue,
            CURRENT_SLOT,
            self.mvp,
            true,
            CURRENT_THICKNESS_PX,
            self.viewport,
            CURRENT_COLOR,
        );
        resources.write_line_slot(
            queue,
            AXES_SLOT,
            self.mvp,
            false,
            AXES_THICKNESS_PX,
            self.viewport,
            [0.0; 4],
        );
        resources.write_solid_uniforms(queue, self.mvp, SPHERE_COLOR);
        Vec::new()
    }

    fn paint(
        &self,
        _info: eframe::epaint::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let resources: &PathRenderResources = callback_resources.get().unwrap();
        // Axes first, as a background reference the toolpath draws over.
        resources.draw_axes(render_pass, self.axis_vertex_count);
        resources.draw_path(render_pass, GHOST_SLOT, self.full_range.clone());
        resources.draw_path(render_pass, SO_FAR_SLOT, self.so_far_range.clone());
        resources.draw_path(render_pass, CURRENT_SLOT, self.current_range.clone());
        resources.draw_sphere(render_pass);
    }
}
