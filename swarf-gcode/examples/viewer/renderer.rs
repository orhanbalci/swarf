//! The custom `wgpu` render pass embedded inside an `egui` panel via
//! `egui_wgpu::CallbackTrait` - the standard "3D viewport inside an
//! immediate-mode UI" pattern. Draws the toolpath as plain line-list
//! geometry in three passes with a fixed isometric camera fit to the
//! toolpath's bounding box (no interactive camera - see the crate root
//! docs on why that's a deliberate v1 scope decision):
//!
//!   1. The ENTIRE path, dimmed - a constant "ghost" of the whole
//!      program so the viewer always has spatial context for where the
//!      current step sits in the bigger picture.
//!   2. The path resolved SO FAR (steps `0..=current`), in real
//!      move-type colors (rapid vs. feed/arc).
//!   3. The CURRENT step's segments, highlighted, drawn last so they
//!      show up on top regardless of 3D depth.
//!
//! Depth testing is deliberately not used: the three passes are an
//! intentional back-to-front overlay (each should be visible over the
//! previous one where they overlap on screen), not real depth-sorted 3D
//! geometry, so a depth buffer would fight the design rather than help.
//!
//! The three passes need three different uniform values (a shared MVP
//! matrix, but a different color-override/tint per pass), and
//! `CallbackTrait::paint` - where the actual draw calls happen - has no
//! `wgpu::Queue` access to write a uniform buffer with (only `prepare`
//! does). So the buffer holds three separate uniform "slots" at
//! `min_uniform_buffer_offset_alignment`-aligned offsets, all written
//! during `prepare`, and `paint` only ever picks a slot via a dynamic
//! bind-group offset - no writes during paint.

use eframe::egui_wgpu::{self, wgpu};

use crate::geometry::Vertex;

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

/// Fixed isometric-style view + orthographic projection, framed to
/// fully contain the given bounding box.
pub fn fit_view_projection(bounds_min: [f32; 3], bounds_max: [f32; 3], aspect_ratio: f32) -> Mat4 {
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
    let radius =
        ((extent[0].powi(2) + extent[1].powi(2) + extent[2].powi(2)).sqrt() / 2.0).max(1.0) * 1.15; // 15% margin

    // Classic isometric direction: equal angles to all three axes.
    let d = 1.0 / 3.0f32.sqrt();
    let eye = [
        center[0] + d * radius * 3.0,
        center[1] + d * radius * 3.0,
        center[2] + d * radius * 3.0,
    ];
    // Z-up, matching G-code convention (Z is the spindle axis).
    let view = Mat4::look_at(eye, center, [0.0, 0.0, 1.0]);

    let (half_h, half_w) = if aspect_ratio >= 1.0 {
        (radius, radius * aspect_ratio)
    } else {
        (radius / aspect_ratio, radius)
    };
    let proj = Mat4::orthographic(-half_w, half_w, -half_h, half_h, 0.01, radius * 10.0);

    proj.mul(&view)
}

const GHOST_SLOT: u32 = 0;
const SO_FAR_SLOT: u32 = 1;
const CURRENT_SLOT: u32 = 2;
const SLOT_COUNT: u64 = 3;

const GHOST_COLOR: [f32; 4] = [0.35, 0.35, 0.35, 1.0];
const CURRENT_COLOR: [f32; 4] = [1.0, 0.85, 0.1, 1.0];

pub struct PathRenderResources {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    uniform_slot_stride: u64,
    bind_group: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
    /// 0 = use each vertex's own color; 1 = ignore it and use `tint`.
    override_color: u32,
    _padding: [u32; 3],
    tint: [f32; 4],
}

const SHADER_SRC: &str = r#"
struct Uniforms {
    mvp: mat4x4<f32>,
    override_color: u32,
    // Three plain u32 fields, NOT `vec3<u32>` - a vec3 has the same
    // 16-byte alignment as vec4 in WGSL's uniform-address-space layout
    // rules, which would insert padding before `tint` that Rust's
    // `#[repr(C)] [u32; 3]` (4-byte aligned) doesn't produce, making
    // the two sides disagree on the struct's total size (confirmed
    // directly: wgpu's validator rejected the pipeline over exactly
    // this mismatch - 112 bytes computed by WGSL vs. 96 by Rust).
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
        vertices: &[Vertex],
    ) -> Self {
        use wgpu::util::DeviceExt as _;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewer.shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let uniform_size = std::mem::size_of::<Uniforms>() as u64;
        let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let uniform_slot_stride = uniform_size.div_ceil(alignment) * alignment;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer.bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(uniform_size),
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewer.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewer.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            // No depth/stencil attachment - see this module's docs for why
            // the three-pass overlay is intentionally depth-free.
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("viewer.vertex_buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewer.uniform_buffer"),
            size: uniform_slot_stride * SLOT_COUNT,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer.bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(uniform_size),
                }),
            }],
        });

        Self {
            pipeline,
            vertex_buffer,
            uniform_buffer,
            uniform_slot_stride,
            bind_group,
        }
    }

    fn write_slot(
        &self,
        queue: &wgpu::Queue,
        slot: u32,
        mvp: Mat4,
        override_color: bool,
        tint: [f32; 4],
    ) {
        let uniforms = Uniforms {
            mvp: mvp.0,
            override_color: override_color as u32,
            _padding: [0; 3],
            tint,
        };
        let offset = slot as u64 * self.uniform_slot_stride;
        queue.write_buffer(&self.uniform_buffer, offset, bytemuck::bytes_of(&uniforms));
    }

    fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>, slot: u32, range: std::ops::Range<u32>) {
        if range.is_empty() {
            return;
        }
        let offset = slot as u32 * self.uniform_slot_stride as u32;
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[offset]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(range, 0..1);
    }
}

/// One frame's worth of parameters for the paint callback.
pub struct PathPaintCallback {
    pub mvp: Mat4,
    pub full_range: std::ops::Range<u32>,
    pub so_far_range: std::ops::Range<u32>,
    pub current_range: std::ops::Range<u32>,
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
        resources.write_slot(queue, GHOST_SLOT, self.mvp, true, GHOST_COLOR);
        resources.write_slot(queue, SO_FAR_SLOT, self.mvp, false, [0.0; 4]);
        resources.write_slot(queue, CURRENT_SLOT, self.mvp, true, CURRENT_COLOR);
        Vec::new()
    }

    fn paint(
        &self,
        _info: eframe::epaint::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let resources: &PathRenderResources = callback_resources.get().unwrap();
        resources.draw(render_pass, GHOST_SLOT, self.full_range.clone());
        resources.draw(render_pass, SO_FAR_SLOT, self.so_far_range.clone());
        resources.draw(render_pass, CURRENT_SLOT, self.current_range.clone());
    }
}
