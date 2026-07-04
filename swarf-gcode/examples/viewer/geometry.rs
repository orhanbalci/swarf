//! Turns a recorded interpretation trace into renderable geometry: one
//! big vertex buffer built once at load time, plus the vertex range
//! each step contributed, so the viewer can slice out "path so far" /
//! "current step" every frame without re-tessellating anything.
//!
//! Two vertex formats are used:
//!
//!   - [`LineVertex`] for the toolpath and axes. `wgpu`/WebGPU has no
//!     portable "line width" control for `LineList` geometry, so
//!     drawing anything other than hairline-thin paths means expanding
//!     each segment into a camera-facing quad ourselves. Rather than
//!     bake a fixed thickness into these vertices (which would prevent
//!     drawing the SAME geometry thin for the "ghost" pass and thick
//!     for "so far"/"current" - see `renderer.rs`), each vertex carries
//!     its own position, its segment's OTHER endpoint, and a `side`
//!     sign; the actual thickening happens in the vertex shader using a
//!     per-draw-call thickness uniform, in screen space.
//!   - [`Vertex`] (plain position + color, `TriangleList`) for ordinary
//!     solid 3D geometry - currently just the small origin marker
//!     sphere, which needs real filled geometry, not a thickened line.

use std::ops::Range;

use swarf_gcode::{LineOutput, MotionMode, Position, ResolvedMotionCommand};

/// Plain solid-geometry vertex: position + color, rendered with a
/// direct MVP transform and no thickening - see this module's docs.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

/// A thick-line vertex: this vertex's own position, its segment's
/// other endpoint (so the vertex shader can compute the segment's
/// on-screen direction), and which side of the line ([-1.0, 1.0]) this
/// vertex should be pushed toward once thickened - see this module's
/// docs and `renderer.rs`'s vertex shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineVertex {
    pub position: [f32; 3],
    pub other: [f32; 3],
    pub side: f32,
    pub color: [f32; 3],
}

const RAPID_COLOR: [f32; 3] = [0.3, 0.6, 1.0];
const FEED_COLOR: [f32; 3] = [0.9, 0.9, 0.9];

/// How many line segments to tessellate a full-turn arc into; shorter
/// arcs use proportionally fewer, down to a minimum of 2.
const MAX_ARC_SEGMENTS: usize = 64;

/// Which of the three axis-aligned planes (G17/G18/G19) an arc lies in.
/// `ResolvedMotionCommand` doesn't carry the active plane directly
/// (Interface 2 only stores the resolved center), so we recover it from
/// the geometry itself: `resolve_arc_center` always copies the center's
/// out-of-plane coordinate from the arc's start point, so whichever axis
/// has `center` closest to `start` on that axis is the out-of-plane one.
#[derive(Clone, Copy)]
enum Plane {
    Xy,
    Yz,
    Zx,
}

fn detect_plane(start: Position, center: Position) -> Plane {
    let dx = (center.x - start.x).abs();
    let dy = (center.y - start.y).abs();
    let dz = (center.z - start.z).abs();
    if dz <= dy && dz <= dx {
        Plane::Xy
    } else if dy <= dx {
        Plane::Zx
    } else {
        Plane::Yz
    }
}

fn in_plane(p: Position, plane: Plane) -> (f64, f64) {
    match plane {
        Plane::Xy => (p.x, p.y),
        Plane::Yz => (p.y, p.z),
        Plane::Zx => (p.z, p.x),
    }
}

fn linear_coord(p: Position, plane: Plane) -> f64 {
    match plane {
        Plane::Xy => p.z,
        Plane::Yz => p.x,
        Plane::Zx => p.y,
    }
}

fn from_plane(u: f64, v: f64, linear: f64, plane: Plane) -> Position {
    match plane {
        Plane::Xy => Position {
            x: u,
            y: v,
            z: linear,
        },
        Plane::Yz => Position {
            x: linear,
            y: u,
            z: v,
        },
        Plane::Zx => Position {
            x: v,
            y: linear,
            z: u,
        },
    }
}

fn to_line_vertex(position: Position, other: Position, side: f32, color: [f32; 3]) -> LineVertex {
    LineVertex {
        position: [position.x as f32, position.y as f32, position.z as f32],
        other: [other.x as f32, other.y as f32, other.z as f32],
        side,
        color,
    }
}

/// Expand one segment (`a` -> `b`) into a quad (2 triangles, 6
/// vertices, non-indexed) - the thickening itself happens later, in the
/// vertex shader, using each vertex's `side` and `other` fields.
fn push_segment(vertices: &mut Vec<LineVertex>, a: Position, b: Position, color: [f32; 3]) {
    let a_neg = to_line_vertex(a, b, -1.0, color);
    let a_pos = to_line_vertex(a, b, 1.0, color);
    let b_neg = to_line_vertex(b, a, -1.0, color);
    let b_pos = to_line_vertex(b, a, 1.0, color);
    vertices.push(a_neg);
    vertices.push(a_pos);
    vertices.push(b_neg);
    vertices.push(a_pos);
    vertices.push(b_pos);
    vertices.push(b_neg);
}

/// Tessellate one resolved move into thick-line quads. The vertex count
/// per segment (6, vs. 2 for a plain line list) is irrelevant for a
/// toolpath's scale - even a large real-world file stays well within
/// trivial limits for any GPU.
fn push_motion(vertices: &mut Vec<LineVertex>, m: &ResolvedMotionCommand) {
    let color = if m.motion_mode == MotionMode::Rapid {
        RAPID_COLOR
    } else {
        FEED_COLOR
    };

    let Some(arc) = m.arc else {
        push_segment(vertices, m.start, m.target, color);
        return;
    };

    let plane = detect_plane(m.start, arc.center);
    let (cu, cv) = in_plane(arc.center, plane);
    let (u0, v0) = in_plane(m.start, plane);
    let (u1, v1) = in_plane(m.target, plane);
    let (du0, dv0) = (u0 - cu, v0 - cv);
    let (du1, dv1) = (u1 - cu, v1 - cv);
    let radius = (du0 * du0 + dv0 * dv0).sqrt();

    let angle0 = dv0.atan2(du0);
    let angle1 = dv1.atan2(du1);

    let clockwise = m.motion_mode == MotionMode::ArcClockwise;
    let full_circle = (m.start.x - m.target.x).abs() < 1e-9
        && (m.start.y - m.target.y).abs() < 1e-9
        && (m.start.z - m.target.z).abs() < 1e-9;

    const TAU: f64 = std::f64::consts::TAU;
    let sweep = if full_circle {
        if clockwise {
            -TAU
        } else {
            TAU
        }
    } else if clockwise {
        // Clockwise = decreasing angle; wrap to a negative sweep.
        let mut d = angle0 - angle1;
        if d <= 0.0 {
            d += TAU;
        }
        -d
    } else {
        let mut d = angle1 - angle0;
        if d <= 0.0 {
            d += TAU;
        }
        d
    };

    let segments = ((sweep.abs() / TAU) * MAX_ARC_SEGMENTS as f64)
        .ceil()
        .max(2.0) as usize;
    let linear0 = linear_coord(m.start, plane);
    let linear1 = linear_coord(m.target, plane);

    let mut prev = m.start;
    for i in 1..=segments {
        let t = i as f64 / segments as f64;
        let angle = angle0 + sweep * t;
        let (u, v) = (cu + radius * angle.cos(), cv + radius * angle.sin());
        let linear = linear0 + (linear1 - linear0) * t;
        let point = from_plane(u, v, linear, plane);
        push_segment(vertices, prev, point, color);
        prev = point;
    }
}

/// One entry in the recorded interpretation trace: which source line
/// produced this output, and the output itself.
pub struct TraceEntry {
    pub line: usize,
    pub output: LineOutput,
}

const AXIS_X_COLOR: [f32; 3] = [1.0, 0.25, 0.25];
const AXIS_Y_COLOR: [f32; 3] = [0.25, 1.0, 0.25];
const AXIS_Z_COLOR: [f32; 3] = [0.35, 0.55, 1.0];

/// Build thick-line quads for X/Y/Z reference axes through the machine
/// origin, each `length` long - red/green/blue by the usual CAD
/// convention. Kept as its own small, separate geometry (not part of
/// `Scene::vertices`) since axes should always be fully visible
/// regardless of the current step, unlike the toolpath itself.
pub fn axis_vertices(length: f64) -> Vec<LineVertex> {
    let origin = Position {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    let mut vertices = Vec::with_capacity(18);
    push_segment(
        &mut vertices,
        origin,
        Position {
            x: length,
            y: 0.0,
            z: 0.0,
        },
        AXIS_X_COLOR,
    );
    push_segment(
        &mut vertices,
        origin,
        Position {
            x: 0.0,
            y: length,
            z: 0.0,
        },
        AXIS_Y_COLOR,
    );
    push_segment(
        &mut vertices,
        origin,
        Position {
            x: 0.0,
            y: 0.0,
            z: length,
        },
        AXIS_Z_COLOR,
    );
    vertices
}

/// Build a small UV sphere (`TriangleList`, position-only - color comes
/// from a uniform tint at render time, see `renderer.rs`) centered on
/// the origin, marking it clearly in 3D regardless of viewing angle
/// (unlike the axes, which vanish end-on from some angles).
pub fn sphere_vertices(radius: f64) -> Vec<Vertex> {
    const LON_SEGMENTS: usize = 16;
    const LAT_SEGMENTS: usize = 10;
    const TAU: f64 = std::f64::consts::TAU;
    const PI: f64 = std::f64::consts::PI;

    let point = |theta: f64, phi: f64| -> Vertex {
        Vertex {
            position: [
                (radius * theta.sin() * phi.cos()) as f32,
                (radius * theta.sin() * phi.sin()) as f32,
                (radius * theta.cos()) as f32,
            ],
            color: [1.0, 1.0, 1.0], // unused - sphere pipeline always uses a uniform tint
        }
    };

    let mut vertices = Vec::with_capacity(LON_SEGMENTS * LAT_SEGMENTS * 6);
    for lat in 0..LAT_SEGMENTS {
        let theta0 = PI * lat as f64 / LAT_SEGMENTS as f64;
        let theta1 = PI * (lat + 1) as f64 / LAT_SEGMENTS as f64;
        for lon in 0..LON_SEGMENTS {
            let phi0 = TAU * lon as f64 / LON_SEGMENTS as f64;
            let phi1 = TAU * (lon + 1) as f64 / LON_SEGMENTS as f64;
            // A quad per (lat, lon) cell, split into 2 triangles - the
            // triangles at the poles (theta0=0 or theta1=PI) degenerate
            // to zero area, which is harmless (just a few wasted
            // vertices) for geometry this small.
            let p00 = point(theta0, phi0);
            let p10 = point(theta1, phi0);
            let p11 = point(theta1, phi1);
            let p01 = point(theta0, phi1);
            vertices.push(p00);
            vertices.push(p10);
            vertices.push(p11);
            vertices.push(p00);
            vertices.push(p11);
            vertices.push(p01);
        }
    }
    vertices
}

/// Build a small "spindle head" shape (`TriangleList`, position-only,
/// same convention as [`sphere_vertices`]) for the tool-position marker:
/// a tapered cone tip with a narrower cylindrical shank above it,
/// reading as a tool bit rather than a generic blob. The apex sits at
/// the local origin - `renderer.rs` translates that point to the
/// current tool tip position, so the marker's apex is exactly where the
/// tool tip actually is, with the "body" extending upward from there.
pub fn spindle_marker_vertices(radius: f64) -> Vec<Vertex> {
    const SEGMENTS: usize = 20;
    const TAU: f64 = std::f64::consts::TAU;

    let cone_radius = radius;
    let cone_height = radius * 1.6;
    let shank_radius = radius * 0.5;
    let shank_height = radius * 1.8;

    let ring = |z: f64, r: f64| -> Vec<[f32; 3]> {
        (0..SEGMENTS)
            .map(|i| {
                let angle = TAU * i as f64 / SEGMENTS as f64;
                [
                    (r * angle.cos()) as f32,
                    (r * angle.sin()) as f32,
                    z as f32,
                ]
            })
            .collect()
    };
    let vert = |position: [f32; 3]| -> Vertex {
        Vertex {
            position,
            color: [1.0, 1.0, 1.0], // unused - marker pipeline always uses a uniform tint
        }
    };

    let apex = [0.0f32, 0.0, 0.0];
    let cone_base = ring(cone_height, cone_radius);
    let shank_base = ring(cone_height, shank_radius);
    let shank_top_ring = ring(cone_height + shank_height, shank_radius);
    let shank_top_center = [0.0f32, 0.0, (cone_height + shank_height) as f32];

    let mut vertices = Vec::with_capacity(SEGMENTS * 9);
    for i in 0..SEGMENTS {
        let j = (i + 1) % SEGMENTS;

        // Cone side: apex to the cutting-tip base circle.
        vertices.push(vert(apex));
        vertices.push(vert(cone_base[i]));
        vertices.push(vert(cone_base[j]));

        // Shoulder: the flat step from the (wider) cone base up to the
        // (narrower) shank base, closing the cone/shank junction.
        vertices.push(vert(cone_base[i]));
        vertices.push(vert(shank_base[i]));
        vertices.push(vert(shank_base[j]));
        vertices.push(vert(cone_base[i]));
        vertices.push(vert(shank_base[j]));
        vertices.push(vert(cone_base[j]));

        // Shank side: the cylindrical body above the tip.
        vertices.push(vert(shank_base[i]));
        vertices.push(vert(shank_top_ring[i]));
        vertices.push(vert(shank_top_ring[j]));
        vertices.push(vert(shank_base[i]));
        vertices.push(vert(shank_top_ring[j]));
        vertices.push(vert(shank_base[j]));

        // Shank top cap.
        vertices.push(vert(shank_top_center));
        vertices.push(vert(shank_top_ring[j]));
        vertices.push(vert(shank_top_ring[i]));
    }
    vertices
}

/// The full precomputed scene: every resolved segment in emission
/// order, plus the vertex range each trace step contributed (empty for
/// steps with no geometry, e.g. a spindle command).
pub struct Scene {
    pub vertices: Vec<LineVertex>,
    pub step_ranges: Vec<Range<u32>>,
    /// Always includes the machine origin, even if the toolpath itself
    /// doesn't come near it - both so the camera (fit to these bounds)
    /// always frames the origin, and so a program with no motion at
    /// all still has a sane, non-degenerate box to fit.
    pub bounds_min: Position,
    pub bounds_max: Position,
}

impl Scene {
    pub fn build(entries: &[TraceEntry]) -> Self {
        let mut vertices = Vec::new();
        let mut step_ranges = Vec::with_capacity(entries.len());
        let mut bounds_min = Position {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let mut bounds_max = bounds_min;

        for entry in entries {
            let start = vertices.len() as u32;
            if let LineOutput::Motion(m) = &entry.output {
                push_motion(&mut vertices, m);
                for p in [m.start, m.target] {
                    bounds_min.x = bounds_min.x.min(p.x);
                    bounds_min.y = bounds_min.y.min(p.y);
                    bounds_min.z = bounds_min.z.min(p.z);
                    bounds_max.x = bounds_max.x.max(p.x);
                    bounds_max.y = bounds_max.y.max(p.y);
                    bounds_max.z = bounds_max.z.max(p.z);
                }
            }
            let end = vertices.len() as u32;
            step_ranges.push(start..end);
        }

        Self {
            vertices,
            step_ranges,
            bounds_min,
            bounds_max,
        }
    }

    /// A reasonable length for `axis_vertices` given this scene's
    /// extent - the largest of the (origin-inclusive) bounding box's
    /// three dimensions, so each axis arm reaches roughly as far as the
    /// toolpath does in whichever direction is largest.
    pub fn suggested_axis_length(&self) -> f64 {
        let dx = self.bounds_max.x - self.bounds_min.x;
        let dy = self.bounds_max.y - self.bounds_min.y;
        let dz = self.bounds_max.z - self.bounds_min.z;
        dx.max(dy).max(dz).max(1.0)
    }

    /// A reasonable radius for `sphere_vertices` (the origin marker)
    /// given this scene's extent - small enough not to dominate the
    /// view, but not so small it disappears at typical zoom levels.
    pub fn suggested_marker_radius(&self) -> f64 {
        (self.suggested_axis_length() * 0.01).max(0.4)
    }

    /// A reasonable radius for `spindle_marker_vertices` (the tool-tip
    /// marker) - smaller than the origin marker, since the cone+shank
    /// shape already reads as taller/bulkier than a sphere of the same
    /// radius (its total height is over 3x the radius).
    pub fn suggested_tool_marker_radius(&self) -> f64 {
        (self.suggested_axis_length() * 0.006).max(0.25)
    }
}
