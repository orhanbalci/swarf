//! Turns a recorded interpretation trace into renderable line-segment
//! geometry: one big vertex buffer (line list) built once at load time,
//! plus the vertex range each step contributed, so the viewer can slice
//! out "path so far" / "current step" every frame without
//! re-tessellating anything.

use std::ops::Range;

use swarf_gcode::{LineOutput, MotionMode, Position, ResolvedMotionCommand};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
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

fn to_vertex(p: Position, color: [f32; 3]) -> Vertex {
    Vertex {
        position: [p.x as f32, p.y as f32, p.z as f32],
        color,
    }
}

fn push_segment(vertices: &mut Vec<Vertex>, a: Position, b: Position, color: [f32; 3]) {
    vertices.push(to_vertex(a, color));
    vertices.push(to_vertex(b, color));
}

/// Tessellate one resolved move into line-list vertices (2 per segment,
/// no shared-index strip - simplest thing that's portable across
/// backends, and the vertex count involved is trivial for a toolpath).
fn push_motion(vertices: &mut Vec<Vertex>, m: &ResolvedMotionCommand) {
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

/// The full precomputed scene: every resolved segment in emission
/// order, plus the vertex range each trace step contributed (empty for
/// steps with no geometry, e.g. a spindle command).
pub struct Scene {
    pub vertices: Vec<Vertex>,
    pub step_ranges: Vec<Range<u32>>,
    pub bounds_min: Position,
    pub bounds_max: Position,
}

impl Scene {
    pub fn build(entries: &[TraceEntry]) -> Self {
        let mut vertices = Vec::new();
        let mut step_ranges = Vec::with_capacity(entries.len());
        let mut bounds_min = Position {
            x: f64::INFINITY,
            y: f64::INFINITY,
            z: f64::INFINITY,
        };
        let mut bounds_max = Position {
            x: f64::NEG_INFINITY,
            y: f64::NEG_INFINITY,
            z: f64::NEG_INFINITY,
        };

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

        if !bounds_min.x.is_finite() {
            // No motion at all (e.g. a program with only M-codes) - fall
            // back to a small default box so the camera has something
            // sane to frame.
            bounds_min = Position {
                x: -10.0,
                y: -10.0,
                z: -10.0,
            };
            bounds_max = Position {
                x: 10.0,
                y: 10.0,
                z: 10.0,
            };
        }

        Self {
            vertices,
            step_ranges,
            bounds_min,
            bounds_max,
        }
    }
}
