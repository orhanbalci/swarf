//! Arc tessellation: turns one arc move (start/target/center/direction)
//! into a sequence of short linear segments, each becoming its own
//! `Block` (see `block.rs`).
//!
//! This crate has no notion of "G2/G3" or any other G-code-specific
//! idea - `swarf-bridge` is the layer that recognizes a G-code arc and
//! calls `Planner::push_arc` with plain geometry. Segment length is
//! derived from the exact chord/sagitta relationship (not a small-angle
//! approximation, and not a fixed segment count): for a circle of
//! radius `r`, a chord of length `c` deviates from the true arc by a
//! "sagitta" `s = r - sqrt(r² - (c/2)²)`. Solving for `c` given a
//! maximum allowed `s` (`MachineLimits::arc_tolerance`, grblHAL's `$12`)
//! gives `c = 2·sqrt(s·(2r - s))` - the same relationship grblHAL's own
//! arc tessellation uses. A tighter tolerance or smaller radius means
//! more, shorter segments for the same sweep.
//!
//! No heap allocation: `ArcSegments` computes each segment's endpoint on
//! demand from the arc's closed-form parametrization, not a `Vec` of
//! precomputed points.

use crate::position::Position;

/// Which of the three axis-aligned planes an arc lies in - this crate's
/// own local notion, not tied to any particular G-code's G17/G18/G19
/// vocabulary (that mapping, if a caller needs it, is `swarf-bridge`'s
/// job).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Plane {
    Xy,
    Yz,
    Zx,
}

/// Recovers which plane an arc lies in from its own geometry: whichever
/// axis has `center` closest to `start` on that axis is the out-of-plane
/// (helical) one - callers only ever supply a resolved absolute center,
/// never raw offsets, so this is always well-defined.
fn detect_plane(start: Position, center: Position) -> Plane {
    let dx = libm::fabs(center.x - start.x);
    let dy = libm::fabs(center.y - start.y);
    let dz = libm::fabs(center.z - start.z);
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
        Plane::Xy => Position { x: u, y: v, z: linear },
        Plane::Yz => Position { x: linear, y: u, z: v },
        Plane::Zx => Position { x: v, y: linear, z: u },
    }
}

/// The angle/center/sweep parameters of one arc move - the closed-form
/// parametrization `ArcSegments` samples at each segment boundary, and
/// `queue.rs`/`block.rs` use to compute exact arc length.
#[derive(Debug, Clone, Copy)]
pub struct ArcParams {
    plane: Plane,
    center_uv: (f64, f64),
    radius: f64,
    angle0: f64,
    sweep: f64,
    linear0: f64,
    linear1: f64,
}

impl ArcParams {
    /// `clockwise` matches the caller's own notion of arc direction
    /// (G-code's G2 = clockwise, G3 = counterclockwise, if that's the
    /// source - but this crate doesn't care what produced it).
    /// `start == target` is treated as an explicit full-circle request
    /// (matching G-code's own convention for a full-circle arc), not a
    /// zero-length move.
    pub fn new(start: Position, target: Position, center: Position, clockwise: bool) -> Self {
        let plane = detect_plane(start, center);
        let (cu, cv) = in_plane(center, plane);
        let (u0, v0) = in_plane(start, plane);
        let (u1, v1) = in_plane(target, plane);
        let (du0, dv0) = (u0 - cu, v0 - cv);
        let (du1, dv1) = (u1 - cu, v1 - cv);
        let radius = libm::sqrt(du0 * du0 + dv0 * dv0);

        let angle0 = libm::atan2(dv0, du0);
        let angle1 = libm::atan2(dv1, du1);

        let full_circle = (start.x - target.x).abs() < 1e-9
            && (start.y - target.y).abs() < 1e-9
            && (start.z - target.z).abs() < 1e-9;

        const TAU: f64 = core::f64::consts::TAU;
        let sweep = if full_circle {
            if clockwise {
                -TAU
            } else {
                TAU
            }
        } else if clockwise {
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

        Self {
            plane,
            center_uv: (cu, cv),
            radius,
            angle0,
            sweep,
            linear0: linear_coord(start, plane),
            linear1: linear_coord(target, plane),
        }
    }

    /// A point at fraction `t` (0.0 = start, 1.0 = target) along the arc.
    pub fn point_at(&self, t: f64) -> Position {
        let angle = self.angle0 + self.sweep * t;
        let (cu, cv) = self.center_uv;
        let (u, v) = (
            cu + self.radius * libm::cos(angle),
            cv + self.radius * libm::sin(angle),
        );
        let linear = self.linear0 + (self.linear1 - self.linear0) * t;
        from_plane(u, v, linear, self.plane)
    }

    /// Exact arc length in mm (ignores the linear/helical axis's tiny
    /// contribution to path length, same simplification a real
    /// controller's feed-rate accounting makes).
    pub fn length(&self) -> f64 {
        self.radius * libm::fabs(self.sweep)
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }

    pub fn sweep(&self) -> f64 {
        self.sweep
    }
}

/// Chord length for a circle of `radius` such that the chord deviates
/// from the true arc by at most `tolerance` (the sagitta) - see this
/// module's doc comment for the derivation. Falls back to a length of
/// `radius` itself (i.e. roughly a 1-radian segment) if `tolerance` is
/// degenerate (non-positive, or as large as/larger than the diameter,
/// which would make the exact formula's `sqrt` argument non-positive) -
/// callers still get a finite, reasonable segment count rather than a
/// division by zero or a NaN propagating into the block queue.
fn chord_length(radius: f64, tolerance: f64) -> f64 {
    if radius <= 0.0 {
        return f64::INFINITY;
    }
    let arg = tolerance * (2.0 * radius - tolerance);
    if tolerance <= 0.0 || arg <= 0.0 {
        return radius;
    }
    2.0 * libm::sqrt(arg)
}

/// How many segments to split an arc of these `params` into, given
/// `tolerance` (mm, `MachineLimits::arc_tolerance`) - always at least 2,
/// matching the fact that even a very gentle arc needs at least a start
/// and end segment to read as curved rather than a single straight jump.
pub fn segment_count(params: &ArcParams, tolerance: f64) -> usize {
    let arc_length = params.length();
    if arc_length <= 1e-9 {
        return 1;
    }
    let chord = chord_length(params.radius(), tolerance);
    let count = libm::ceil(arc_length / chord.max(1e-9)) as usize;
    count.max(2)
}

/// Iterates the `(start, end)` endpoints of each tessellated segment of
/// an arc move, in order - no heap allocation, each segment's endpoint
/// is computed from `ArcParams::point_at` on demand.
pub struct ArcSegments {
    params: ArcParams,
    segments: usize,
    next_index: usize,
    prev: Position,
}

impl ArcSegments {
    pub fn new(start: Position, target: Position, center: Position, clockwise: bool, tolerance: f64) -> Self {
        let params = ArcParams::new(start, target, center, clockwise);
        let segments = segment_count(&params, tolerance);
        Self {
            params,
            segments,
            next_index: 1,
            prev: start,
        }
    }
}

impl Iterator for ArcSegments {
    /// One tessellated segment's `(start, end)` endpoints.
    type Item = (Position, Position);

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index > self.segments {
            return None;
        }
        let t = self.next_index as f64 / self.segments as f64;
        let point = self.params.point_at(t);
        let segment = (self.prev, point);
        self.prev = point;
        self.next_index += 1;
        Some(segment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quarter_circle() -> (Position, Position, Position) {
        // Start (10, 0), center (0, 0), end (0, 10): a CCW quarter circle
        // of radius 10 in the XY plane.
        let start = Position { x: 10.0, y: 0.0, z: 0.0 };
        let target = Position { x: 0.0, y: 10.0, z: 0.0 };
        let center = Position { x: 0.0, y: 0.0, z: 0.0 };
        (start, target, center)
    }

    #[test]
    fn arc_params_recovers_radius_and_quarter_turn_sweep() {
        let (start, target, center) = quarter_circle();
        let params = ArcParams::new(start, target, center, false);
        assert!((params.radius() - 10.0).abs() < 1e-9);
        assert!((params.sweep() - core::f64::consts::FRAC_PI_2).abs() < 1e-9);
    }

    #[test]
    fn point_at_endpoints_matches_start_and_target() {
        let (start, target, center) = quarter_circle();
        let params = ArcParams::new(start, target, center, false);
        let p0 = params.point_at(0.0);
        let p1 = params.point_at(1.0);
        assert!((p0.x - start.x).abs() < 1e-9 && (p0.y - start.y).abs() < 1e-9);
        assert!((p1.x - target.x).abs() < 1e-9 && (p1.y - target.y).abs() < 1e-9);
    }

    #[test]
    fn tighter_tolerance_produces_more_segments() {
        let (start, target, center) = quarter_circle();
        let params = ArcParams::new(start, target, center, false);
        let loose = segment_count(&params, 0.1);
        let tight = segment_count(&params, 0.001);
        assert!(tight > loose);
    }

    #[test]
    fn tessellated_segments_stay_within_chord_tolerance() {
        let (start, target, center) = quarter_circle();
        let tolerance = 0.01;
        let params = ArcParams::new(start, target, center, false);
        let segments = ArcSegments::new(start, target, center, false, tolerance);
        for (a, b) in segments {
            // Sagitta of the midpoint of chord (a, b) against the true
            // circle centered at the origin with radius 10.
            let mid = Position {
                x: (a.x + b.x) / 2.0,
                y: (a.y + b.y) / 2.0,
                z: 0.0,
            };
            let mid_radius = libm::sqrt(mid.x * mid.x + mid.y * mid.y);
            let sagitta = params.radius() - mid_radius;
            assert!(sagitta < tolerance * 1.01, "sagitta {sagitta} exceeded tolerance {tolerance}");
        }
    }

    #[test]
    fn full_circle_tessellates_into_multiple_segments() {
        let start = Position { x: 10.0, y: 0.0, z: 0.0 };
        let center = Position { x: 0.0, y: 0.0, z: 0.0 };
        let params = ArcParams::new(start, start, center, true);
        assert!((params.sweep() + core::f64::consts::TAU).abs() < 1e-9);
        let count = ArcSegments::new(start, start, center, true, 0.01).count();
        assert!(count > 4);
    }
}
