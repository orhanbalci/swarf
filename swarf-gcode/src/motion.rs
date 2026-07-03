//! `ResolvedMotionCommand` - Interface 2, the planner-facing boundary.
//!
//! Per our explicit design decision: keep this RICH rather than
//! pre-digested. A simple planner can ignore fields it doesn't need; a
//! sophisticated one (arc-aware, jerk-limited, multi-axis) has what it
//! needs without a round-trip back through the interpreter. This is the
//! same asymmetry argument we used earlier for not flattening arcs
//! before the planner sees them: discarding information at a boundary
//! is cheap to do and expensive to undo.
//!
//! CRITICAL INVARIANT: every value in here is an owned COPY, never a
//! reference back into `ModalState`. By the time a downstream consumer
//! looks at this command, the interpreter's live state has very likely
//! already moved on to a later line. This is the exact bug class
//! grblHAL's own comments warn about (reported work position derived
//! from parser state desyncing from what's actually executing, because
//! the parser runs ahead of the machine) - see `state.rs` module docs.

use crate::state::{Plane, Position};

/// The currently active motion mode - what a bare axis-word line (no
/// G-word) will do, per NIST's modal semantics ("if a G1 command is
/// given on one line, it will be executed again on the next line if one
/// or more axis words is available, unless an explicit command using
/// the axis words is given on that next line").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionMode {
    /// G0 - rapid positioning, not coordinated with feed rate.
    Rapid,
    /// G1 - linear motion at the active feed rate.
    Linear,
    /// G2 - clockwise arc at the active feed rate.
    ArcClockwise,
    /// G3 - counterclockwise arc at the active feed rate.
    ArcCounterclockwise,
    /// G81 - straight drilling canned cycle: rapid to the R plane, feed
    /// to the programmed depth, rapid retract.
    Drill,
    /// G82 - like `Drill`, but dwells at the bottom of the hole before
    /// retracting.
    DrillDwell,
    /// G83 - peck drilling: feeds to depth in `Q`-sized increments,
    /// fully retracting to the R plane between pecks to clear chips.
    PeckDrill,
    /// G85 - like `Drill`, but retracts at the active feed rate rather
    /// than rapid (boring, so the tool doesn't spring on the way out).
    BoreFeedOut,
    /// G86 - like `Drill`, but stops the spindle at the bottom of the
    /// hole before rapid-retracting. Does not restart the spindle.
    BoreSpindleStop,
    /// G89 - `DrillDwell` and `BoreFeedOut` combined: dwells at the
    /// bottom, then retracts at the active feed rate.
    BoreDwellFeedOut,
    /// G80 - motion mode cancelled; no motion mode is active. A bare
    /// axis-word line in this state is an error, not a silent no-op -
    /// see NIST's "cancel modal motion" semantics.
    None,
}

impl Default for MotionMode {
    fn default() -> Self {
        // NIST default modal state has no motion mode active until one
        // is explicitly commanded - matching G80's "cancelled" state,
        // NOT defaulting to G0 or G1. A bare axis-word line before any
        // G0/G1/G2/G3 has ever been seen is a real error to surface,
        // not something to silently interpret as a rapid move.
        MotionMode::None
    }
}

/// Arc-specific geometry, present only when `MotionMode` is
/// `ArcClockwise` or `ArcCounterclockwise`. Kept as a separate optional
/// payload rather than folding center/radius into every command, so a
/// straight-line move carries no unused arc fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArcGeometry {
    /// Arc center, in the same absolute machine-mm coordinates as
    /// `ResolvedMotionCommand::target` - already resolved from I/J/K
    /// (center-relative-to-start) or R (radius form) by
    /// `resolve_arc_center`, not carried as raw G-code offsets. The
    /// center's coordinate on the plane's linear (out-of-plane) axis is
    /// copied from `start`'s - an arc's center has no independent
    /// meaning on that axis, since the plane it lies in is
    /// perpendicular to it.
    pub center: Position,
}

/// Why a G2/G3 line's I/J/K/R data could not be turned into a concrete
/// arc center.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArcError {
    /// Neither I/J/K nor R appeared on a G2/G3 line - NIST requires
    /// exactly one of the two forms.
    MissingGeometry,
    /// R was given, but its magnitude is smaller than half the distance
    /// between start and end - no circle of that radius passes through
    /// both points.
    RadiusTooSmall,
    /// R was given, but start and end coincide. A radius alone can't
    /// determine a unique circle through a single point - use I/J/K
    /// (with the endpoint equal to the start point) for a full circle
    /// instead.
    CoincidentEndpoints,
}

/// Extract the two in-plane coordinates of `p` for `plane`, in a fixed
/// order chosen so that G17/G18/G19 correspond to the three cyclic
/// (handedness-preserving) permutations of (X, Y, Z): XY, YZ, ZX. Using
/// cyclic order rather than the more common-sounding "X then Z" for
/// G18 means CW/CCW and the I/J/K-offset math below work out identically
/// for all three planes, with no per-plane sign flip needed.
fn plane_uv(plane: Plane, p: Position) -> (f64, f64) {
    match plane {
        Plane::Xy => (p.x, p.y),
        Plane::Yz => (p.y, p.z),
        Plane::Zx => (p.z, p.x),
    }
}

/// The coordinate on the axis perpendicular to `plane` - the helical
/// axis for this arc.
fn plane_linear(plane: Plane, p: Position) -> f64 {
    match plane {
        Plane::Xy => p.z,
        Plane::Yz => p.x,
        Plane::Zx => p.y,
    }
}

fn uv_to_position(plane: Plane, u: f64, v: f64, linear: f64) -> Position {
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

/// Select the two of (I, J, K) relevant to `plane`, in the same (u, v)
/// order as `plane_uv` - I/J for XY, J/K for YZ, K/I for ZX.
fn ijk_uv(plane: Plane, ijk: (f64, f64, f64)) -> (f64, f64) {
    let (i, j, k) = ijk;
    match plane {
        Plane::Xy => (i, j),
        Plane::Yz => (j, k),
        Plane::Zx => (k, i),
    }
}

/// Resolve the absolute center of a G2/G3 arc from already-unit-and-mode
/// -resolved `start`/`end` points and the line's I/J/K and/or R data.
///
/// I/J/K, when present, are always incremental offsets from `start`
/// (NIST's G91.1 default - this crate does not implement the rarely
/// supported G90.1 absolute-IJK mode, matching grblHAL's own scope).
/// R, when present with no I/J/K, is resolved per NIST's convention: a
/// positive R selects the arc of <=180 degrees between start and end: a
/// negative R selects the arc of >=180 degrees. If both I/J/K and R
/// appear on one line (NIST does not define this case), I/J/K wins.
pub fn resolve_arc_center(
    plane: Plane,
    start: Position,
    end: Position,
    ijk: Option<(f64, f64, f64)>,
    r: Option<f64>,
    clockwise: bool,
) -> Result<Position, ArcError> {
    let (u0, v0) = plane_uv(plane, start);
    let (u1, v1) = plane_uv(plane, end);
    let linear = plane_linear(plane, start);

    let (cu, cv) = if let Some(ijk) = ijk {
        let (iu, iv) = ijk_uv(plane, ijk);
        (u0 + iu, v0 + iv)
    } else if let Some(r) = r {
        let dx = u1 - u0;
        let dy = v1 - v0;
        let d2 = dx * dx + dy * dy;
        if d2 == 0.0 {
            return Err(ArcError::CoincidentEndpoints);
        }
        let d = libm::sqrt(d2);
        let h2 = r * r - d2 / 4.0;
        if h2 < 0.0 {
            return Err(ArcError::RadiusTooSmall);
        }
        let h = libm::sqrt(h2);
        let mu = (u0 + u1) / 2.0;
        let mv = (v0 + v1) / 2.0;
        // +90-degree rotation of the start->end direction vector.
        let perp_u = -dy / d;
        let perp_v = dx / d;
        // Positive R picks the minor-arc center; negative R picks the
        // other (major-arc) one. CW vs CCW picks which of the two
        // perpendicular directions is "minor" to begin with. Verified
        // against a worked example: plane Xy, start (10,0), end (0,10),
        // r=10 gives center (0,0) for CCW and (10,10) for CW - the two
        // quarter-circle solutions through those points.
        let r_sign = if r >= 0.0 { 1.0 } else { -1.0 };
        let dir = if clockwise { -1.0 } else { 1.0 };
        let k = dir * r_sign * h;
        (mu + k * perp_u, mv + k * perp_v)
    } else {
        return Err(ArcError::MissingGeometry);
    };

    Ok(uv_to_position(plane, cu, cv, linear))
}

/// One fully resolved motion command - the output of interpreting one
/// line (or, for modal carry-forward, the effective output of a line
/// with only axis words). This is what crosses Interface 2 into a
/// motion planner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedMotionCommand {
    /// Position this move starts from. Included explicitly (not left
    /// for the planner to infer from "the previous command's target")
    /// because the planner should not have to reconstruct interpreter
    /// state to do its job - another instance of the "keep it rich"
    /// principle.
    pub start: Position,

    /// Position this move ends at, in absolute machine-mm coordinates,
    /// with the active work offset already applied. G90/G91 distance
    /// mode and G20/G21 units have already been resolved by this point
    /// - the planner never needs to know which mode produced this value.
    pub target: Position,

    pub motion_mode: MotionMode,

    /// Present only for arc moves. `None` for Rapid/Linear/None.
    pub arc: Option<ArcGeometry>,

    /// Feed rate in mm/min. Meaningless for Rapid moves (rapids move at
    /// the machine's maximum rate, not a programmed feed rate) - still
    /// included for uniformity, with the understanding that a planner
    /// should special-case `MotionMode::Rapid` rather than trusting this
    /// value in that case.
    pub feed_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_motion_mode_is_none_not_a_silent_guess() {
        // This is a deliberate, tested design choice: we do NOT want
        // MotionMode::default() to quietly become Rapid or Linear,
        // since that would mask a real "axis words before any motion
        // command was ever given" error as if it were valid G-code.
        assert_eq!(MotionMode::default(), MotionMode::None);
    }
}
