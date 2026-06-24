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

use crate::state::Position;

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
    /// (center-relative-to-start) or R (radius form), not carried as
    /// raw G-code offsets. Resolving I/J/K/R into a concrete center is
    /// real, non-trivial geometry (see grblHAL's `mc_arc` for the
    /// reference computation) - deliberately left as a TODO for this
    /// scaffold rather than guessed at.
    pub center: Position,
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
