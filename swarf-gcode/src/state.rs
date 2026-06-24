//! `ModalState` - the persistent, motion-scoped interpreter state.
//!
//! This is our Interface 1: deliberately narrower than grblHAL's full
//! `gc_state` (no spindle RPM, no coolant flags, no tool table) per our
//! explicit decision to scope this to exactly what a motion planner
//! needs. Everything here mutates in place, line by line, and NEVER
//! gets rolled back mid-line - this mirrors grblHAL's `gc_state` being
//! a single global struct, not a stack of per-line snapshots.
//!
//! IMPORTANT: values read out of `ModalState` to build a
//! `ResolvedMotionCommand` must be copied, not referenced - by the time
//! a downstream consumer (planner, buffer) looks at a resolved command
//! again, this struct may have moved on several more lines. This is the
//! exact hazard grblHAL's own comments warn about (work position
//! historically derived from parser state could desync from what was
//! actually executing, since the parser runs ahead of the machine).

/// Active plane for arc interpretation (G17/G18/G19).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plane {
    /// G17 - XY plane (Z is the linear/helical axis). NIST default.
    Xy,
    /// G18 - ZX plane (Y is the linear/helical axis).
    Zx,
    /// G19 - YZ plane (X is the linear/helical axis).
    Yz,
}

impl Default for Plane {
    fn default() -> Self {
        // NIST RS274NGC default modal state is G17 (XY plane).
        Plane::Xy
    }
}

/// Length units (G20/G21). Internally we always store position in
/// millimeters in `ModalState.position`, regardless of the active
/// units mode - this field exists ONLY to interpret incoming literals
/// correctly at the moment they're parsed, not to change storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Units {
    /// G20 - inches. Incoming literals get multiplied by 25.4 on entry.
    Inches,
    /// G21 - millimeters. NIST default... actually NIST's documented
    /// default is inches (G20) for historical reasons, but essentially
    /// every real controller defaults to mm (G21) instead. We follow
    /// the de facto convention since this crate targets real machines,
    /// not strict NIST default-state pedantry. Senders should still
    /// send an explicit G20/G21 - relying on either default is unsafe
    /// practice regardless of what we pick here.
    Millimeters,
}

impl Default for Units {
    fn default() -> Self {
        Units::Millimeters
    }
}

/// Distance mode (G90/G91): are axis words absolute positions or
/// incremental deltas from the current position?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMode {
    /// G90 - absolute. NIST default.
    Absolute,
    /// G91 - incremental.
    Incremental,
}

impl Default for DistanceMode {
    fn default() -> Self {
        DistanceMode::Absolute
    }
}

/// A 3D position, always stored in millimeters internally regardless of
/// the active Units mode. Kept deliberately minimal (no operator
/// overloads, no external geometry crate dependency) since this crate's
/// only job is to produce these values, not to do path math with them -
/// that's the downstream planner's job, per our layering decision.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Position {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    // NOTE: rotary axes (A/B/C) intentionally omitted from this first
    // pass - real scope decision, not an oversight. Adding them later
    // is a additive, non-breaking change to this struct.
}

/// The full motion-relevant persistent interpreter state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModalState {
    /// Current tool tip position, machine-relative, after all active
    /// offsets are applied. This is the position the NEXT move starts
    /// from - i.e. it only updates once a move actually resolves and is
    /// committed (see `visitor::Interpreter::end_line`), not eagerly.
    pub position: Position,

    /// Which motion mode is currently active. This is the field that
    /// makes G-code "modal" in the sense NIST describes: a bare `X10`
    /// line with no G-word reuses whatever this was last set to.
    pub motion_mode: super::motion::MotionMode,

    pub plane: Plane,
    pub units: Units,
    pub distance_mode: DistanceMode,

    /// Active work coordinate system offset (G54-G59.3), applied to
    /// every absolute-mode move. Stored as a plain offset vector added
    /// to machine coordinates - NOT as an index into a settings table,
    /// since this crate has no concept of persistent machine settings
    /// storage; callers are responsible for loading the right offset
    /// into this field when a G54-G59 line is interpreted (the
    /// interpreter records WHICH system is selected, but resolving that
    /// selection to an actual offset value is a host-provided lookup -
    /// see `visitor::WorkOffsetProvider`).
    pub work_offset: Position,

    /// Feed rate in mm/min, already unit-converted regardless of the
    /// active Units mode at the time F was parsed.
    pub feed_rate: f64,
}

impl Default for ModalState {
    fn default() -> Self {
        Self {
            position: Position::default(),
            motion_mode: super::motion::MotionMode::default(),
            plane: Plane::default(),
            units: Units::default(),
            distance_mode: DistanceMode::default(),
            work_offset: Position::default(),
            feed_rate: 0.0,
        }
    }
}

impl ModalState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert a raw literal value (as it appeared in the source) into
    /// millimeters, using the currently active Units mode. Call this at
    /// the moment a literal is read, not later - Units is itself modal
    /// and may change on a subsequent line.
    pub fn to_mm(&self, raw: f64) -> f64 {
        match self.units {
            Units::Millimeters => raw,
            Units::Inches => raw * 25.4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_matches_common_controller_conventions() {
        let state = ModalState::new();
        assert_eq!(state.plane, Plane::Xy);
        assert_eq!(state.units, Units::Millimeters);
        assert_eq!(state.distance_mode, DistanceMode::Absolute);
        assert_eq!(state.position, Position::default());
    }

    #[test]
    fn to_mm_converts_inches_correctly() {
        let mut state = ModalState::new();
        state.units = Units::Inches;
        assert!((state.to_mm(1.0) - 25.4).abs() < 1e-9);
    }

    #[test]
    fn to_mm_is_identity_in_millimeter_mode() {
        let state = ModalState::new();
        assert_eq!(state.units, Units::Millimeters);
        assert_eq!(state.to_mm(42.0), 42.0);
    }
}
