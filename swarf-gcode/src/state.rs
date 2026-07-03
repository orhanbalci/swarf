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

/// A selectable work coordinate system (G54-G59.3). NIST defines nine of
/// these; each carries its own offset, set by the host (see
/// `ModalState::set_coordinate_system_offset`) since this crate has no
/// persistent settings storage of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinateSystem {
    /// G54 - NIST default active system at power-on.
    G54,
    G55,
    G56,
    G57,
    G58,
    G59,
    G59_1,
    G59_2,
    G59_3,
}

impl CoordinateSystem {
    /// How many systems exist - the size of the offset table backing
    /// them (`ModalState`'s internal `coordinate_system_offsets`).
    pub const COUNT: usize = 9;

    const fn index(self) -> usize {
        match self {
            CoordinateSystem::G54 => 0,
            CoordinateSystem::G55 => 1,
            CoordinateSystem::G56 => 2,
            CoordinateSystem::G57 => 3,
            CoordinateSystem::G58 => 4,
            CoordinateSystem::G59 => 5,
            CoordinateSystem::G59_1 => 6,
            CoordinateSystem::G59_2 => 7,
            CoordinateSystem::G59_3 => 8,
        }
    }
}

impl Default for CoordinateSystem {
    fn default() -> Self {
        CoordinateSystem::G54
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

    /// Which work coordinate system (G54-G59.3) is currently selected.
    pub coordinate_system: CoordinateSystem,

    /// Per-system offset table backing `coordinate_system`. This crate
    /// has no persistent settings storage of its own - the host
    /// populates this table (typically once at startup, from whatever
    /// config/EEPROM backs its controller) via
    /// `set_coordinate_system_offset` before or during a parse. Private:
    /// read through `coordinate_system_offset`, which callers also use
    /// to inspect the currently active entry.
    coordinate_system_offsets: [Position; CoordinateSystem::COUNT],

    /// Additional origin shift set by G92, layered on top of whichever
    /// `coordinate_system_offsets` entry is active (see
    /// `Interpreter::resolve_target_from_values`). Reset to zero by
    /// G92.1. G92.2 (suspend) and G92.3 (restore) are not implemented -
    /// real scope decision, see `visitor` module docs.
    pub g92_offset: Position,

    /// G28's stored reference position, in absolute machine
    /// coordinates (no work/G92 offset applied) - what a bare "G28"
    /// rapids to. This crate has no persistent settings storage, so the
    /// host sets this directly (it's `pub`) from its own config, the
    /// same way it populates `coordinate_system_offsets`; G28.1 also
    /// updates it, from the current machine position.
    pub g28_position: Position,

    /// G30's stored reference position - same shape as `g28_position`,
    /// set by the host or by G30.1, consulted by G30.
    pub g30_position: Position,

    /// Feed rate in mm/min, already unit-converted regardless of the
    /// active Units mode at the time F was parsed.
    pub feed_rate: f64,

    /// Spindle speed in RPM, modal like feed rate: an S word persists
    /// across lines and is only consumed (turned into a
    /// `command::SpindleCommand`) when M3/M4 actually runs.
    pub spindle_speed: f64,

    /// The tool number most recently selected by a T word - modal, but
    /// distinct from "the tool that is actually loaded": selecting a
    /// tool (T) and changing to it (M6) are two separate NIST actions.
    /// `None` until the first T word is ever seen.
    pub selected_tool: Option<u32>,

    /// Sticky parameters for the canned drilling cycles (G81-G89) - see
    /// `CannedCycleParams`.
    pub canned_cycle: CannedCycleParams,

    /// G98 (true, NIST default) / G99 (false): whether a canned cycle
    /// retracts to the higher of the R plane and the position Z was at
    /// before the cycle started (G98), or always just to the R plane
    /// (G99, faster when the operator knows R already clears
    /// everything).
    pub canned_cycle_return_to_initial_z: bool,
}

/// Sticky parameters for the canned drilling cycles (G81-G89). NIST
/// specifies these as modal in their own right, separate from the
/// G8x motion mode itself: a canned-cycle line (or a bare axis-word
/// repeat of one) that omits Z/R/Q/P reuses whichever value was last
/// given. All distances are absolute machine-mm (already resolved
/// through the active work offset and distance mode at the moment they
/// were last given - see `Interpreter::execute_canned_cycle`), except
/// `p` which is a plain duration in seconds. `None` until first given;
/// consuming a `None` value where NIST requires one is
/// `InterpretError::CannedCycleMissingParameter`, never a silent
/// default - the same "no correctness footguns" principle as
/// everywhere else in this crate.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CannedCycleParams {
    /// Final depth (the bottom of the hole).
    pub z: Option<f64>,
    /// Retract plane - rapid down to this before feeding, and the
    /// default retract target after cutting.
    pub r: Option<f64>,
    /// Peck increment (G83 only).
    pub q: Option<f64>,
    /// Dwell duration in seconds at the bottom of the hole (G82/G89
    /// only).
    pub p: Option<f64>,
}

impl Default for ModalState {
    fn default() -> Self {
        Self {
            position: Position::default(),
            motion_mode: super::motion::MotionMode::default(),
            plane: Plane::default(),
            units: Units::default(),
            distance_mode: DistanceMode::default(),
            coordinate_system: CoordinateSystem::default(),
            coordinate_system_offsets: [Position::default(); CoordinateSystem::COUNT],
            g92_offset: Position::default(),
            g28_position: Position::default(),
            g30_position: Position::default(),
            feed_rate: 0.0,
            spindle_speed: 0.0,
            selected_tool: None,
            canned_cycle: CannedCycleParams::default(),
            canned_cycle_return_to_initial_z: true,
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

    /// Read the offset table entry for `system` - `Position::default()`
    /// (identity) until the host sets one with
    /// `set_coordinate_system_offset`.
    pub fn coordinate_system_offset(&self, system: CoordinateSystem) -> Position {
        self.coordinate_system_offsets[system.index()]
    }

    /// Populate (or update) the offset table entry for `system`. Meant
    /// to be called by the host - typically once at startup for each of
    /// the 9 systems from its own settings storage, though nothing
    /// prevents calling it again mid-parse if a controller supports
    /// redefining a work offset live (NIST's G10 L2/L20 do this from
    /// within a program; this crate does not implement those codes yet,
    /// so today this is exclusively a host-driven call).
    pub fn set_coordinate_system_offset(&mut self, system: CoordinateSystem, offset: Position) {
        self.coordinate_system_offsets[system.index()] = offset;
    }

    /// The total offset applied to absolute-mode moves: the active
    /// work coordinate system's table entry plus the G92 shift.
    pub fn active_offset(&self) -> Position {
        let base = self.coordinate_system_offset(self.coordinate_system);
        Position {
            x: base.x + self.g92_offset.x,
            y: base.y + self.g92_offset.y,
            z: base.z + self.g92_offset.z,
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
