//! Modal group classification - Interface 1's authoritative shape.
//!
//! Sourced from two places we verified directly in this investigation:
//!   - NIST RS274NGC Interpreter Version 3, Table 4 ("Modal Groups"),
//!     which defines modal groups as: "a group of g-code commands that
//!     are mutually exclusive, or cannot exist on the same line, because
//!     they each toggle a state or execute a unique motion."
//!   - grblHAL's gcode.h / gcode.c, whose modal_groups_t bitfield comment
//!     explicitly cites "the NIST RS274-NGC v3 g-code standard" as its
//!     source and is "similar/identical to other g-code interpreters by
//!     manufacturers (Haas, Fanuc, Mazak, etc.)" - i.e. this is the
//!     genuinely standard classification, not one interpreter's quirk.
//!
//! NOTE ON SCOPE: this enumerates the groups relevant to MOTION (per our
//! explicit decision to scope Interface 1 to motion-relevant state, not
//! the full machine state - spindle/coolant/tool-table groups are
//! tracked here only enough to detect line-level conflicts; their actual
//! *effects* are out of scope for this crate, which only resolves motion).

/// Which modal group a given G or M code belongs to. Two codes from the
/// SAME group may not appear on the same line (NIST: "a line may have
/// only one word" per group, with motion explicitly called out as
/// modal - once set, stays active until explicitly changed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModalGroup {
    /// G4, G10, G28, G28.1, G30, G30.1, G53, G92, G92.1-G92.3 - non-modal.
    /// NOTE: non-modal codes are a special case - despite the name, NIST
    /// groups them together for "only one per line" purposes, but they
    /// do NOT persist across lines the way true modal groups do.
    NonModal,
    /// G0, G1, G2, G3, G38.2-G38.5, G80, G81-G89 - motion mode. The
    /// single most important group for this crate: this is what carries
    /// forward from line to line per NIST's modal semantics.
    Motion,
    /// G17, G18, G19 - plane selection (XY, ZX, YZ).
    Plane,
    /// G90, G91 - distance mode (absolute / incremental).
    DistanceMode,
    /// G91.1 - arc IJK distance mode.
    ArcDistanceMode,
    /// G93, G94 - feed rate mode (inverse time / units-per-minute).
    FeedRateMode,
    /// G20, G21 - units (inch / mm).
    Units,
    /// G54-G59(.x) - coordinate system selection (work offset).
    CoordinateSystem,
    /// G98, G99 - canned cycle return mode. Tracked for conflict
    /// detection; canned cycles themselves are out of scope for now.
    CannedCycleReturnMode,
    /// M0, M1, M2, M30 - program stopping.
    ProgramStopping,
    /// M3, M4, M5 - spindle turning. Tracked for conflict detection only;
    /// spindle state itself is out of this crate's motion-only scope.
    SpindleTurning,
    /// M7, M8, M9 - coolant control. Same scope note as spindle.
    CoolantControl,
}

/// A small fixed-size set of modal groups seen so far on the current
/// line, used purely for O(1)-ish conflict detection without allocation.
/// Backed by a bitmask over a fixed, small set of known groups - this is
/// deliberately NOT a general Set<T>, to keep the no_std story trivial.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModalGroupSet {
    mask: u16,
}

impl ModalGroupSet {
    pub const fn new() -> Self {
        Self { mask: 0 }
    }

    fn bit(group: ModalGroup) -> u16 {
        // Stable small integer per variant. Using a match rather than
        // `as u16` on the enum directly keeps this independent of enum
        // discriminant layout, and keeps the non_exhaustive enum from
        // silently breaking this if variants are reordered upstream.
        match group {
            ModalGroup::NonModal => 1 << 0,
            ModalGroup::Motion => 1 << 1,
            ModalGroup::Plane => 1 << 2,
            ModalGroup::DistanceMode => 1 << 3,
            ModalGroup::ArcDistanceMode => 1 << 4,
            ModalGroup::FeedRateMode => 1 << 5,
            ModalGroup::Units => 1 << 6,
            ModalGroup::CoordinateSystem => 1 << 7,
            ModalGroup::CannedCycleReturnMode => 1 << 8,
            ModalGroup::ProgramStopping => 1 << 9,
            ModalGroup::SpindleTurning => 1 << 10,
            ModalGroup::CoolantControl => 1 << 11,
        }
    }

    /// Returns true if this group has already been seen on the current
    /// line - i.e. adding `group` again would be a NIST modal-group
    /// conflict ("two M words from the same modal group may not appear
    /// on the same line"; the same rule is stated for G-word groups).
    pub fn contains(&self, group: ModalGroup) -> bool {
        self.mask & Self::bit(group) != 0
    }

    /// Marks `group` as seen on the current line. Call `contains` first
    /// if you need to detect the conflict before recording it.
    pub fn insert(&mut self, group: ModalGroup) {
        self.mask |= Self::bit(group);
    }

    /// Clears all seen groups - call once per new line, in `start_block`.
    pub fn clear(&mut self) {
        self.mask = 0;
    }
}

/// Classify a G-code number into its modal group, per NIST Table 4.
///
/// Returns `None` for G-codes not yet covered by this crate's scope
/// (canned cycles G81-G89, cutter compensation, tool length offset,
/// scaling, etc.) - callers should treat `None` as "recognized as a
/// valid G-word number-wise, but this interpreter does not yet implement
/// its semantics," which is a different, milder condition than an
/// unrecognized/invalid number entirely.
///
/// `minor` is the NIST-style decimal suffix (e.g. the `1` in `G92.1`),
/// matching `gcode::core::Number::minor()`'s `Option<NonZeroU32>` shape
/// collapsed to a plain `Option<u32>` here to keep this module
/// independent of the `gcode` crate's types.
pub fn classify_general_code(major: u32, minor: Option<u32>) -> Option<ModalGroup> {
    use ModalGroup::*;
    Some(match (major, minor) {
        // --- Non-modal (Group 0) ---
        (4, None) => NonModal, // G4 dwell
        (10, _) => NonModal,   // G10 set coordinate data
        (28, None) | (28, Some(1)) => NonModal,
        (30, None) | (30, Some(1)) => NonModal,
        (53, None) => NonModal,
        (92, None) => NonModal,
        (92, Some(1)) | (92, Some(2)) | (92, Some(3)) => NonModal,

        // --- Motion (Group 1) ---
        (0, None) => Motion, // rapid
        (1, None) => Motion, // linear feed
        (2, None) => Motion, // CW arc
        (3, None) => Motion, // CCW arc
        (38, Some(2)) | (38, Some(3)) | (38, Some(4)) | (38, Some(5)) => Motion, // probing
        (80, None) => Motion, // cancel motion mode

        // --- Plane selection (Group 2) ---
        (17, None) => Plane,
        (18, None) => Plane,
        (19, None) => Plane,

        // --- Distance mode (Group 3) ---
        (90, None) => DistanceMode,
        (91, None) => DistanceMode,

        // --- Arc IJK distance mode (Group 4) ---
        (91, Some(1)) => ArcDistanceMode,

        // --- Feed rate mode (Group 5) ---
        (93, None) => FeedRateMode,
        (94, None) => FeedRateMode,

        // --- Units (Group 6) ---
        (20, None) => Units,
        (21, None) => Units,

        // --- Coordinate system selection (Group 12) ---
        (54, None) | (55, None) | (56, None) | (57, None) | (58, None) | (59, None) => {
            CoordinateSystem
        }
        (59, Some(1)) | (59, Some(2)) | (59, Some(3)) => CoordinateSystem,

        // --- Canned cycle return mode (Group 10) ---
        (98, None) => CannedCycleReturnMode,
        (99, None) => CannedCycleReturnMode,

        _ => return None,
    })
}

/// Classify an M-code number into its modal group, per NIST Table 4.
/// Same `None`-means-"not yet implemented" convention as above.
pub fn classify_miscellaneous_code(major: u32, minor: Option<u32>) -> Option<ModalGroup> {
    use ModalGroup::*;
    Some(match (major, minor) {
        (0, None) | (1, None) | (2, None) | (30, None) => ProgramStopping,
        (3, None) | (4, None) | (5, None) => SpindleTurning,
        (7, None) | (8, None) | (9, None) => CoolantControl,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motion_codes_classify_correctly() {
        assert_eq!(classify_general_code(0, None), Some(ModalGroup::Motion));
        assert_eq!(classify_general_code(1, None), Some(ModalGroup::Motion));
        assert_eq!(classify_general_code(2, None), Some(ModalGroup::Motion));
        assert_eq!(classify_general_code(3, None), Some(ModalGroup::Motion));
    }

    #[test]
    fn plane_and_motion_are_different_groups() {
        // G1 and G17 must NOT conflict - they're different groups and
        // legitimately coexist on one line (e.g. "G17 G1 X10").
        assert_ne!(
            classify_general_code(1, None),
            classify_general_code(17, None)
        );
    }

    #[test]
    fn two_motion_codes_are_same_group_and_would_conflict() {
        // G0 and G1 ARE the same group - "G0 G1 X10" is illegal per NIST,
        // since two G-words from modal group 1 cannot coexist on a line.
        assert_eq!(
            classify_general_code(0, None),
            classify_general_code(1, None)
        );
    }

    #[test]
    fn modal_group_set_detects_conflict() {
        let mut seen = ModalGroupSet::new();
        assert!(!seen.contains(ModalGroup::Motion));
        seen.insert(ModalGroup::Motion);
        assert!(seen.contains(ModalGroup::Motion));
        // A second motion-group code on the same line is now detectable
        // as a conflict by the caller checking `contains` before insert.
    }

    #[test]
    fn modal_group_set_clears_between_lines() {
        let mut seen = ModalGroupSet::new();
        seen.insert(ModalGroup::Motion);
        seen.insert(ModalGroup::Plane);
        seen.clear();
        assert!(!seen.contains(ModalGroup::Motion));
        assert!(!seen.contains(ModalGroup::Plane));
    }

    #[test]
    fn unrecognized_g_code_returns_none_not_panic() {
        // G81 (canned drill cycle) is real G-code but out of this
        // crate's current scope - must return None, not panic or guess.
        assert_eq!(classify_general_code(81, None), None);
    }

    #[test]
    fn distance_mode_codes_classify_correctly() {
        assert_eq!(
            classify_general_code(90, None),
            Some(ModalGroup::DistanceMode)
        );
        assert_eq!(
            classify_general_code(91, None),
            Some(ModalGroup::DistanceMode)
        );
        // G91.1 (arc IJK distance mode) is a DIFFERENT group from G91
        // (distance mode) despite the shared major number - the decimal
        // suffix changes meaning entirely per NIST.
        assert_eq!(
            classify_general_code(91, Some(1)),
            Some(ModalGroup::ArcDistanceMode)
        );
        assert_ne!(
            classify_general_code(91, None),
            classify_general_code(91, Some(1))
        );
    }

    #[test]
    fn spindle_m_codes_classify_correctly() {
        assert_eq!(
            classify_miscellaneous_code(3, None),
            Some(ModalGroup::SpindleTurning)
        );
        assert_eq!(
            classify_miscellaneous_code(5, None),
            Some(ModalGroup::SpindleTurning)
        );
    }
}
