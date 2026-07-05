//! `MachineLimits` - the host-supplied machine-capability facts this
//! planner needs but can never invent on its own.
//!
//! `swarf-gcode` draws the same boundary for work-coordinate offsets
//! (`ModalState::set_coordinate_system_offset`): the interpreter/planner
//! consumes machine facts, it doesn't guess them. Per-axis acceleration
//! and velocity limits, and the junction-deviation/arc-tolerance tuning
//! constants, are exactly that kind of fact - they describe what a real
//! machine can physically do, not anything derivable from the G-code
//! being run. grblHAL's equivalent (`settings_t`, `$110`/`$120`/`$11`/
//! `$12`) is loaded once from NVS/EEPROM at boot into one in-RAM struct
//! and read directly by the planner with no further indirection; we take
//! the same shape, minus the persistence/protocol layer (that's the
//! host's job, not this crate's).
//!
//! Axes are array-indexed with a const generic count (mirrors grblHAL's
//! `N_AXIS`, which varies by board/driver - a basic mill is 3, a lathe
//! or rotary-equipped machine is more) rather than fixed X/Y/Z fields, so
//! this doesn't foreclose non-mill configurations later.
//!
//! NOTE: `swarf_gcode::Position` is hardcoded to exactly X/Y/Z today (no
//! rotary axes at the interpreter level yet), so every OTHER type in
//! this crate (`Block`, `BlockQueue`, `Planner`) concretely uses
//! `MachineLimits<3>`, not an arbitrary `N` - this type stays generic
//! for when that changes upstream, but nothing downstream can honestly
//! use `N != 3` until `Position` itself grows more axes.

/// One axis's physical capability ceiling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisLimits {
    /// Maximum velocity, in mm/min - matches G-code's own `F` word
    /// convention (grblHAL's `$110`/`$111`/`$112` etc.), so a host
    /// filling this in from real machine settings doesn't need to
    /// convert units first.
    pub max_velocity: f64,
    /// Maximum acceleration, in mm/s² (grblHAL's `$120`/`$121`/`$122`
    /// convention). NOTE the unit mismatch with `max_velocity` (mm/min
    /// vs mm/s) is deliberate - it matches how the numbers are actually
    /// specified on a real machine and in G-code; `block.rs` converts
    /// to one consistent internal unit (mm/s) when it builds a `Block`,
    /// exactly once, in one place - not scattered across call sites.
    pub max_acceleration: f64,
}

/// The full set of machine facts this planner needs: per-axis limits
/// plus two global (not per-axis) tuning constants.
#[derive(Debug, Clone, Copy)]
pub struct MachineLimits<const N: usize> {
    pub axes: [AxisLimits; N],
    /// grblHAL's `$11`: how far the tool is allowed to deviate off an
    /// exact corner, in mm, in exchange for not stopping dead at every
    /// junction - see `queue.rs`'s junction-velocity formula. Zero means
    /// every corner is taken as a full stop (grblHAL's G61.1 exact-stop
    /// behavior).
    pub junction_deviation: f64,
    /// grblHAL's `$12`: maximum allowed chord-to-arc deviation, in mm,
    /// used by `arc.rs` to decide how many segments to tessellate an arc
    /// into - a tighter tolerance means more, shorter segments.
    pub arc_tolerance: f64,
}

impl<const N: usize> MachineLimits<N> {
    /// The direction-limited acceleration and velocity ceilings for a
    /// move along `unit_vector` - a diagonal move can't exceed ANY
    /// single axis's own limit, so each axis's ceiling is scaled by how
    /// much that axis actually contributes to the direction of travel
    /// (grblHAL's `limit_value_by_axis_maximum`), and the move is capped
    /// by whichever axis saturates first.
    ///
    /// `unit_vector` must already be normalized (unit length) - callers
    /// (`block.rs`) always derive it from `start`/`target` via
    /// `libm::sqrt`, never take it from the G-code directly.
    pub fn direction_limits(&self, unit_vector: [f64; N]) -> (f64, f64) {
        let mut max_velocity = f64::INFINITY;
        let mut max_acceleration = f64::INFINITY;
        for (axis, &component) in self.axes.iter().zip(unit_vector.iter()) {
            let component = libm::fabs(component);
            if component > 1e-12 {
                max_velocity = libm::fmin(max_velocity, axis.max_velocity / component);
                max_acceleration = libm::fmin(max_acceleration, axis.max_acceleration / component);
            }
        }
        (max_velocity, max_acceleration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> MachineLimits<3> {
        MachineLimits {
            axes: [
                AxisLimits {
                    max_velocity: 3000.0,
                    max_acceleration: 500.0,
                },
                AxisLimits {
                    max_velocity: 3000.0,
                    max_acceleration: 500.0,
                },
                AxisLimits {
                    max_velocity: 600.0,
                    max_acceleration: 100.0,
                },
            ],
            junction_deviation: 0.01,
            arc_tolerance: 0.002,
        }
    }

    #[test]
    fn pure_x_move_uses_only_x_axis_limits() {
        let (v, a) = limits().direction_limits([1.0, 0.0, 0.0]);
        assert_eq!(v, 3000.0);
        assert_eq!(a, 500.0);
    }

    #[test]
    fn diagonal_move_is_capped_by_the_slower_axis() {
        // 45-degree XZ move: Z's much lower limits dominate once scaled
        // by its (larger, since further from zero contribution) share.
        let half = libm::sqrt(0.5);
        let (v, a) = limits().direction_limits([0.0, half, half]);
        // Z's ceiling (600/half, 100/half) is far below Y's (3000/half,
        // 500/half), so the direction limit must equal Z's scaled value.
        assert!((v - 600.0 / half).abs() < 1e-9);
        assert!((a - 100.0 / half).abs() < 1e-9);
    }
}
