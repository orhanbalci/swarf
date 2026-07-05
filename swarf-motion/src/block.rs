//! `Block` - one planner-queue entry: mirrors grblHAL's `plan_block_t`,
//! but scoped to exactly what THIS layer owns. grblHAL's struct also
//! carries per-axis step counts and direction bits - those are a
//! stepper's concern (`swarf-step`, staged for later), not this
//! trajectory-planning layer's, so they're not here.
//!
//! A `Block` is built from either a genuine `Linear`/`Rapid`
//! `ResolvedMotionCommand`, or one tessellated segment of an arc (see
//! `arc.rs`) - either way, `Block::new` only needs a start/target pair,
//! a feed rate, and whether it's a rapid, plus the machine's limits and
//! the previous block's unit vector (to compute the junction angle
//! between them). It does NOT decide entry/exit speed - `queue.rs`'s
//! reverse/forward look-ahead passes own that, since a block's entry
//! speed depends on the whole buffer, not just this one move.

use crate::limits::MachineLimits;
use crate::position::Position;

/// Cheap conversion for the fixed floor under a feed move's rate, so a
/// program with no `F` word yet (feed rate 0.0, `ModalState`'s default)
/// can't produce a zero/negative-duration block.
const MIN_FEED_RATE_MM_PER_MIN: f64 = 1.0;

/// One fully-geometrized planner-queue entry, before entry/exit speeds
/// are known - those fields start at their safe defaults (0 for
/// entry/exit, the block's own achievable ceiling for
/// `max_entry_speed_sqr`) and are only ever tightened, never loosened,
/// by `queue.rs`'s recalculate passes.
#[derive(Debug, Clone, Copy)]
pub struct Block {
    pub start: Position,
    pub target: Position,
    /// Unit vector from `start` to `target`, in the same order
    /// `MachineLimits::direction_limits` expects. `[0.0; 3]` for a
    /// zero-length block (shouldn't normally occur - a caller has no
    /// reason to push a no-op move - but guarded rather than dividing by
    /// zero, see `Block::new`).
    pub unit_vector: [f64; 3],
    /// Path length in mm.
    pub distance: f64,
    /// This block's own achievable top speed, in mm/s - the feed rate
    /// (or an assumed rapid rate) as limited by whichever axis
    /// saturates first for this direction of travel. Never exceeded
    /// regardless of look-ahead.
    pub nominal_speed: f64,
    /// Acceleration achievable in this block's direction, in mm/s²,
    /// direction-limited the same way as `nominal_speed`.
    pub acceleration: f64,
    pub is_rapid: bool,

    /// Speed² (mm²/s²) at which this block is entered, once
    /// `queue.rs`'s passes have settled it. Starts at 0 (the safest
    /// possible assumption) until a pass says otherwise.
    pub entry_speed_sqr: f64,
    /// The fastest this block could possibly be entered at, ignoring
    /// what surrounds it in the queue - `min(nominal_speed², junction
    /// cap against the PREVIOUS block)`. The reverse/forward passes
    /// never produce an `entry_speed_sqr` above this.
    pub max_entry_speed_sqr: f64,
    /// Speed² (mm²/s²) this block is exited at, once the forward pass
    /// has settled it - always equal to the next block's final
    /// `entry_speed_sqr`, or the forward pass's own achievable value if
    /// this is currently the newest block in the queue.
    pub exit_speed_sqr: f64,
}

impl Block {
    /// `prev_unit_vector`: the previous block's direction (`None` for
    /// the very first block in a program, or right after a non-motion
    /// gap that breaks continuity) - used only to compute the junction
    /// angle between this block and the one before it.
    pub fn new(
        start: Position,
        target: Position,
        feed_rate: f64,
        is_rapid: bool,
        limits: &MachineLimits<3>,
        prev_unit_vector: Option<[f64; 3]>,
    ) -> Self {
        let dx = target.x - start.x;
        let dy = target.y - start.y;
        let dz = target.z - start.z;
        let distance = libm::sqrt(dx * dx + dy * dy + dz * dz);

        let unit_vector = if distance > 1e-12 {
            [dx / distance, dy / distance, dz / distance]
        } else {
            [0.0; 3]
        };

        let (axis_max_velocity, axis_max_acceleration) = limits.direction_limits(unit_vector);

        // Programmed feed rate is mm/min; internal speed unit is mm/s
        // throughout this crate (see `limits.rs`'s doc comment on why
        // `MachineLimits` itself keeps mm/min - it's converted exactly
        // once, here, not scattered across call sites).
        let programmed_speed = if is_rapid {
            axis_max_velocity
        } else {
            (feed_rate.max(MIN_FEED_RATE_MM_PER_MIN) / 60.0).min(axis_max_velocity)
        };

        let max_junction_speed_sqr = prev_unit_vector.map_or(f64::INFINITY, |prev| {
            crate::queue::junction_speed_sqr(prev, unit_vector, axis_max_acceleration, limits.junction_deviation)
        });

        let nominal_speed_sqr = programmed_speed * programmed_speed;
        let max_entry_speed_sqr = nominal_speed_sqr.min(max_junction_speed_sqr);

        Self {
            start,
            target,
            unit_vector,
            distance,
            nominal_speed: programmed_speed,
            acceleration: axis_max_acceleration,
            is_rapid,
            entry_speed_sqr: 0.0,
            max_entry_speed_sqr,
            exit_speed_sqr: 0.0,
        }
    }

    pub fn nominal_speed_sqr(&self) -> f64 {
        self.nominal_speed * self.nominal_speed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limits::AxisLimits;

    fn limits() -> MachineLimits<3> {
        MachineLimits {
            axes: [
                AxisLimits { max_velocity: 3000.0, max_acceleration: 500.0 },
                AxisLimits { max_velocity: 3000.0, max_acceleration: 500.0 },
                AxisLimits { max_velocity: 600.0, max_acceleration: 100.0 },
            ],
            junction_deviation: 0.01,
            arc_tolerance: 0.002,
        }
    }

    fn pos(x: f64, y: f64, z: f64) -> Position {
        Position { x, y, z }
    }

    #[test]
    fn first_block_has_no_junction_cap() {
        let b = Block::new(pos(0.0, 0.0, 0.0), pos(10.0, 0.0, 0.0), 1000.0, false, &limits(), None);
        assert_eq!(b.distance, 10.0);
        assert_eq!(b.max_entry_speed_sqr, b.nominal_speed_sqr());
    }

    #[test]
    fn feed_rate_converts_mm_per_min_to_mm_per_s() {
        let b = Block::new(pos(0.0, 0.0, 0.0), pos(10.0, 0.0, 0.0), 600.0, false, &limits(), None);
        assert!((b.nominal_speed - 10.0).abs() < 1e-9);
    }

    #[test]
    fn zero_length_move_has_zero_unit_vector_not_nan() {
        let b = Block::new(pos(5.0, 5.0, 5.0), pos(5.0, 5.0, 5.0), 1000.0, false, &limits(), None);
        assert_eq!(b.unit_vector, [0.0; 3]);
        assert_eq!(b.distance, 0.0);
    }

    #[test]
    fn straight_continuation_has_a_high_junction_cap() {
        let first = Block::new(pos(0.0, 0.0, 0.0), pos(10.0, 0.0, 0.0), 3000.0, false, &limits(), None);
        let second = Block::new(
            pos(10.0, 0.0, 0.0),
            pos(20.0, 0.0, 0.0),
            3000.0,
            false,
            &limits(),
            Some(first.unit_vector),
        );
        // Same direction (0 degree corner) - junction cap should not be
        // the limiting factor below nominal speed.
        assert!(second.max_entry_speed_sqr >= second.nominal_speed_sqr() - 1e-6);
    }

    #[test]
    fn reversal_forces_a_low_junction_cap() {
        let first = Block::new(pos(0.0, 0.0, 0.0), pos(10.0, 0.0, 0.0), 3000.0, false, &limits(), None);
        let second = Block::new(
            pos(10.0, 0.0, 0.0),
            pos(0.0, 0.0, 0.0),
            3000.0,
            false,
            &limits(),
            Some(first.unit_vector),
        );
        // 180 degree reversal - must be much slower than nominal.
        assert!(second.max_entry_speed_sqr < second.nominal_speed_sqr() * 0.01);
    }
}
