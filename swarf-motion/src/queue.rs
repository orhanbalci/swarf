//! `BlockQueue` - the fixed-capacity ring buffer plus grblHAL's
//! reverse/forward look-ahead recalculate algorithm.
//!
//! # Junction velocity
//!
//! At a corner between two blocks, both direction vectors point
//! "forward" along their own block's direction of travel. Define `θ` as
//! the angle between the PREVIOUS block's direction reversed and the
//! CURRENT block's direction (equivalently: the angle you'd measure at
//! the corner vertex between the ray back the way you came and the ray
//! forward) - so `θ = π` for continuing perfectly straight (no corner at
//! all) and `θ = 0` for a full reversal. That gives
//! `cos(θ) = -dot(prev_unit, curr_unit)`, and via the half-angle
//! identity `sin(θ/2) = sqrt(0.5·(1 - cos θ))`:
//!
//! - straight through: `θ=π`, `sin(θ/2)=1` → junction speed → infinity
//!   (uncapped by cornering; nominal speed governs instead)
//! - full reversal: `θ=0`, `sin(θ/2)=0` → junction speed → 0 (must stop)
//! - general corner: `v² = acceleration · junction_deviation ·
//!   sin(θ/2) / (1 - sin(θ/2))`
//!
//! This is grblHAL's exact approach (`cos_theta = -dot(...)`, then the
//! same half-angle substitution) - re-derived and verified here from
//! the geometry directly (checked against the 0°/180°/90° cases in this
//! module's tests) rather than copied from a paraphrase, since sign
//! conventions in this specific formula are an easy place to get wrong.
//!
//! # Reverse + forward passes
//!
//! On every new block pushed, a full reverse pass walks from the newest
//! block backward assuming its exit speed is 0 (conservative - revised
//! again the next time a block is pushed), capping each block's
//! `entry_speed_sqr` at `min(max_entry_speed_sqr, exit_speed_sqr +
//! 2·accel·distance)`. A forward pass then walks oldest-to-newest from
//! whatever speed the queue's front actually carries over from
//! (`carry_over_exit_speed_sqr` - 0 at machine start, or an already-
//! popped block's real exit speed), computing what's PHYSICALLY
//! achievable by accelerating from there, and may lower (never raise) a
//! block's entry speed below the reverse pass's ceiling - a short block
//! close to the start of the program may simply not have room to
//! accelerate up to what the reverse pass would otherwise allow.
//!
//! Both passes re-run over the WHOLE currently-queued block set on every
//! push rather than tracking grblHAL's "planned" pointer optimization
//! (which skips re-examining blocks already known optimal) - correct,
//! just not maximally cheap. Deferred: at this crate's fixed, small
//! queue capacity the O(count) cost per push is negligible; the
//! optimization is a pure performance improvement, not a correctness
//! one, and can be added later without changing this module's public
//! shape.
//!
//! # Non-motion entries are generic, not `swarf-gcode`-specific
//!
//! This crate has no idea what a "command" is - spindle on, coolant off,
//! and so on are entirely a G-code (or other source) concept.
//! `BlockQueue`/`Planner` are generic over a caller-supplied `C: Copy`
//! type for exactly this: they preserve a `C` value's position in the
//! ordered stream relative to surrounding motion (see `push_command`),
//! but never look inside it. Deciding what a particular `C` value MEANS
//! (e.g. "this one means the program stopped, so call `flush`") is the
//! caller's job - `swarf-bridge` is where that happens for
//! `swarf_gcode::Command` specifically.

use crate::block::Block;
use crate::position::Position;

/// The junction speed² (mm²/s²) between two consecutive blocks - see
/// this module's doc comment for the derivation. `accel` is the
/// CURRENT block's direction-limited acceleration (matching grblHAL,
/// which uses the entering block's own acceleration ceiling for this
/// corner, not some blend of both blocks').
pub(crate) fn junction_speed_sqr(
    prev_unit: [f64; 3],
    curr_unit: [f64; 3],
    accel: f64,
    junction_deviation: f64,
) -> f64 {
    let dot = prev_unit[0] * curr_unit[0] + prev_unit[1] * curr_unit[1] + prev_unit[2] * curr_unit[2];
    let cos_theta = -dot;

    // Straight through (theta ~ pi): sin(theta/2) ~ 1, denominator of
    // the general formula below goes to zero - short-circuit rather
    // than divide by (near) zero.
    if cos_theta < -0.999_999 {
        return f64::INFINITY;
    }

    let sin_half = libm::sqrt((0.5 * (1.0 - cos_theta)).max(0.0));

    // Full reversal (theta ~ 0): must come to a stop.
    if sin_half > 0.999_999 {
        return 0.0;
    }

    accel * junction_deviation * sin_half / (1.0 - sin_half)
}

/// One entry in the queue - a geometrized, speed-planned `Block`, or a
/// caller-supplied `C` (non-motion command) carried through in-order but
/// otherwise untouched by the velocity passes - see this module's doc
/// comment.
#[derive(Debug, Clone, Copy)]
enum QueueEntry<C> {
    Block(Block),
    Command(C),
}

/// The finalized result of draining one entry from the front of the
/// queue via [`BlockQueue::pop_ready`] - entry/nominal/exit speed are
/// settled and won't change further once returned.
#[derive(Debug, Clone, Copy)]
pub enum PlannedBlock<C> {
    Motion {
        start: Position,
        target: Position,
        distance: f64,
        /// mm/s
        entry_speed: f64,
        /// mm/s
        nominal_speed: f64,
        /// mm/s
        exit_speed: f64,
        /// mm/s²
        acceleration: f64,
        is_rapid: bool,
    },
    Command(C),
}

/// Fixed-capacity ring buffer of `QueueEntry<C>` plus the look-ahead
/// algorithm - see this module's doc comment. `CAPACITY` mirrors
/// grblHAL's own small, fixed planner buffer (typically 16-36 blocks
/// depending on available RAM); this crate makes no allocation, so it's
/// a compile-time constant chosen by the host. `C` defaults to `()` for
/// callers with no non-motion commands to carry through at all.
pub struct BlockQueue<const CAPACITY: usize, C: Copy = ()> {
    entries: [Option<QueueEntry<C>>; CAPACITY],
    /// Index of the oldest (front) occupied slot.
    tail: usize,
    /// Number of currently occupied slots.
    count: usize,
    /// The real entry speed² (mm²/s²) the NEXT block to be planned
    /// carries over from whatever was already popped - 0 at machine
    /// start, or the exit speed of the last `pop_ready`'d motion block.
    carry_over_exit_speed_sqr: f64,
}

impl<const CAPACITY: usize, C: Copy> Default for BlockQueue<CAPACITY, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const CAPACITY: usize, C: Copy> BlockQueue<CAPACITY, C> {
    pub fn new() -> Self {
        Self {
            entries: [None; CAPACITY],
            tail: 0,
            count: 0,
            carry_over_exit_speed_sqr: 0.0,
        }
    }

    pub fn is_full(&self) -> bool {
        self.count == CAPACITY
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    fn physical_index(&self, logical: usize) -> usize {
        (self.tail + logical) % CAPACITY
    }

    fn entry_at(&self, logical: usize) -> &QueueEntry<C> {
        self.entries[self.physical_index(logical)].as_ref().unwrap()
    }

    fn entry_at_mut(&mut self, logical: usize) -> &mut QueueEntry<C> {
        let idx = self.physical_index(logical);
        self.entries[idx].as_mut().unwrap()
    }

    /// Push a fully-geometrized `Block` (see `block.rs`) and re-run the
    /// look-ahead passes. `Err(())` if the queue is full - the caller
    /// (this crate's `Planner`) surfaces this as a rejected push, the
    /// backpressure contract this whole pipeline is built around (see
    /// the root README's architecture diagram).
    pub fn push_block(&mut self, block: Block) -> Result<(), ()> {
        self.push(QueueEntry::Block(block))
    }

    /// Push a non-motion `C`, preserving its position in the ordered
    /// stream relative to surrounding motion - see this module's doc
    /// comment on why these are untouched, generic passthrough.
    pub fn push_command(&mut self, command: C) -> Result<(), ()> {
        self.push(QueueEntry::Command(command))
    }

    fn push(&mut self, entry: QueueEntry<C>) -> Result<(), ()> {
        if self.is_full() {
            return Err(());
        }
        let idx = self.physical_index(self.count);
        self.entries[idx] = Some(entry);
        self.count += 1;
        if matches!(entry, QueueEntry::Block(_)) {
            self.recalculate(false);
        }
        Ok(())
    }

    /// Force the newest currently-queued block to decelerate all the way
    /// to a stop, rather than leaving its exit speed at whatever's
    /// merely achievable.
    ///
    /// Every `recalculate` already assumes (conservatively, for the
    /// PURPOSE OF ENTRY-SPEED SAFETY) that the newest block ends at
    /// rest - that's what lets every earlier block's entry speed stay
    /// safe even if nothing more ever gets queued. But that assumption
    /// is deliberately NOT also applied to the newest block's own
    /// reported exit speed during normal streaming: more motion is
    /// usually about to continue from there, and forcing a hard stop
    /// every time the queue is merely, temporarily empty would turn
    /// smooth continuous motion into needless stop-and-go.
    ///
    /// This is for when that assumption becomes true - genuinely no more
    /// motion is coming (program end/stop; a caller like `swarf-bridge`
    /// decides WHEN that is, since this crate doesn't know what a `C`
    /// value means - see this module's doc comment). Call it once,
    /// before draining, so the last block's reported exit speed actually
    /// reads zero rather than whatever momentum the forward pass would
    /// otherwise have allowed.
    pub fn flush(&mut self) {
        self.recalculate(true);
    }

    /// Remove and return the oldest entry, if any - see `PlannedBlock`'s
    /// doc comment: for a motion entry, entry/nominal/exit speed are
    /// final as of this call.
    pub fn pop_ready(&mut self) -> Option<PlannedBlock<C>> {
        if self.is_empty() {
            return None;
        }
        let entry = *self.entry_at(0);
        self.entries[self.tail] = None;
        self.tail = (self.tail + 1) % CAPACITY;
        self.count -= 1;

        Some(match entry {
            QueueEntry::Command(c) => PlannedBlock::Command(c),
            QueueEntry::Block(b) => {
                self.carry_over_exit_speed_sqr = b.exit_speed_sqr;
                PlannedBlock::Motion {
                    start: b.start,
                    target: b.target,
                    distance: b.distance,
                    entry_speed: libm::sqrt(b.entry_speed_sqr),
                    nominal_speed: b.nominal_speed,
                    exit_speed: libm::sqrt(b.exit_speed_sqr),
                    acceleration: b.acceleration,
                    is_rapid: b.is_rapid,
                }
            }
        })
    }

    /// The logical indices (0 = oldest) of every `Block` entry currently
    /// queued, in order - `Command` entries are skipped entirely, since
    /// they carry no distance/acceleration/speed and aren't part of the
    /// velocity chain (see this module's doc comment).
    fn block_indices(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.count).filter(|&i| matches!(self.entry_at(i), QueueEntry::Block(_)))
    }

    fn recalculate(&mut self, force_stop_at_end: bool) {
        // Reverse pass: newest block assumed to end at rest (revised
        // again on the next push), walking backward.
        let mut exit_speed_sqr = 0.0;
        let indices: heapless_indices::Indices<CAPACITY> = self.block_indices().collect();
        for &i in indices.as_slice().iter().rev() {
            let block = match self.entry_at_mut(i) {
                QueueEntry::Block(b) => b,
                QueueEntry::Command(_) => unreachable!("block_indices only yields Block entries"),
            };
            let achievable = exit_speed_sqr + 2.0 * block.acceleration * block.distance;
            block.entry_speed_sqr = block.max_entry_speed_sqr.min(achievable);
            exit_speed_sqr = block.entry_speed_sqr;
        }

        // Forward pass: from whatever speed genuinely carries over into
        // the front of the queue, compute what's physically achievable
        // moving forward, capping against each next block's reverse-pass
        // ceiling (already sitting in its `entry_speed_sqr`) rather than
        // exceeding it.
        let mut incoming_speed_sqr = self.carry_over_exit_speed_sqr;
        for &i in indices.as_slice() {
            // Cap this block's entry by what's actually carried in.
            {
                let block = match self.entry_at_mut(i) {
                    QueueEntry::Block(b) => b,
                    QueueEntry::Command(_) => unreachable!(),
                };
                block.entry_speed_sqr = block.entry_speed_sqr.min(incoming_speed_sqr);
            }
            let (achievable_exit, nominal_sqr) = {
                let block = match self.entry_at(i) {
                    QueueEntry::Block(b) => b,
                    QueueEntry::Command(_) => unreachable!(),
                };
                (
                    block.entry_speed_sqr + 2.0 * block.acceleration * block.distance,
                    block.nominal_speed_sqr(),
                )
            };
            let mut exit_sqr = achievable_exit.min(nominal_sqr);
            let next = indices.as_slice().iter().skip_while(|&&x| x != i).nth(1);
            match next {
                // Don't exceed the NEXT block's reverse-pass ceiling -
                // still sitting in its entry_speed_sqr since the forward
                // loop hasn't reached it yet.
                Some(next) => {
                    let next_ceiling = match self.entry_at(*next) {
                        QueueEntry::Block(b) => b.entry_speed_sqr,
                        QueueEntry::Command(_) => unreachable!(),
                    };
                    exit_sqr = exit_sqr.min(next_ceiling);
                }
                // This is the newest block currently queued - only force
                // it to a hard stop if the caller told us (via `flush`)
                // that nothing more is coming. Otherwise leave whatever
                // momentum is achievable, since more motion usually IS
                // about to continue from here (see `flush`'s doc
                // comment).
                None if force_stop_at_end => exit_sqr = 0.0,
                None => {}
            }
            {
                let block = match self.entry_at_mut(i) {
                    QueueEntry::Block(b) => b,
                    QueueEntry::Command(_) => unreachable!(),
                };
                block.exit_speed_sqr = exit_sqr;
            }
            incoming_speed_sqr = exit_sqr;
        }
    }
}

/// A tiny fixed-capacity index list, purely so `recalculate` doesn't
/// need heap allocation to remember which logical positions are `Block`
/// entries while it mutates the queue - `CAPACITY` bounds it exactly
/// like the queue itself.
mod heapless_indices {
    pub struct Indices<const CAPACITY: usize> {
        buf: [usize; CAPACITY],
        len: usize,
    }

    impl<const CAPACITY: usize> Indices<CAPACITY> {
        pub fn as_slice(&self) -> &[usize] {
            &self.buf[..self.len]
        }
    }

    impl<const CAPACITY: usize> FromIterator<usize> for Indices<CAPACITY> {
        fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
            let mut buf = [0usize; CAPACITY];
            let mut len = 0;
            for value in iter {
                buf[len] = value;
                len += 1;
            }
            Self { buf, len }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_through_junction_is_unbounded() {
        let v = junction_speed_sqr([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 500.0, 0.01);
        assert!(v.is_infinite());
    }

    #[test]
    fn full_reversal_junction_is_zero() {
        let v = junction_speed_sqr([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 500.0, 0.01);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn right_angle_junction_is_between_the_extremes() {
        let v = junction_speed_sqr([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 500.0, 0.01);
        assert!(v > 0.0 && v.is_finite());
    }

    #[test]
    fn a_sharp_corner_forces_an_earlier_straight_block_to_slow_down() {
        // The worked 3-block example: block A (straight, fast) -> block
        // B (sharp corner into it) -> block C. The reverse pass must
        // lower block A's EXIT speed (= block B's entry) below A's own
        // nominal speed, even though A itself is a straight, unrestricted
        // segment - proving look-ahead propagates backward across
        // blocks, not just within one block's own local cap.
        use crate::limits::{AxisLimits, MachineLimits};
        let limits = MachineLimits {
            axes: [
                AxisLimits { max_velocity: 3000.0, max_acceleration: 500.0 },
                AxisLimits { max_velocity: 3000.0, max_acceleration: 500.0 },
                AxisLimits { max_velocity: 600.0, max_acceleration: 100.0 },
            ],
            junction_deviation: 0.01,
            arc_tolerance: 0.002,
        };
        let pos = |x: f64, y: f64| Position { x, y, z: 0.0 };

        let a = Block::new(pos(0.0, 0.0), pos(50.0, 0.0), 3000.0, false, &limits, None);
        let b = Block::new(pos(50.0, 0.0), pos(50.0, 0.5), 3000.0, false, &limits, Some(a.unit_vector));
        let c = Block::new(pos(50.0, 0.5), pos(100.0, 0.5), 3000.0, false, &limits, Some(b.unit_vector));

        let mut queue: BlockQueue<8> = BlockQueue::new();
        queue.push_block(a).unwrap();
        queue.push_block(b).unwrap();
        queue.push_block(c).unwrap();

        let planned_a = queue.pop_ready().unwrap();
        let PlannedBlock::Motion { exit_speed, nominal_speed, .. } = planned_a else {
            panic!("expected motion");
        };
        // The near-90-degree corner into B must have forced A's exit
        // speed well below A's own nominal (unrestricted) speed.
        assert!(exit_speed < nominal_speed * 0.5, "exit {exit_speed} vs nominal {nominal_speed}");
    }

    #[test]
    fn commands_pass_through_in_order_untouched_by_speed_planning() {
        use crate::limits::{AxisLimits, MachineLimits};

        let limits = MachineLimits {
            axes: [
                AxisLimits { max_velocity: 3000.0, max_acceleration: 500.0 },
                AxisLimits { max_velocity: 3000.0, max_acceleration: 500.0 },
                AxisLimits { max_velocity: 600.0, max_acceleration: 100.0 },
            ],
            junction_deviation: 0.01,
            arc_tolerance: 0.002,
        };
        let pos = |x: f64, y: f64| Position { x, y, z: 0.0 };
        let a = Block::new(pos(0.0, 0.0), pos(10.0, 0.0), 1000.0, false, &limits, None);

        // `C` here is a plain `u32` - this crate doesn't care what a
        // "command" means, only that it round-trips in order.
        let mut queue: BlockQueue<8, u32> = BlockQueue::new();
        queue.push_command(1).unwrap();
        queue.push_block(a).unwrap();
        queue.push_command(2).unwrap();

        assert!(matches!(queue.pop_ready(), Some(PlannedBlock::Command(1))));
        assert!(matches!(queue.pop_ready(), Some(PlannedBlock::Motion { .. })));
        assert!(matches!(queue.pop_ready(), Some(PlannedBlock::Command(2))));
        assert!(queue.pop_ready().is_none());
    }

    #[test]
    fn without_flush_the_last_queued_block_may_report_nonzero_exit_speed() {
        use crate::limits::{AxisLimits, MachineLimits};
        let limits = MachineLimits {
            axes: [
                AxisLimits { max_velocity: 3000.0, max_acceleration: 500.0 },
                AxisLimits { max_velocity: 3000.0, max_acceleration: 500.0 },
                AxisLimits { max_velocity: 600.0, max_acceleration: 100.0 },
            ],
            junction_deviation: 0.01,
            arc_tolerance: 0.002,
        };
        let pos = |x: f64, y: f64| Position { x, y, z: 0.0 };
        let mut queue: BlockQueue<8> = BlockQueue::new();
        // Long enough, and fast enough entry, that the forward pass has
        // genuine momentum to report at the end without a flush.
        let a = Block::new(pos(0.0, 0.0), pos(50.0, 0.0), 3000.0, false, &limits, None);
        queue.push_block(a).unwrap();
        let planned = queue.pop_ready().unwrap();
        let PlannedBlock::Motion { exit_speed, .. } = planned else { panic!("expected motion") };
        assert!(exit_speed > 0.0, "expected nonzero exit speed without a flush, got {exit_speed}");
    }

    #[test]
    fn flush_forces_the_last_queued_block_to_decelerate_to_a_stop() {
        use crate::limits::{AxisLimits, MachineLimits};
        let limits = MachineLimits {
            axes: [
                AxisLimits { max_velocity: 3000.0, max_acceleration: 500.0 },
                AxisLimits { max_velocity: 3000.0, max_acceleration: 500.0 },
                AxisLimits { max_velocity: 600.0, max_acceleration: 100.0 },
            ],
            junction_deviation: 0.01,
            arc_tolerance: 0.002,
        };
        let pos = |x: f64, y: f64| Position { x, y, z: 0.0 };
        let mut queue: BlockQueue<8> = BlockQueue::new();
        let a = Block::new(pos(0.0, 0.0), pos(50.0, 0.0), 3000.0, false, &limits, None);
        queue.push_block(a).unwrap();
        queue.flush();
        let planned = queue.pop_ready().unwrap();
        let PlannedBlock::Motion { exit_speed, .. } = planned else { panic!("expected motion") };
        assert_eq!(exit_speed, 0.0);
    }

    #[test]
    fn queue_rejects_pushes_past_capacity() {
        use crate::limits::{AxisLimits, MachineLimits};
        let limits = MachineLimits {
            axes: [
                AxisLimits { max_velocity: 3000.0, max_acceleration: 500.0 },
                AxisLimits { max_velocity: 3000.0, max_acceleration: 500.0 },
                AxisLimits { max_velocity: 600.0, max_acceleration: 100.0 },
            ],
            junction_deviation: 0.01,
            arc_tolerance: 0.002,
        };
        let pos = |x: f64, y: f64| Position { x, y, z: 0.0 };
        let mut queue: BlockQueue<2> = BlockQueue::new();
        assert!(queue.push_block(Block::new(pos(0.0, 0.0), pos(1.0, 0.0), 1000.0, false, &limits, None)).is_ok());
        assert!(queue.push_block(Block::new(pos(1.0, 0.0), pos(2.0, 0.0), 1000.0, false, &limits, None)).is_ok());
        assert!(queue.push_block(Block::new(pos(2.0, 0.0), pos(3.0, 0.0), 1000.0, false, &limits, None)).is_err());
    }
}
