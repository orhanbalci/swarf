//! `swarf-motion`: trajectory/acceleration motion planner.
//!
//! Part of the **Swarf** project - Layer 3 of the stack described in the
//! repository root README. This crate is a generic, junction-deviation
//! look-ahead motion planner: it doesn't know or care whether its moves
//! came from G-code, a robotics API, or anything else. `swarf-gcode`
//! (Layer 2) resolves G-code into an ordered stream of moves and
//! non-motion commands, but deliberately leaves arcs unflattened and
//! never assigns a move any speed beyond its own programmed feed rate -
//! per that crate's own docs, that's specifically so a downstream
//! planner can own arc interpolation and look-ahead velocity planning.
//! This crate is that planner - but it has NO dependency on
//! `swarf-gcode` at all, or any other upstream format. See
//! `swarf-bridge` for the thin adapter crate that translates
//! `swarf_gcode::LineOutput` into calls on this crate's plain API - kept
//! deliberately separate so this crate stays reusable for any motion
//! source, and `swarf-gcode` never needs to know a motion planner exists
//! either.
//!
//! We could find no reusable Rust crate for this: `scurve_motion` is a
//! single-axis jerk-limited profile generator with no multi-block
//! cornering/look-ahead, and `motion-planning` is a pre-alpha,
//! Hermite-spline-based crate unrelated to CNC block planning. This is
//! genuinely new ground, verified against grblHAL's `planner.c` (block
//! queue, junction-deviation cornering, reverse/forward look-ahead
//! passes) as the reference implementation - see `queue.rs`'s module
//! docs for the re-derived (not copy-pasted) math.
//!
//! # What this crate supports
//!
//! - **Arc tessellation**: arcs (`Planner::push_arc`) are flattened into
//!   short linear segments sized by an exact chord/sagitta tolerance
//!   (`arc.rs`), not a fixed segment count.
//! - **Junction-deviation cornering**: the speed a corner between two
//!   consecutive moves can be taken at, derived from the machine's
//!   direction-limited acceleration and a configurable deviation
//!   tolerance (`queue.rs`).
//! - **Reverse + forward look-ahead**: a sharp corner forces earlier,
//!   otherwise-unrestricted blocks to start decelerating before they
//!   reach it - not just a per-block-independent speed cap.
//! - **Trapezoidal velocity profiles**: each block's final entry/
//!   nominal/exit speed, ready for an executor to turn into an actual
//!   accel/cruise/decel ramp.
//! - **In-order passthrough of caller-supplied non-motion commands**
//!   (`Planner::push_command`, generic over any `C: Copy`) - preserved
//!   in the exact position they occurred relative to surrounding motion,
//!   though not acted on (this crate doesn't know what a `C` value
//!   means - see below).
//!
//! # Architecture
//!
//! [`Planner`] is the top-level facade: `push_linear`/`push_arc` turn a
//! move into one or more [`Block`]s (`block.rs`; more than one for an
//! arc - see `arc.rs`), each geometrized against a host-supplied
//! [`MachineLimits`] and pushed into a fixed-capacity
//! [`queue::BlockQueue`]. Pushing a block re-runs the reverse/forward
//! look-ahead passes over everything currently queued.
//! [`Planner::pop_ready`] drains the oldest entry once its speed profile
//! is finalized, as a [`PlannedBlock`] - this crate's own output
//! boundary for whatever executor consumes it next (staged for later,
//! per the root README's architecture diagram).
//!
//! `no_std`, no heap allocation anywhere - the block queue is a fixed-
//! size array (`CAPACITY` is a compile-time const generic chosen by the
//! host), matching `swarf-gcode`'s own ethos.
//!
//! # Example
//!
//! ```
//! use swarf_motion::{AxisLimits, MachineLimits, Planner, PlannedBlock};
//!
//! let limits = MachineLimits {
//!     axes: [AxisLimits { max_velocity: 3000.0, max_acceleration: 500.0 }; 3],
//!     junction_deviation: 0.01,
//!     arc_tolerance: 0.002,
//! };
//! let mut planner: Planner<16> = Planner::new(limits);
//!
//! let origin = swarf_motion::Position { x: 0.0, y: 0.0, z: 0.0 };
//! let target = swarf_motion::Position { x: 10.0, y: 0.0, z: 0.0 };
//! planner.push_linear(origin, target, 600.0, false).unwrap();
//! planner.flush(); // no more motion coming - decelerate to rest
//!
//! match planner.pop_ready() {
//!     Some(PlannedBlock::Motion { distance, nominal_speed, exit_speed, .. }) => {
//!         assert_eq!(distance, 10.0);
//!         assert!((nominal_speed - 10.0).abs() < 1e-9); // 600 mm/min = 10 mm/s
//!         assert_eq!(exit_speed, 0.0);
//!     }
//!     _ => panic!("expected one planned motion block"),
//! }
//! ```
//!
//! # What this crate deliberately does NOT do (yet)
//!
//! - **S-curve / jerk-limited profiles** - blocks are planned as
//!   trapezoidal (accel/cruise/decel), matching grblHAL's own default
//!   and lower risk for a first implementation. `scurve_motion` (an
//!   existing single-axis Rust crate) could plausibly supply this later
//!   without changing this crate's block-level API.
//! - **Tool-change sequencing** (grblHAL's M6 state machine: stop
//!   spindle/coolant, drain the buffer, move to a change position,
//!   probe, restore) - out of scope for a generic planner; a bridge
//!   crate or host decides when a `C` command means "stop everything".
//! - **Tool length offset** (G43/G43.1/G49) - this is a G-code
//!   interpreter concern, not a motion-planner one at all.
//! - The `queue.rs` "planned pointer" performance optimization grblHAL
//!   uses to avoid re-examining already-optimal blocks on every push -
//!   this crate re-runs both look-ahead passes over the whole queue
//!   every time instead. Correct, not maximally cheap; fine at this
//!   crate's small, fixed queue capacities.

#![cfg_attr(not(test), no_std)]

pub mod arc;
pub mod block;
pub mod limits;
pub mod position;
pub mod queue;

pub use block::Block;
pub use limits::{AxisLimits, MachineLimits};
pub use position::Position;
pub use queue::PlannedBlock;

use arc::ArcSegments;
use block::Block as BlockType;
use limits::MachineLimits as Limits;
use queue::BlockQueue;

/// The planner facade: turns plain move descriptions into speed-planned
/// [`PlannedBlock`]s. See this crate's module docs for the full
/// architecture. `C` is the caller's own non-motion "command" type
/// (`()` if there are none) - see `queue.rs`'s doc comment on why this
/// crate stays generic over it rather than assuming any particular
/// upstream format.
pub struct Planner<const CAPACITY: usize, C: Copy = ()> {
    queue: BlockQueue<CAPACITY, C>,
    limits: Limits<3>,
    /// The last pushed `Block`'s unit vector, used to compute the
    /// junction angle for the NEXT motion pushed - `None` initially, and
    /// deliberately left unchanged by `push_command` (a non-motion
    /// command doesn't break directional continuity between the moves
    /// on either side of it).
    last_unit_vector: Option<[f64; 3]>,
}

impl<const CAPACITY: usize, C: Copy> Planner<CAPACITY, C> {
    pub fn new(limits: Limits<3>) -> Self {
        Self {
            queue: BlockQueue::new(),
            limits,
            last_unit_vector: None,
        }
    }

    /// Push a straight-line move. `Err(())` if the queue is full.
    pub fn push_linear(&mut self, start: Position, target: Position, feed_rate: f64, is_rapid: bool) -> Result<(), ()> {
        let block = BlockType::new(start, target, feed_rate, is_rapid, &self.limits, self.last_unit_vector);
        if block.distance > 0.0 {
            self.last_unit_vector = Some(block.unit_vector);
        }
        self.queue.push_block(block)
    }

    /// Push an arc move, tessellated into short linear segments sized by
    /// `MachineLimits::arc_tolerance` (see `arc.rs`). `clockwise` and
    /// `start == target` (an explicit full-circle request) follow the
    /// same convention as G-code's G2/G3, but this crate doesn't care
    /// what produced them.
    ///
    /// NOTE: if the queue fills partway through the tessellated
    /// segments, the remaining segments are dropped and this returns
    /// `Err(())` - the caller can't cleanly "retry the whole arc" since
    /// some segments already made it into the queue. Acceptable for now
    /// (mirrors the reality that a real controller's planner buffer
    /// filling mid-arc is also just backpressure, not a rewindable
    /// event), but worth keeping in mind if a future caller needs
    /// stronger atomicity guarantees.
    pub fn push_arc(
        &mut self,
        start: Position,
        target: Position,
        center: Position,
        clockwise: bool,
        feed_rate: f64,
        is_rapid: bool,
    ) -> Result<(), ()> {
        for (seg_start, seg_end) in ArcSegments::new(start, target, center, clockwise, self.limits.arc_tolerance) {
            let block = BlockType::new(seg_start, seg_end, feed_rate, is_rapid, &self.limits, self.last_unit_vector);
            if block.distance > 0.0 {
                self.last_unit_vector = Some(block.unit_vector);
            }
            self.queue.push_block(block)?;
        }
        Ok(())
    }

    /// Push a non-motion `C`, preserving its position in the ordered
    /// stream relative to surrounding motion - see `queue.rs`'s doc
    /// comment on why this crate stays generic over what `C` means.
    pub fn push_command(&mut self, command: C) -> Result<(), ()> {
        self.queue.push_command(command)
    }

    /// Force the newest currently-queued block to decelerate to a stop -
    /// call this when the caller knows no more motion is coming (e.g.
    /// `swarf-bridge` calls it on seeing a G-code program-stop command).
    /// See `queue::BlockQueue::flush`'s doc comment for why this isn't
    /// automatic.
    pub fn flush(&mut self) {
        self.queue.flush();
    }

    /// Remove and return the oldest queued entry once its speed profile
    /// is finalized - see `queue::BlockQueue::pop_ready`.
    pub fn pop_ready(&mut self) -> Option<PlannedBlock<C>> {
        self.queue.pop_ready()
    }

    pub fn is_full(&self) -> bool {
        self.queue.is_full()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}
