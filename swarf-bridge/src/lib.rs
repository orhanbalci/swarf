//! `swarf-bridge`: the thin adapter between `swarf-gcode` (Layer 2) and
//! `swarf-motion` (Layer 3).
//!
//! Neither of those two crates depends on the other - `swarf-gcode` has
//! no idea a motion planner exists, and `swarf-motion` has no idea
//! G-code exists (see that crate's own docs: it's a generic
//! junction-deviation planner, reusable for any source of coordinated
//! moves). Something still has to translate one into the other for the
//! common case of actually running a G-code file through this pipeline
//! - that's this crate's entire job, and its only reason to exist.
//!
//! # What this crate does
//!
//! [`GcodePlanner`] wraps a `swarf_motion::Planner<CAPACITY,
//! swarf_gcode::Command>` and implements `swarf_gcode::OutputSink`, so
//! it can sit directly downstream of a `swarf_gcode::Interpreter` the
//! same way any other sink does:
//!
//! - `LineOutput::Motion` becomes `Planner::push_linear` or
//!   `Planner::push_arc`, translating `swarf_gcode::Position` into
//!   `swarf_motion::Position` and reading off rapid/arc/direction from
//!   `swarf_gcode::MotionMode`.
//! - `LineOutput::Command` becomes `Planner::push_command`, generic over
//!   `swarf_gcode::Command` (this is the ONE place in the whole pipeline
//!   that decides "a `Command::ProgramFlow` means the machine is
//!   stopping" and calls `Planner::flush` accordingly - `swarf-motion`
//!   itself has no opinion on what any particular command means, see its
//!   `queue.rs` module docs).
//!
//! # Example
//!
//! ```
//! use swarf_gcode::{ErrorSink, Interpreter, InterpretError};
//! use swarf_motion::{AxisLimits, MachineLimits, PlannedBlock};
//! use swarf_bridge::GcodePlanner;
//!
//! #[derive(Default)]
//! struct Errors(Vec<InterpretError>);
//! impl ErrorSink for Errors {
//!     fn push(&mut self, error: InterpretError) {
//!         self.0.push(error);
//!     }
//! }
//!
//! let limits = MachineLimits {
//!     axes: [AxisLimits { max_velocity: 3000.0, max_acceleration: 500.0 }; 3],
//!     junction_deviation: 0.01,
//!     arc_tolerance: 0.002,
//! };
//! let planner: GcodePlanner<16> = GcodePlanner::new(limits);
//!
//! let mut interp = Interpreter::new(planner, Errors::default());
//! interp.run("G21 G90\nG1 X10 F600\nM2\n");
//! let (mut planner, errors) = interp.into_sinks();
//! assert!(errors.0.is_empty());
//!
//! match planner.pop_ready() {
//!     Some(PlannedBlock::Motion { distance, exit_speed, .. }) => {
//!         assert_eq!(distance, 10.0);
//!         // M2 triggered a flush - the program ends at rest.
//!         assert_eq!(exit_speed, 0.0);
//!     }
//!     _ => panic!("expected one planned motion block"),
//! }
//! ```

#![cfg_attr(not(test), no_std)]

use swarf_gcode::{Command, LineOutput, MotionMode, OutputSink, ResolvedMotionCommand};
use swarf_motion::{MachineLimits, PlannedBlock, Planner, Position};

fn to_motion_position(p: swarf_gcode::Position) -> Position {
    Position { x: p.x, y: p.y, z: p.z }
}

/// Wraps a `swarf_motion::Planner<CAPACITY, swarf_gcode::Command>` as a
/// `swarf_gcode::OutputSink` - see this crate's module docs.
pub struct GcodePlanner<const CAPACITY: usize> {
    inner: Planner<CAPACITY, Command>,
}

impl<const CAPACITY: usize> GcodePlanner<CAPACITY> {
    pub fn new(limits: MachineLimits<3>) -> Self {
        Self {
            inner: Planner::new(limits),
        }
    }

    /// Remove and return the oldest queued entry once its speed profile
    /// is finalized - see `swarf_motion::Planner::pop_ready`.
    pub fn pop_ready(&mut self) -> Option<PlannedBlock<Command>> {
        self.inner.pop_ready()
    }

    pub fn is_full(&self) -> bool {
        self.inner.is_full()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn push_motion(&mut self, m: &ResolvedMotionCommand) -> Result<(), ()> {
        let is_rapid = m.motion_mode == MotionMode::Rapid;
        let start = to_motion_position(m.start);
        let target = to_motion_position(m.target);

        match m.arc {
            None => self.inner.push_linear(start, target, m.feed_rate, is_rapid),
            Some(arc) => {
                let clockwise = m.motion_mode == MotionMode::ArcClockwise;
                let center = to_motion_position(arc.center);
                self.inner.push_arc(start, target, center, clockwise, m.feed_rate, is_rapid)
            }
        }
    }
}

impl<const CAPACITY: usize> OutputSink for GcodePlanner<CAPACITY> {
    fn push(&mut self, output: LineOutput) -> Result<(), ()> {
        match output {
            LineOutput::Motion(m) => self.push_motion(&m),
            LineOutput::Command(c) => {
                // M0/M1/M2/M30 all mean the machine physically comes to
                // a stop at this point in the program - retroactively
                // force whatever's currently the newest queued block to
                // decelerate all the way to rest, rather than reporting
                // whatever momentum a forward pass assuming "more motion
                // is coming" would otherwise allow (see
                // `swarf_motion::Planner::flush`'s doc comment for why
                // this ISN'T done on every push, only here - and why
                // `swarf-motion` itself can't make this call, since it
                // doesn't know what a `Command::ProgramFlow` means).
                if matches!(c, Command::ProgramFlow(_)) {
                    self.inner.flush();
                }
                self.inner.push_command(c)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use swarf_gcode::{CoolantCommand, ErrorSink, InterpretError, Interpreter, ProgramFlow, SpindleCommand};

    #[derive(Default)]
    struct Errors(std::vec::Vec<InterpretError>);
    impl ErrorSink for Errors {
        fn push(&mut self, error: InterpretError) {
            self.0.push(error);
        }
    }

    fn limits() -> MachineLimits<3> {
        MachineLimits {
            axes: [
                swarf_motion::AxisLimits { max_velocity: 3000.0, max_acceleration: 500.0 },
                swarf_motion::AxisLimits { max_velocity: 3000.0, max_acceleration: 500.0 },
                swarf_motion::AxisLimits { max_velocity: 600.0, max_acceleration: 100.0 },
            ],
            junction_deviation: 0.01,
            arc_tolerance: 0.002,
        }
    }

    fn run(src: &str) -> GcodePlanner<64> {
        let planner: GcodePlanner<64> = GcodePlanner::new(limits());
        let mut interp = Interpreter::new(planner, Errors::default());
        interp.run(src);
        let (planner, errors) = interp.into_sinks();
        assert!(errors.0.is_empty(), "unexpected errors: {:?}", errors.0);
        planner
    }

    #[test]
    fn linear_move_becomes_one_planned_block_with_matching_distance() {
        let mut planner = run("G21 G90\nG1 X10 F600\n");
        let block = planner.pop_ready().unwrap();
        let PlannedBlock::Motion { distance, nominal_speed, .. } = block else { panic!("expected motion") };
        assert_eq!(distance, 10.0);
        assert!((nominal_speed - 10.0).abs() < 1e-9); // 600mm/min = 10mm/s
    }

    #[test]
    fn arc_move_tessellates_into_multiple_planned_blocks() {
        // G3 (CCW) from (10,0) to (0,10) around center (0,0) is the
        // short quarter-circle way; G2 (CW) for the same endpoints would
        // be the long 270-degree way around - worth remembering when
        // writing arc test G-code, since it's an easy mixup.
        let mut planner = run("G21 G90\nG1 X10 Y0 F600\nG3 X0 Y10 I-10 J0\n");
        // First block: the leading G1. Everything after it is the arc's
        // tessellated segments.
        assert!(matches!(planner.pop_ready(), Some(PlannedBlock::Motion { .. })));
        let mut arc_segments = 0;
        while let Some(block) = planner.pop_ready() {
            assert!(matches!(block, PlannedBlock::Motion { .. }));
            arc_segments += 1;
        }
        assert!(arc_segments > 1, "expected the arc to tessellate into multiple segments");
    }

    #[test]
    fn commands_pass_through_in_order() {
        let mut planner = run("G21 G90\nM3 S1000\nG1 X10 F600\nM5\n");
        assert!(matches!(
            planner.pop_ready(),
            Some(PlannedBlock::Command(Command::Spindle(SpindleCommand::Clockwise(_))))
        ));
        assert!(matches!(planner.pop_ready(), Some(PlannedBlock::Motion { .. })));
        assert!(matches!(
            planner.pop_ready(),
            Some(PlannedBlock::Command(Command::Spindle(SpindleCommand::Stop)))
        ));
    }

    #[test]
    fn program_end_flushes_the_last_block_to_rest() {
        let mut planner = run("G21 G90\nG1 X50 F3000\nM2\n");
        let block = planner.pop_ready().unwrap();
        let PlannedBlock::Motion { exit_speed, .. } = block else { panic!("expected motion") };
        assert_eq!(exit_speed, 0.0);
        assert!(matches!(
            planner.pop_ready(),
            Some(PlannedBlock::Command(Command::ProgramFlow(ProgramFlow::End)))
        ));
    }

    #[test]
    fn coolant_and_program_flow_also_pass_through() {
        let mut planner = run("G21 G90\nM8\nG1 X10 F600\nM9\nM30\n");
        assert!(matches!(planner.pop_ready(), Some(PlannedBlock::Command(Command::Coolant(CoolantCommand::Flood)))));
        assert!(matches!(planner.pop_ready(), Some(PlannedBlock::Motion { .. })));
        assert!(matches!(planner.pop_ready(), Some(PlannedBlock::Command(Command::Coolant(CoolantCommand::Off)))));
        assert!(matches!(
            planner.pop_ready(),
            Some(PlannedBlock::Command(Command::ProgramFlow(ProgramFlow::EndAndRewind)))
        ));
    }
}
