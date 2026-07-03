//! `swarf-gcode`: a NIST RS274NGC-compatible modal-state interpreter.
//!
//! Part of the **Swarf** project — a family of crates filling gaps for
//! CNC firmware development in Rust. This crate is Layer 2 of that stack:
//! it turns parsed G-code into resolved motion and machine commands for
//! a downstream motion planner / ring buffer.
//!
//! This crate fills the gap we identified across a long investigation of
//! the Rust G-code ecosystem: parsers (`gcode`, `async_gcode`, `g-code`)
//! exist and are mature, but none of them track modal state or resolve
//! parsed G-code into actual motion commands. That interpretation layer,
//! what grblHAL calls `gc_state` and what the NIST reference
//! implementation calls the interpreter proper, did not exist as a
//! standalone, reusable Rust crate. This is an attempt to build it,
//! verified against:
//!   - the NIST RS274NGC Interpreter Version 3 spec (modal groups, Table 4)
//!   - grblHAL's gc_state / gcode.c as a live reference implementation,
//!     including its published list of supported G/M-codes
//!   - the `gcode` 0.7 crate's zero-allocation visitor trait API
//!
//! # What this crate supports
//!
//! - **Motion**: G0/G1 straight moves, G2/G3 arcs (both I/J/K center form
//!   and R radius form, in all three planes), G80 cancel.
//! - **Canned drilling cycles**: G81, G82, G83 (peck), G85, G86, G89,
//!   with G98/G99 retract-mode control - G17/XY plane only, one hole per
//!   line (no `L` repeat count).
//! - **Non-modal positioning**: G28/G30 (return to a stored reference
//!   position, with G28.1/G30.1 to set it) and G53 (one-line machine-
//!   coordinate override).
//! - **Work coordinates**: G54-G59.3 selection (offsets preloaded by the
//!   host - see `state::ModalState::set_coordinate_system_offset`) and
//!   G92/G92.1 (set/cancel an additional origin shift).
//! - **Modal state**: G17/G18/G19 plane, G20/G21 units, G90/G91 distance
//!   mode, G91.1 arc-distance mode (tracked; IJK is always incremental,
//!   matching grblHAL), G93/G94 feed rate mode (tracked).
//! - **Non-motion machine commands** (via a separate `command::Command`
//!   sink, not mixed into the motion stream): M3/M4/M5 spindle (with
//!   modal S), M7/M8/M9 coolant, M0/M1/M2/M30 program flow, T + M6 tool
//!   select/change, G4 dwell.
//! - **Modal-group conflict detection** (e.g. `"G0 G1 X10"`) per NIST
//!   Table 4 - see `modal_groups` module docs for current coverage.
//!
//! See "What this crate deliberately does NOT do" below for the
//! boundary of this list, and each module's docs for the exact
//! semantics of what's in it.
//!
//! # Architecture
//!
//! We build directly against `gcode::core`'s zero-allocation visitor
//! traits (`ProgramVisitor` / `BlockVisitor` / `CommandVisitor`) rather
//! than the higher-level, `alloc`-requiring `gcode::parse()` AST API,
//! because the embedded firmware target this is ultimately destined for
//! cannot assume a heap allocator - this crate is `no_std` and performs
//! no allocation anywhere.
//!
//! The interpreter's state is split across three visitor levels, each
//! holding a mutable borrow back into the one real state owner:
//!
//!   `Interpreter`      (impl ProgramVisitor) - owns `ModalState`
//!                       (Interface 1), the only long-lived state;
//!                       persists across the whole parse / across many
//!                       lines.
//!   `BlockCtx<'a>`      (impl BlockVisitor)   - per-line scratch: which
//!                       axis words were seen, which modal groups were
//!                       touched this line (for conflict detection),
//!                       borrows `&'a mut Interpreter`.
//!   `CommandCtx<'a>`    (impl CommandVisitor) - per-command scratch for
//!                       commands that take their own arguments (e.g. the
//!                       X10 Y20 following a G1), shares the SAME
//!                       pending-axis accumulator as its parent BlockCtx.
//!
//! This mirrors the verified call order: a block produces zero or more
//! commands, each command receives zero or more arguments, then
//! `end_command` -> back to the block, then (after all commands)
//! `end_line` -> back to the program. `end_line` is where a fully
//! resolved line becomes output and gets handed to one of the
//! interpreter's two caller-supplied sinks:
//!
//!   - [`OutputSink`] receives [`LineOutput`] - either a
//!     [`ResolvedMotionCommand`] (Interface 2: every resolved move -
//!     straight, arc, or a canned cycle's constituent rapid/feed legs)
//!     or a [`Command`] (Interface 3: everything that isn't a move -
//!     spindle, coolant, program flow, tool change, dwell) - in the
//!     exact order the interpreter produced them.
//!   - [`ErrorSink`] receives [`InterpretError`] - both syntax errors
//!     from `gcode` and semantic errors detected here (modal-group
//!     conflicts, missing canned-cycle parameters, invalid arcs, a
//!     rejecting `OutputSink`, etc).
//!
//! Motion and non-motion output share one sink (rather than each having
//! its own) so a downstream real-time consumer can always tell exactly
//! where a `Command` (e.g. "spindle on") falls relative to the
//! surrounding moves - see `command.rs`'s and `visitor.rs`'s module
//! docs for the full rationale and call-flow detail. For feeding this
//! interpreter from a real-time loop one line at a time rather than a
//! whole program at once, see [`Interpreter::step`].
//!
//! # Example
//!
//! ```
//! use swarf_gcode::{Command, ErrorSink, Interpreter, InterpretError, LineOutput, OutputSink};
//!
//! #[derive(Default)]
//! struct Outputs(Vec<LineOutput>);
//! impl OutputSink for Outputs {
//!     fn push(&mut self, output: LineOutput) -> Result<(), ()> {
//!         self.0.push(output);
//!         Ok(())
//!     }
//! }
//!
//! #[derive(Default)]
//! struct Errors(Vec<InterpretError>);
//! impl ErrorSink for Errors {
//!     fn push(&mut self, error: InterpretError) {
//!         self.0.push(error);
//!     }
//! }
//!
//! let mut interp = Interpreter::new(Outputs::default(), Errors::default());
//! interp.run("G21 G90\nG0 X10 Y0\nM3 S1000\nG1 X20 F300\n");
//! let (outputs, errors) = interp.into_sinks();
//!
//! // Order is preserved: the G0, then the M3, then the G1.
//! assert_eq!(outputs.0.len(), 3);
//! assert!(matches!(outputs.0[0], LineOutput::Motion(_)));
//! assert!(matches!(outputs.0[1], LineOutput::Command(Command::Spindle(_))));
//! assert!(matches!(outputs.0[2], LineOutput::Motion(_)));
//! assert!(errors.0.is_empty());
//! ```
//!
//! # What this crate deliberately does NOT do (yet)
//!
//! - Parameter / expression evaluation (`#1`, `#<expr>`). `gcode` 0.7's
//!   `Value` enum has a `Variable(&str)` case for this in principle, but
//!   this release's actual lexer never constructs it - verified
//!   directly - so `#` currently surfaces as a plain syntax error
//!   (`InterpretError::SyntaxError`) rather than something we can
//!   distinguish and reject on purpose. `record_axis` still checks for
//!   `Value::Variable` and would map it to `InterpretError::UnsupportedSyntax`
//!   if a future `gcode` release starts producing it. Either way, we
//!   never silently default a parameter reference to zero - that would
//!   be a correctness footgun for anyone using this crate without
//!   reading this note. Parameter support is real, deferrable scope.
//! - Block delete (`/`). Not a feature of `gcode` 0.7's grammar either -
//!   a leading `/` is reported as a syntax diagnostic, not stripped.
//! - G73 (high-speed peck drilling with a partial "chip break" retract -
//!   G83's full-retract-between-pecks variant IS implemented).
//! - G10 (set coordinate data from within a program) - the in-program
//!   alternative to a host preloading
//!   `state::ModalState::set_coordinate_system_offset`.
//! - G92.2/G92.3 (suspend and restore the G92 offset) - only G92's set
//!   and G92.1's cancel are implemented.
//! - The `L` repeat count on canned cycles, and canned cycles in any
//!   plane other than G17/XY.
//! - Tool length offset resolution (G43/G43.1/G49), cutter compensation
//!   (G40/G41/G42), threading (G33), splines (G5/G5.1), scaling
//!   (G50/G51), and all lathe-specific codes (G96/G97, G7*/G8*) - out of
//!   scope for a mill-focused "basic CNC" target.
//!
//! These are real parts of the NIST spec, deliberately staged for later
//! per our explicit decision to target core mill motion semantics first.

#![cfg_attr(not(test), no_std)]

pub mod command;
pub mod modal_groups;
pub mod motion;
pub mod state;
pub mod visitor;

pub use command::{Command, CoolantCommand, ProgramFlow, SpindleCommand};
pub use modal_groups::{ModalGroup, ModalGroupSet};
pub use motion::{ArcError, ArcGeometry, MotionMode, ResolvedMotionCommand};
pub use state::{CoordinateSystem, DistanceMode, ModalState, Plane, Position, Units};
pub use visitor::{ErrorSink, InterpretError, Interpreter, LineOutput, OutputSink};
