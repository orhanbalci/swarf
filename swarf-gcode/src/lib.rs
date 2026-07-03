//! `swarf-gcode`: a NIST RS274NGC-compatible modal-state interpreter.
//!
//! Part of the **Swarf** project — a family of crates filling gaps for
//! CNC firmware development in Rust. This crate is Layer 2 of that stack:
//! it turns parsed G-code into resolved motion commands for a downstream
//! motion planner / ring buffer.
//!
//! This crate fills the gap we identified across a long investigation of
//! the Rust G-code ecosystem: parsers (`gcode`, `async_gcode`, `g-code`)
//! exist and are mature, but none of them track modal state or resolve
//! parsed G-code into actual motion commands. That interpretation layer
//! - what grblHAL calls `gc_state`, what the NIST reference implementation
//! calls the interpreter proper - did not exist as a standalone, reusable
//! Rust crate. This is an attempt to build it, verified against:
//!   - the NIST RS274NGC Interpreter Version 3 spec (modal groups, Table 4)
//!   - grblHAL's gc_state / gcode.c as a live reference implementation
//!   - the `gcode` 0.7 crate's zero-allocation visitor trait API
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
//!   `Interpreter`      (impl ProgramVisitor) - owns ModalState, the only
//!                       long-lived state; persists across the whole
//!                       parse / across many lines.
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
//! resolved line becomes a `ResolvedMotionCommand` and gets handed to
//! whatever consumes this crate's output (a planner, a buffer, a test).
//! See `visitor.rs`'s module docs for the full detail.
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
//! - Canned cycles (G73, G81-G89), G10 (set coordinate data - the
//!   in-program alternative to a host preloading
//!   `ModalState::set_coordinate_system_offset`), G92.2/G92.3 (suspend
//!   and restore - only G92's set and G92.1's cancel are implemented),
//!   tool length offset resolution, cutter compensation, threading,
//!   splines. These are real parts of the NIST spec but represent a
//!   large amount of additional surface area we are deliberately
//!   staging for later, per our explicit decision to target "core
//!   motion semantics first." (G28/G30/G53 - machine-coordinate and
//!   predefined-position moves - ARE implemented; see `visitor` module
//!   docs.)
//! - Modal-group conflict detection is started here but intentionally
//!   minimal; see `modal_groups` module docs for current coverage.

#![cfg_attr(not(test), no_std)]

pub mod command;
pub mod modal_groups;
pub mod motion;
pub mod state;
pub mod visitor;

pub use command::{Command, CommandSink, CoolantCommand, ProgramFlow, SpindleCommand};
pub use modal_groups::{ModalGroup, ModalGroupSet};
pub use motion::{ArcError, ArcGeometry, MotionMode, ResolvedMotionCommand};
pub use state::{CoordinateSystem, DistanceMode, ModalState, Plane, Position, Units};
pub use visitor::{ErrorSink, InterpretError, Interpreter, MotionSink};
