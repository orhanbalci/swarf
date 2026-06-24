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
//!   - the `gcode` crate's actual zero-allocation visitor trait API
//!
//! # Architecture
//!
//! We build directly against `gcode::core`'s zero-allocation visitor
//! traits (`ProgramVisitor` / `BlockVisitor` / `CommandVisitor`), not the
//! higher-level `gcode::parse()` AST API, because the embedded firmware
//! target this is ultimately destined for cannot assume a heap allocator.
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
//!
//! # What this crate deliberately does NOT do (yet)
//!
//! - Parameter / expression evaluation (`#1`, `#<expr>`). The `gcode`
//!   crate hands these to us as `Value::Variable(&str)`, unevaluated by
//!   design (see its docs). We reject them with a clear diagnostic for
//!   now rather than silently defaulting them to zero or similar - that
//!   would be a correctness footgun for anyone using this crate without
//!   reading this note. Parameter support is real, deferrable scope.
//! - Canned cycles (G73, G81-G89), tool length offset resolution, cutter
//!   compensation, threading, splines. These are real parts of the NIST
//!   spec but represent a large amount of additional surface area we are
//!   deliberately staging for later, per our explicit decision to target
//!   "core motion semantics first."
//! - Modal-group conflict detection is started here but intentionally
//!   minimal; see `modal_groups` module docs for current coverage.
//!
//! # Toolchain / dependency note (read before extending this crate)
//!
//! This crate is currently pinned to `gcode = "0.6"`, NOT the `gcode`
//! 0.7 zero-allocation visitor API we originally designed against. 0.7
//! requires a 1.85+ / edition-2024 toolchain; this was verified directly
//! against the environment this was scaffolded in (rustc 1.75.0, which
//! fails even to parse 0.7's Cargo.toml). See `visitor.rs`'s module docs
//! for the full explanation of what changed in the implementation as a
//! result, and Cargo.toml for the action item to revisit this once a
//! newer toolchain is available. Consequently this crate currently
//! depends on `std` (via gcode's `std` feature) rather than being
//! `no_std` - that is also an action item for the 0.7 port, not a
//! permanent design decision.

pub mod modal_groups;
pub mod motion;
pub mod state;
pub mod visitor;

pub use modal_groups::{ModalGroup, ModalGroupSet};
pub use motion::{ArcGeometry, MotionMode, ResolvedMotionCommand};
pub use state::{DistanceMode, ModalState, Plane, Position, Units};
pub use visitor::{InterpretError, Interpreter, MotionSink};
