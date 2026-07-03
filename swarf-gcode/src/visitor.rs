//! The interpreter itself, built against `gcode` 0.7's `core`
//! zero-allocation visitor API (`ProgramVisitor` / `BlockVisitor` /
//! `CommandVisitor`) - see `lib.rs`'s module docs for why this API was
//! chosen over the higher-level, allocating `gcode::parse()`.
//!
//! # Structure
//!
//! Three borrow levels, each holding a mutable borrow back into the one
//! real state owner (`ModalState`):
//!
//!   - `Interpreter` (impl `ProgramVisitor`) owns `ModalState` and the
//!     three output sinks (`MotionSink`, `CommandSink`, `ErrorSink`);
//!     persists across the whole parse.
//!   - `BlockCtx<'a>` (impl `BlockVisitor`) is per-line scratch: which
//!     axis/word values were seen, which modal groups were touched this
//!     line (for conflict detection), which non-motion effects were
//!     requested. Borrows `&'a mut Interpreter`.
//!   - `CommandCtx<'a, 'b>` (impl `CommandVisitor`) is per-command
//!     scratch for a G/M/T word's own arguments (e.g. the `X10 Y20`
//!     following a `G1`); shares the SAME pending-word accumulator as
//!     its parent `BlockCtx` rather than keeping its own.
//!
//! This mirrors `core::parse`'s call order: a block produces zero or
//! more commands, each command receives zero or more arguments via
//! `argument`, then `end_command` returns control to the block; after
//! all commands, `end_line` returns control to the program and is where
//! a fully resolved line turns into output.
//!
//! # Why axis words are pooled at the block level
//!
//! Bare axis words with no G/M/T word on their own line arrive via
//! `BlockVisitor::word_address`; a command's own arguments arrive via
//! `CommandVisitor::argument`, scoped to that command. Both are pooled
//! into the SAME shared fields on `BlockCtx` (not kept separate),
//! because NIST semantics treat a line's axis words as one pool that
//! applies to whichever motion mode is active for that line - not as
//! data belonging to whichever G-word happens to precede it
//! syntactically (e.g. `"G1 G91 X10"`: X is the line's target, not
//! specifically G91's argument).
//!
//! # Non-motion output
//!
//! Spindle, coolant, tool change, dwell, and program-flow effects are
//! resolved the same way as motion, but pushed to `CommandSink` as a
//! `Command` (Interface 3, `command.rs`) instead of a
//! `ResolvedMotionCommand` - see that module's docs for why they're a
//! separate sink rather than folded into the motion stream.
//!
//! # Allocation
//!
//! This module (and the crate as a whole) is allocation-free: `gcode`
//! 0.7's `core` module has no `alloc` dependency, and neither do we -
//! per-line state is a handful of `Option` fields on `BlockCtx`, not a
//! `Vec`.

use core::num::ParseIntError;

use gcode::core::{
    parse, BlockVisitor, CommandVisitor, ControlFlow, Diagnostics as CoreDiagnostics, Noop, Number,
    ProgramVisitor, Span, TokenType, Value,
};

use crate::command::{Command, CommandSink, CoolantCommand, ProgramFlow, SpindleCommand};
use crate::modal_groups::{
    classify_general_code, classify_miscellaneous_code, ModalGroup, ModalGroupSet,
};
use crate::motion::{resolve_arc_center, ArcError, ArcGeometry, MotionMode, ResolvedMotionCommand};
use crate::state::{CoordinateSystem, DistanceMode, ModalState, Plane, Position, Units};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpretError {
    /// A bare axis-word line appeared with no motion mode active -
    /// either no G0/G1/G2/G3 has ever been seen, or the last motion
    /// word seen was G80 (cancel). NIST treats both as the same error:
    /// there is no active motion mode to carry the axis words forward
    /// into.
    NoActiveMotionMode,
    /// Two G or M words from the same NIST modal group appeared on one
    /// line (e.g. "G0 G1 X10").
    ModalGroupConflict,
    /// M6 appeared with no T word ever having selected a tool.
    NoToolSelected,
    /// A G2/G3 line's I/J/K/R data could not be turned into a valid arc
    /// - see `motion::ArcError` for the specific reason.
    InvalidArc(ArcError),
    /// A canned cycle (G81-G89) needed its Z (depth), R (retract plane),
    /// or P (dwell, G82/G89 only) value, and none was ever given. These
    /// are sticky (see `state::CannedCycleParams`) but still require a
    /// first value before a cycle can run.
    CannedCycleMissingParameter,
    /// G83 (peck drilling) requires a positive Q (peck increment); one
    /// was missing, zero, or negative.
    InvalidPeckIncrement,
    /// A canned cycle (G81-G89) was invoked while G18 or G19 was the
    /// active plane. Only G17 (XY, Z as the drilling axis) is
    /// implemented - real scope decision, see `lib.rs` module docs.
    UnsupportedCannedCyclePlane,
    /// A `#...` parameter reference or expression appeared as a
    /// `Value::Variable` rather than a literal. This crate does not
    /// evaluate parameters/expressions (see `lib.rs` module docs).
    /// NOTE: this `gcode` 0.7.0 release never actually constructs
    /// `Value::Variable` (verified directly) - `#` currently surfaces as
    /// `InterpretError::SyntaxError` instead. This variant is kept ready
    /// for when a future release implements parameter parsing.
    UnsupportedSyntax,
    /// Any other syntax the underlying parser could not interpret
    /// (unknown tokens, malformed numbers, etc.), reported via `gcode`
    /// 0.7's `Diagnostics` trait. Kept as a single variant carrying only
    /// a `Span` (not the offending text) to stay allocation-free.
    SyntaxError(Span),
}

/// Which stored reference position a G28/G30 (or G28.1/G30.1) line
/// refers to - see `ModalState::g28_position` / `g30_position`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HomeReference {
    G28,
    G30,
}

/// The words a canned-cycle line (G81-G89) can carry, bundled into one
/// value so `execute_canned_cycle` takes a reasonable number of
/// arguments. Mirrors the relevant subset of `BlockCtx`'s per-line
/// scratch fields.
#[derive(Debug, Clone, Copy, Default)]
struct CannedCycleWords {
    x: Option<f32>,
    y: Option<f32>,
    z: Option<f32>,
    r: Option<f32>,
    q: Option<f32>,
    p: Option<f32>,
}

/// Sink for fully resolved motion commands.
pub trait MotionSink {
    fn push(&mut self, command: ResolvedMotionCommand) -> Result<(), ()>;
}

/// Sink for interpretation errors - kept as a caller-supplied trait
/// (mirroring `MotionSink`) rather than an internal `Vec`, so this
/// crate never allocates: a `no_std` caller can log/count errors in
/// place, while a `std` caller can still trivially collect them into a
/// `Vec<InterpretError>` via their own sink implementation.
pub trait ErrorSink {
    fn push(&mut self, error: InterpretError);
}

/// The interpreter. Owns the one persistent piece of state
/// (`ModalState`), a motion sink, a non-motion command sink, and an
/// error sink.
pub struct Interpreter<M: MotionSink, C: CommandSink, E: ErrorSink> {
    pub state: ModalState,
    sink: M,
    commands: C,
    errors: E,
}

impl<M: MotionSink, C: CommandSink, E: ErrorSink> Interpreter<M, C, E> {
    pub fn new(sink: M, commands: C, errors: E) -> Self {
        Self {
            state: ModalState::new(),
            sink,
            commands,
            errors,
        }
    }

    /// Parse and interpret an entire source string.
    pub fn run(&mut self, src: &str) {
        parse(src, self);
    }

    /// Consume the interpreter, handing back the sinks it was built
    /// with - typically to inspect what a `std`-based test/caller
    /// collected into them.
    pub fn into_sinks(self) -> (M, C, E) {
        (self.sink, self.commands, self.errors)
    }

    /// Resolve one axis word against `current` (the position to fall
    /// back to when no word was given) and `offset` (the active work/G92
    /// offset for that axis), honoring G90/G91. The building block both
    /// `resolve_target_from_values` and the canned-cycle machinery
    /// (`execute_canned_cycle`) use for every axis they touch.
    fn resolve_axis_word(&self, current: f64, offset: f64, word: Option<f32>) -> f64 {
        match (word, self.state.distance_mode) {
            (Some(v), DistanceMode::Absolute) => self.state.to_mm(v as f64) + offset,
            (Some(v), DistanceMode::Incremental) => current + self.state.to_mm(v as f64),
            (None, _) => current,
        }
    }

    /// Resolve a line's X/Y/Z words into an absolute target.
    ///
    /// `machine_coordinates` is G53's line-local override: when true,
    /// work/G92 offsets are ignored and every given word is read as an
    /// absolute machine coordinate regardless of the active G90/G91
    /// distance mode - both per NIST's definition of G53.
    fn resolve_target_from_values(
        &self,
        x: Option<f32>,
        y: Option<f32>,
        z: Option<f32>,
        machine_coordinates: bool,
    ) -> Position {
        let offset = if machine_coordinates {
            Position::default()
        } else {
            self.state.active_offset()
        };
        let combine = |current: f64, offset: f64, word: Option<f32>| -> f64 {
            if machine_coordinates {
                return match word {
                    Some(v) => self.state.to_mm(v as f64),
                    None => current,
                };
            }
            self.resolve_axis_word(current, offset, word)
        };

        Position {
            x: combine(self.state.position.x, offset.x, x),
            y: combine(self.state.position.y, offset.y, y),
            z: combine(self.state.position.z, offset.z, z),
        }
    }

    /// Push a rapid move from `start` to `target` at the active feed
    /// rate field (meaningless for rapids, kept for
    /// `ResolvedMotionCommand` field uniformity - see its docs).
    fn push_rapid(&mut self, start: Position, target: Position) {
        let _ = self.sink.push(ResolvedMotionCommand {
            start,
            target,
            motion_mode: MotionMode::Rapid,
            arc: None,
            feed_rate: self.state.feed_rate,
        });
    }

    /// Push a linear (feed-rate-coordinated) move from `start` to
    /// `target`.
    fn push_linear(&mut self, start: Position, target: Position) {
        let _ = self.sink.push(ResolvedMotionCommand {
            start,
            target,
            motion_mode: MotionMode::Linear,
            arc: None,
            feed_rate: self.state.feed_rate,
        });
    }

    /// Execute one canned-cycle hole: resolve the hole's X/Y and the
    /// (sticky) Z/R/Q/P parameters, then push the rapid/feed/rapid (or
    /// feed, for G85/G89's retract) legs described by `kind`. See
    /// `motion::MotionMode`'s canned-cycle variants and `state`'s
    /// `CannedCycleParams` docs for what each parameter means.
    ///
    /// Scope: only the G17 (XY) plane is supported - Z is always the
    /// drilling axis. L (repeat count) is not implemented; each line
    /// drills exactly one hole. See `lib.rs` module docs.
    fn execute_canned_cycle(
        &mut self,
        kind: MotionMode,
        words: CannedCycleWords,
    ) -> Result<(), InterpretError> {
        if self.state.plane != Plane::Xy {
            return Err(InterpretError::UnsupportedCannedCyclePlane);
        }

        let start = self.state.position;
        let offset = self.state.active_offset();

        let hole_x = self.resolve_axis_word(start.x, offset.x, words.x);
        let hole_y = self.resolve_axis_word(start.y, offset.y, words.y);

        if let Some(z) = words.z {
            self.state.canned_cycle.z = Some(self.resolve_axis_word(start.z, offset.z, Some(z)));
        }
        let target_z = self
            .state
            .canned_cycle
            .z
            .ok_or(InterpretError::CannedCycleMissingParameter)?;

        if let Some(r) = words.r {
            self.state.canned_cycle.r = Some(self.resolve_axis_word(start.z, offset.z, Some(r)));
        }
        let r_plane = self
            .state
            .canned_cycle
            .r
            .ok_or(InterpretError::CannedCycleMissingParameter)?;

        if let Some(q) = words.q {
            self.state.canned_cycle.q = Some(self.state.to_mm(q as f64));
        }
        if let Some(p) = words.p {
            self.state.canned_cycle.p = Some(p as f64);
        }

        // Validate everything this cycle needs BEFORE pushing any
        // motion, so a bad line (e.g. G83 with no valid Q) produces no
        // partial output - consistent with how an invalid arc (see
        // `resolve_arc_center`) never emits a half-resolved move either.
        let peck_q = if matches!(kind, MotionMode::PeckDrill) {
            Some(
                self.state
                    .canned_cycle
                    .q
                    .filter(|q| *q > 0.0)
                    .ok_or(InterpretError::InvalidPeckIncrement)?,
            )
        } else {
            None
        };
        if matches!(kind, MotionMode::DrillDwell | MotionMode::BoreDwellFeedOut)
            && self.state.canned_cycle.p.is_none()
        {
            return Err(InterpretError::CannedCycleMissingParameter);
        }

        // Step 1: rapid to the hole's X/Y at the current Z.
        let mut pos = start;
        let above_hole = Position {
            x: hole_x,
            y: hole_y,
            z: pos.z,
        };
        self.push_rapid(pos, above_hole);
        pos = above_hole;

        // Step 2: rapid down (or up) to the retract plane.
        let at_r = Position {
            x: hole_x,
            y: hole_y,
            z: r_plane,
        };
        self.push_rapid(pos, at_r);
        pos = at_r;

        // +1 if drilling moves Z upward from R to the target, -1 if
        // downward - keeps the peck loop and the G98 "higher of the
        // two" comparison below sign-agnostic.
        let sign: f64 = if target_z >= r_plane { 1.0 } else { -1.0 };

        // Step 3: cut to the bottom. Peck drilling repeats this in
        // `q`-sized bites with a full retract to R between each (to
        // clear chips); every other cycle cuts in a single pass.
        if let Some(q) = peck_q {
            let total_depth = (target_z - r_plane).abs();
            let mut travelled = 0.0;
            loop {
                travelled = (travelled + q).min(total_depth);
                let this_bottom = Position {
                    x: hole_x,
                    y: hole_y,
                    z: r_plane + sign * travelled,
                };
                self.push_linear(pos, this_bottom);
                pos = this_bottom;
                if travelled >= total_depth {
                    break;
                }
                self.push_rapid(pos, at_r);
                pos = at_r;
            }
        } else {
            let bottom = Position {
                x: hole_x,
                y: hole_y,
                z: target_z,
            };
            self.push_linear(pos, bottom);
            pos = bottom;
        }

        // Step 4: action at the bottom of the hole, if any.
        match kind {
            MotionMode::DrillDwell | MotionMode::BoreDwellFeedOut => {
                let seconds = self
                    .state
                    .canned_cycle
                    .p
                    .ok_or(InterpretError::CannedCycleMissingParameter)?;
                let _ = self.commands.push(Command::Dwell { seconds });
            }
            MotionMode::BoreSpindleStop => {
                let _ = self.commands.push(Command::Spindle(SpindleCommand::Stop));
            }
            _ => {}
        }

        // Step 5: retract - to R (G99), or the higher of R and the
        // pre-cycle Z (G98, NIST default); at the active feed rate for
        // G85/G89 (boring), rapid otherwise.
        let away_from_bottom = |v: f64| -sign * (v - target_z);
        let retract_z = if self.state.canned_cycle_return_to_initial_z
            && away_from_bottom(start.z) > away_from_bottom(r_plane)
        {
            start.z
        } else {
            r_plane
        };
        let retracted = Position {
            x: hole_x,
            y: hole_y,
            z: retract_z,
        };
        if matches!(kind, MotionMode::BoreFeedOut | MotionMode::BoreDwellFeedOut) {
            self.push_linear(pos, retracted);
        } else {
            self.push_rapid(pos, retracted);
        }
        self.state.position = retracted;
        self.state.motion_mode = kind;

        Ok(())
    }
}

impl<M: MotionSink, C: CommandSink, E: ErrorSink> CoreDiagnostics for Interpreter<M, C, E> {
    fn emit_unknown_content(&mut self, _text: &str, span: Span) {
        self.errors.push(InterpretError::SyntaxError(span));
    }

    fn emit_unexpected(&mut self, _actual: &str, _expected: &[TokenType], span: Span) {
        self.errors.push(InterpretError::SyntaxError(span));
    }

    // NOTE: `emit_parse_number_error` is intentionally not overridden -
    // `gcode` 0.7's `ParseNumberError` type is defined `pub` inside its
    // private `core::types` module and not re-exported, so it cannot be
    // named outside the crate (confirmed: `gcode`'s own `AstBuilder`
    // doesn't override this method either, for the same reason). The
    // trait's default no-op body applies instead.

    fn emit_parse_int_error(&mut self, _value: &str, _error: ParseIntError, span: Span) {
        self.errors.push(InterpretError::SyntaxError(span));
    }
}

impl<M: MotionSink, C: CommandSink, E: ErrorSink> ProgramVisitor for Interpreter<M, C, E> {
    fn start_block(&mut self) -> ControlFlow<impl BlockVisitor + '_> {
        ControlFlow::Continue(BlockCtx {
            interp: self,
            seen_groups: ModalGroupSet::new(),
            resolved_mode: None,
            x: None,
            y: None,
            z: None,
            f: None,
            p: None,
            s: None,
            i: None,
            j: None,
            k: None,
            r: None,
            q: None,
            dwell_requested: false,
            program_flow: None,
            spindle_word: None,
            coolant_word: None,
            tool_change_requested: false,
            tool_select: None,
            suppress_motion: false,
            g92_requested: false,
            g92_cancel_requested: false,
            home_request: None,
            set_home_reference: None,
            machine_coordinates: false,
        })
    }
}

/// Per-line scratch: which modal groups were touched this line (for
/// conflict detection), which motion mode (if any) this line's G-word
/// resolved to, the pooled axis/feed/dwell/spindle values seen anywhere
/// on the line (whether via a bare `word_address` or as an `argument`
/// of some command), and which non-motion effects (dwell, program flow,
/// spindle, coolant, tool change/select) were requested. Borrows the
/// one real state owner, `Interpreter`.
struct BlockCtx<'a, M: MotionSink, C: CommandSink, E: ErrorSink> {
    interp: &'a mut Interpreter<M, C, E>,
    seen_groups: ModalGroupSet,
    resolved_mode: Option<MotionMode>,
    x: Option<f32>,
    y: Option<f32>,
    z: Option<f32>,
    f: Option<f32>,
    /// P word - dwell time in seconds (G4).
    p: Option<f32>,
    /// S word - spindle speed in RPM, consumed by M3/M4.
    s: Option<f32>,
    /// I/J/K words - arc center offset from `start`, incremental
    /// (NIST's G91.1 default; see `motion::resolve_arc_center`).
    i: Option<f32>,
    j: Option<f32>,
    k: Option<f32>,
    /// R word - arc radius (G2/G3 radius form) or canned-cycle retract
    /// plane - unambiguous since a line's motion mode is exactly one of
    /// the two, never both.
    r: Option<f32>,
    /// Q word - canned-cycle peck increment (G83 only).
    q: Option<f32>,
    dwell_requested: bool,
    /// Major number of an M0/M1/M2/M30 word seen this line, if any.
    program_flow: Option<u32>,
    /// Major number of an M3/M4/M5 word seen this line, if any.
    spindle_word: Option<u32>,
    /// Major number of an M7/M8/M9 word seen this line, if any.
    coolant_word: Option<u32>,
    /// Whether an M6 word was seen this line.
    tool_change_requested: bool,
    /// Tool number selected by a T word on this line, if any.
    tool_select: Option<u32>,
    /// Set when a G-word appeared whose axis-like words are NOT motion
    /// targets (currently: G92/G92.1) - prevents this line's X/Y/Z from
    /// being misread as a move using whatever motion mode was last
    /// active.
    suppress_motion: bool,
    /// G92 appeared: X/Y/Z on this line (if any) redefine the G92
    /// offset rather than commanding a move.
    g92_requested: bool,
    /// G92.1 appeared: reset the G92 offset to zero.
    g92_cancel_requested: bool,
    /// G28 or G30 appeared: rapid (via any given axis words as an
    /// intermediate point) to the named stored reference position.
    home_request: Option<HomeReference>,
    /// G28.1 or G30.1 appeared: record the current machine position as
    /// the named stored reference position.
    set_home_reference: Option<HomeReference>,
    /// G53 appeared: this line's axis words are absolute machine
    /// coordinates, ignoring work/G92 offsets and G90/G91.
    machine_coordinates: bool,
}

impl<M: MotionSink, C: CommandSink, E: ErrorSink> BlockCtx<'_, M, C, E> {
    /// Record an X/Y/Z/F/P/S value into this line's shared pool,
    /// rejecting (with `UnsupportedSyntax`) anything that isn't a
    /// literal number - parameters/expressions are out of scope, see
    /// `InterpretError`.
    fn record_axis(&mut self, letter: char, value: Value<'_>) {
        let v = match value {
            Value::Literal(v) => v,
            Value::Variable(_) => {
                self.interp.errors.push(InterpretError::UnsupportedSyntax);
                return;
            }
        };
        match letter {
            'X' => self.x = Some(v),
            'Y' => self.y = Some(v),
            'Z' => self.z = Some(v),
            'F' => self.f = Some(v),
            'P' => self.p = Some(v),
            'S' => self.s = Some(v),
            'I' => self.i = Some(v),
            'J' => self.j = Some(v),
            'K' => self.k = Some(v),
            'R' => self.r = Some(v),
            'Q' => self.q = Some(v),
            _ => {}
        }
    }

    fn classify_and_record_group(&mut self, group: ModalGroup) {
        if self.seen_groups.contains(group) {
            self.interp.errors.push(InterpretError::ModalGroupConflict);
        }
        self.seen_groups.insert(group);
    }
}

impl<M: MotionSink, C: CommandSink, E: ErrorSink> CoreDiagnostics for BlockCtx<'_, M, C, E> {
    fn emit_unknown_content(&mut self, text: &str, span: Span) {
        self.interp.emit_unknown_content(text, span);
    }

    fn emit_unexpected(&mut self, actual: &str, expected: &[TokenType], span: Span) {
        self.interp.emit_unexpected(actual, expected, span);
    }

    fn emit_parse_int_error(&mut self, value: &str, error: ParseIntError, span: Span) {
        self.interp.emit_parse_int_error(value, error, span);
    }
}

impl<M: MotionSink, C: CommandSink, E: ErrorSink> BlockVisitor for BlockCtx<'_, M, C, E> {
    fn word_address(&mut self, letter: char, value: Value<'_>, _span: Span) {
        self.record_axis(letter, value);
    }

    fn start_general_code(&mut self, number: Number) -> ControlFlow<impl CommandVisitor + '_> {
        let major = number.major();
        let minor = number.minor().map(|n| n.get());

        match classify_general_code(major, minor) {
            Some(ModalGroup::Motion) => {
                self.classify_and_record_group(ModalGroup::Motion);
                self.resolved_mode = Some(match (major, minor) {
                    (0, None) => MotionMode::Rapid,
                    (1, None) => MotionMode::Linear,
                    (2, None) => MotionMode::ArcClockwise,
                    (3, None) => MotionMode::ArcCounterclockwise,
                    (81, None) => MotionMode::Drill,
                    (82, None) => MotionMode::DrillDwell,
                    (83, None) => MotionMode::PeckDrill,
                    (85, None) => MotionMode::BoreFeedOut,
                    (86, None) => MotionMode::BoreSpindleStop,
                    (89, None) => MotionMode::BoreDwellFeedOut,
                    _ => MotionMode::None, // G38.x probing, G80 cancel
                });
            }
            Some(ModalGroup::Plane) => {
                self.classify_and_record_group(ModalGroup::Plane);
                self.interp.state.plane = match major {
                    17 => Plane::Xy,
                    18 => Plane::Zx,
                    19 => Plane::Yz,
                    _ => self.interp.state.plane,
                };
            }
            Some(ModalGroup::Units) => {
                self.classify_and_record_group(ModalGroup::Units);
                self.interp.state.units = match major {
                    20 => Units::Inches,
                    21 => Units::Millimeters,
                    _ => self.interp.state.units,
                };
            }
            Some(ModalGroup::DistanceMode) => {
                self.classify_and_record_group(ModalGroup::DistanceMode);
                self.interp.state.distance_mode = match major {
                    90 => DistanceMode::Absolute,
                    91 => DistanceMode::Incremental,
                    _ => self.interp.state.distance_mode,
                };
            }
            Some(ModalGroup::NonModal) => {
                self.classify_and_record_group(ModalGroup::NonModal);
                match (major, minor) {
                    (4, None) => self.dwell_requested = true,
                    (92, None) => {
                        self.g92_requested = true;
                        self.suppress_motion = true;
                    }
                    (92, Some(1)) => {
                        self.g92_cancel_requested = true;
                        self.suppress_motion = true;
                    }
                    (28, None) => {
                        self.home_request = Some(HomeReference::G28);
                        self.suppress_motion = true;
                    }
                    (28, Some(1)) => {
                        self.set_home_reference = Some(HomeReference::G28);
                        self.suppress_motion = true;
                    }
                    (30, None) => {
                        self.home_request = Some(HomeReference::G30);
                        self.suppress_motion = true;
                    }
                    (30, Some(1)) => {
                        self.set_home_reference = Some(HomeReference::G30);
                        self.suppress_motion = true;
                    }
                    (53, None) => self.machine_coordinates = true,
                    _ => {}
                }
            }
            Some(ModalGroup::CoordinateSystem) => {
                self.classify_and_record_group(ModalGroup::CoordinateSystem);
                if let Some(cs) = match (major, minor) {
                    (54, None) => Some(CoordinateSystem::G54),
                    (55, None) => Some(CoordinateSystem::G55),
                    (56, None) => Some(CoordinateSystem::G56),
                    (57, None) => Some(CoordinateSystem::G57),
                    (58, None) => Some(CoordinateSystem::G58),
                    (59, None) => Some(CoordinateSystem::G59),
                    (59, Some(1)) => Some(CoordinateSystem::G59_1),
                    (59, Some(2)) => Some(CoordinateSystem::G59_2),
                    (59, Some(3)) => Some(CoordinateSystem::G59_3),
                    _ => None,
                } {
                    self.interp.state.coordinate_system = cs;
                }
            }
            Some(ModalGroup::CannedCycleReturnMode) => {
                self.classify_and_record_group(ModalGroup::CannedCycleReturnMode);
                self.interp.state.canned_cycle_return_to_initial_z = major == 98;
            }
            Some(group) => self.classify_and_record_group(group),
            None => {}
        }

        ControlFlow::Continue(CommandCtx { block: self })
    }

    fn start_miscellaneous_code(
        &mut self,
        number: Number,
    ) -> ControlFlow<impl CommandVisitor + '_> {
        let major = number.major();
        let minor = number.minor().map(|n| n.get());
        if let Some(group) = classify_miscellaneous_code(major, minor) {
            self.classify_and_record_group(group);
            match group {
                ModalGroup::ProgramStopping => self.program_flow = Some(major),
                ModalGroup::SpindleTurning => self.spindle_word = Some(major),
                ModalGroup::CoolantControl => self.coolant_word = Some(major),
                ModalGroup::ToolChange => self.tool_change_requested = true,
                _ => {}
            }
        }
        ControlFlow::Continue(CommandCtx { block: self })
    }

    fn start_tool_change_code(&mut self, number: Number) -> ControlFlow<impl CommandVisitor + '_> {
        self.tool_select = Some(number.major());
        ControlFlow::Continue(Noop)
    }

    fn end_line(self, _span: Span) {
        let effective_mode = self.resolved_mode.unwrap_or(self.interp.state.motion_mode);
        let is_arc = matches!(
            effective_mode,
            MotionMode::ArcClockwise | MotionMode::ArcCounterclockwise
        );
        let is_canned_cycle = matches!(
            effective_mode,
            MotionMode::Drill
                | MotionMode::DrillDwell
                | MotionMode::PeckDrill
                | MotionMode::BoreFeedOut
                | MotionMode::BoreSpindleStop
                | MotionMode::BoreDwellFeedOut
        );
        let has_axis_word = self.x.is_some() || self.y.is_some() || self.z.is_some();
        let has_arc_geometry =
            self.i.is_some() || self.j.is_some() || self.k.is_some() || self.r.is_some();
        // A canned cycle can be (re)triggered by a G8x word alone, with
        // no new axis words at all (e.g. "G81 R2 F100" re-drilling the
        // current X/Y at a new retract height) - unlike ordinary motion
        // codes, where a bare G-word with no axis data never produces a
        // move.
        let fresh_canned_cycle_word = is_canned_cycle && self.resolved_mode.is_some();
        // A full-circle arc (G2/G3 I.. J.. with no X/Y/Z) has no axis
        // words at all - target defaults to start, which
        // `resolve_target_from_values` already does for omitted words -
        // but it's still a real move, so it must not be swallowed by
        // the "nothing to resolve" branch below. `suppress_motion` is
        // the opposite case: axis-shaped words ARE present, but this
        // line's G-word (G92/G92.1, G28/G30, G28.1/G30.1) means they
        // aren't a motion target for the STANDARD resolution below - a
        // home move (G28/G30) still produces real motion, just through
        // its own dedicated handling further down in this function.
        let has_motion_data = !self.suppress_motion
            && (has_axis_word || (is_arc && has_arc_geometry) || fresh_canned_cycle_word);

        if !has_motion_data {
            // No axis data on this line: either a bare modal-mode word
            // (e.g. a lone "G1") to remember for later, or a line with
            // nothing motion-relevant at all (comment/M-code/G17-only) -
            // either way, there is no move to resolve.
            if let Some(mode) = self.resolved_mode {
                self.interp.state.motion_mode = mode;
            }
        } else if effective_mode == MotionMode::None {
            self.interp.errors.push(InterpretError::NoActiveMotionMode);
        } else if is_canned_cycle {
            let words = CannedCycleWords {
                x: self.x,
                y: self.y,
                z: self.z,
                r: self.r,
                q: self.q,
                p: self.p,
            };
            if let Err(e) = self.interp.execute_canned_cycle(effective_mode, words) {
                self.interp.errors.push(e);
            }
        } else {
            let start = self.interp.state.position;
            let target = self.interp.resolve_target_from_values(
                self.x,
                self.y,
                self.z,
                self.machine_coordinates,
            );

            let mut arc = None;
            let mut arc_failed = false;

            if is_arc {
                let ijk = if self.i.is_some() || self.j.is_some() || self.k.is_some() {
                    Some((
                        self.i.map_or(0.0, |v| self.interp.state.to_mm(v as f64)),
                        self.j.map_or(0.0, |v| self.interp.state.to_mm(v as f64)),
                        self.k.map_or(0.0, |v| self.interp.state.to_mm(v as f64)),
                    ))
                } else {
                    None
                };
                let r = self.r.map(|v| self.interp.state.to_mm(v as f64));

                match resolve_arc_center(
                    self.interp.state.plane,
                    start,
                    target,
                    ijk,
                    r,
                    effective_mode == MotionMode::ArcClockwise,
                ) {
                    Ok(center) => arc = Some(ArcGeometry { center }),
                    Err(e) => {
                        self.interp.errors.push(InterpretError::InvalidArc(e));
                        arc_failed = true;
                    }
                }
            }

            if !arc_failed {
                if let Some(f) = self.f {
                    self.interp.state.feed_rate = self.interp.state.to_mm(f as f64);
                }

                let command = ResolvedMotionCommand {
                    start,
                    target,
                    motion_mode: effective_mode,
                    arc,
                    feed_rate: self.interp.state.feed_rate,
                };

                self.interp.state.position = target;
                self.interp.state.motion_mode = effective_mode;

                let _ = self.interp.sink.push(command);
            }
        }

        if let Some(s) = self.s {
            self.interp.state.spindle_speed = s as f64;
        }

        if let Some(major) = self.spindle_word {
            let speed = self.interp.state.spindle_speed;
            let cmd = match major {
                3 => SpindleCommand::Clockwise(speed),
                4 => SpindleCommand::CounterClockwise(speed),
                _ => SpindleCommand::Stop, // M5
            };
            let _ = self.interp.commands.push(Command::Spindle(cmd));
        }

        if let Some(major) = self.coolant_word {
            let cmd = match major {
                7 => CoolantCommand::Mist,
                8 => CoolantCommand::Flood,
                _ => CoolantCommand::Off, // M9
            };
            let _ = self.interp.commands.push(Command::Coolant(cmd));
        }

        if let Some(major) = self.program_flow {
            let flow = match major {
                0 => ProgramFlow::Stop,
                1 => ProgramFlow::OptionalStop,
                2 => ProgramFlow::End,
                _ => ProgramFlow::EndAndRewind, // M30
            };
            let _ = self.interp.commands.push(Command::ProgramFlow(flow));
        }

        if self.tool_select.is_some() {
            self.interp.state.selected_tool = self.tool_select;
        }

        if self.dwell_requested {
            let seconds = self.p.unwrap_or(0.0) as f64;
            let _ = self.interp.commands.push(Command::Dwell { seconds });
        }

        if self.tool_change_requested {
            match self.interp.state.selected_tool {
                Some(tool) => {
                    let _ = self.interp.commands.push(Command::ToolChange { tool });
                }
                None => self.interp.errors.push(InterpretError::NoToolSelected),
            }
        }

        if self.g92_requested {
            // For each axis word given, solve for the G92 offset
            // component that makes the CURRENT machine position equal
            // the requested value in the active coordinate system - see
            // `motion.rs`'s docs / `ModalState::active_offset`. Axes
            // with no word keep their previous G92 offset unchanged.
            let cs_offset = self
                .interp
                .state
                .coordinate_system_offset(self.interp.state.coordinate_system);
            if let Some(v) = self.x {
                let want = self.interp.state.to_mm(v as f64);
                self.interp.state.g92_offset.x = self.interp.state.position.x - cs_offset.x - want;
            }
            if let Some(v) = self.y {
                let want = self.interp.state.to_mm(v as f64);
                self.interp.state.g92_offset.y = self.interp.state.position.y - cs_offset.y - want;
            }
            if let Some(v) = self.z {
                let want = self.interp.state.to_mm(v as f64);
                self.interp.state.g92_offset.z = self.interp.state.position.z - cs_offset.z - want;
            }
        }

        if self.g92_cancel_requested {
            self.interp.state.g92_offset = Position::default();
        }

        if let Some(reference) = self.home_request {
            // G28/G30: rapid through any given axis words as an
            // intermediate point (normal work-offset resolution, same
            // as any other move), then rapid on to the stored reference
            // position (raw machine coordinates, per NIST). Neither
            // move updates `motion_mode` - G28/G30 are non-modal and do
            // not become the carried-forward mode for later lines.
            let mut leg_start = self.interp.state.position;

            if has_axis_word {
                let intermediate = self
                    .interp
                    .resolve_target_from_values(self.x, self.y, self.z, false);
                let _ = self.interp.sink.push(ResolvedMotionCommand {
                    start: leg_start,
                    target: intermediate,
                    motion_mode: MotionMode::Rapid,
                    arc: None,
                    feed_rate: self.interp.state.feed_rate,
                });
                self.interp.state.position = intermediate;
                leg_start = intermediate;
            }

            let reference_position = match reference {
                HomeReference::G28 => self.interp.state.g28_position,
                HomeReference::G30 => self.interp.state.g30_position,
            };
            let _ = self.interp.sink.push(ResolvedMotionCommand {
                start: leg_start,
                target: reference_position,
                motion_mode: MotionMode::Rapid,
                arc: None,
                feed_rate: self.interp.state.feed_rate,
            });
            self.interp.state.position = reference_position;
        }

        if let Some(reference) = self.set_home_reference {
            match reference {
                HomeReference::G28 => self.interp.state.g28_position = self.interp.state.position,
                HomeReference::G30 => self.interp.state.g30_position = self.interp.state.position,
            }
        }
    }
}

/// Per-command scratch, sharing the SAME pending-axis accumulator as
/// its parent `BlockCtx` - a command's arguments (e.g. the `X10 Y20`
/// following a `G1`, or the `S1000` following an `M3`) are pooled into
/// the line's data exactly like a bare `word_address` would be, per
/// NIST's line-is-the-unit semantics (see this module's docs).
struct CommandCtx<'a, 'b, M: MotionSink, C: CommandSink, E: ErrorSink> {
    block: &'a mut BlockCtx<'b, M, C, E>,
}

impl<M: MotionSink, C: CommandSink, E: ErrorSink> CoreDiagnostics for CommandCtx<'_, '_, M, C, E> {
    fn emit_unknown_content(&mut self, text: &str, span: Span) {
        self.block.emit_unknown_content(text, span);
    }

    fn emit_unexpected(&mut self, actual: &str, expected: &[TokenType], span: Span) {
        self.block.emit_unexpected(actual, expected, span);
    }

    fn emit_parse_int_error(&mut self, value: &str, error: ParseIntError, span: Span) {
        self.block.emit_parse_int_error(value, error, span);
    }
}

impl<M: MotionSink, C: CommandSink, E: ErrorSink> CommandVisitor for CommandCtx<'_, '_, M, C, E> {
    fn argument(&mut self, letter: char, value: Value<'_>, _span: Span) {
        self.block.record_axis(letter, value);
    }

    fn end_command(self, _span: Span) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct CollectingSink {
        commands: std::vec::Vec<ResolvedMotionCommand>,
    }

    impl MotionSink for CollectingSink {
        fn push(&mut self, command: ResolvedMotionCommand) -> Result<(), ()> {
            self.commands.push(command);
            Ok(())
        }
    }

    #[derive(Default)]
    struct CollectingCommands {
        commands: std::vec::Vec<Command>,
    }

    impl CommandSink for CollectingCommands {
        fn push(&mut self, command: Command) -> Result<(), ()> {
            self.commands.push(command);
            Ok(())
        }
    }

    #[derive(Default)]
    struct CollectingErrors {
        errors: std::vec::Vec<InterpretError>,
    }

    impl ErrorSink for CollectingErrors {
        fn push(&mut self, error: InterpretError) {
            self.errors.push(error);
        }
    }

    fn run(src: &str) -> (CollectingSink, CollectingCommands, CollectingErrors) {
        let mut interp = Interpreter::new(
            CollectingSink::default(),
            CollectingCommands::default(),
            CollectingErrors::default(),
        );
        interp.run(src);
        interp.into_sinks()
    }

    #[test]
    fn simple_linear_move_resolves_to_one_command() {
        let (sink, _, _) = run("G1 X10 Y20\n");
        assert_eq!(sink.commands.len(), 1);
        let cmd = sink.commands[0];
        assert_eq!(cmd.motion_mode, MotionMode::Linear);
        assert_eq!(cmd.target.x, 10.0);
        assert_eq!(cmd.target.y, 20.0);
        assert_eq!(cmd.start, Position::default());
    }

    #[test]
    fn modal_carry_forward_reuses_previous_motion_mode() {
        let (sink, _, _) = run("G1 X10\nY20\n");
        assert_eq!(sink.commands.len(), 2);
        assert_eq!(sink.commands[1].motion_mode, MotionMode::Linear);
        assert_eq!(sink.commands[1].target.y, 20.0);
        assert_eq!(sink.commands[1].target.x, 10.0);
    }

    #[test]
    fn bare_axis_words_before_any_command_have_no_active_motion_mode() {
        let (sink, _, errors) = run("X10\n");
        assert_eq!(sink.commands.len(), 0);
        assert!(errors.errors.contains(&InterpretError::NoActiveMotionMode));
    }

    #[test]
    fn incremental_distance_mode_adds_to_current_position() {
        let (sink, _, _) = run("G1 G91 X10\nX10\n");
        assert_eq!(sink.commands.len(), 2);
        assert_eq!(sink.commands[0].target.x, 10.0);
        assert_eq!(sink.commands[1].target.x, 20.0);
    }

    #[test]
    fn conflicting_motion_words_on_one_line_is_detected() {
        let (_, _, errors) = run("G0 G1 X10\n");
        assert!(errors.errors.contains(&InterpretError::ModalGroupConflict));
    }

    #[test]
    fn plane_and_motion_words_coexist_without_conflict() {
        let (sink, _, errors) = run("G17 G1 X10\n");
        assert!(!errors.errors.contains(&InterpretError::ModalGroupConflict));
        assert_eq!(sink.commands.len(), 1);
    }

    #[test]
    fn units_conversion_applies_to_subsequent_literals() {
        let (sink, _, _) = run("G20 G1 X1\n");
        assert_eq!(sink.commands.len(), 1);
        assert!((sink.commands[0].target.x - 25.4).abs() < 1e-6);
    }

    #[test]
    fn rapid_and_linear_are_distinct_modes_in_output() {
        let (sink, _, _) = run("G0 X5\nG1 X10\n");
        assert_eq!(sink.commands.len(), 2);
        assert_eq!(sink.commands[0].motion_mode, MotionMode::Rapid);
        assert_eq!(sink.commands[1].motion_mode, MotionMode::Linear);
    }

    #[test]
    fn multiple_lines_accumulate_position_correctly() {
        let (sink, _, _) = run("G1 X10 Y0\nX20 Y10\nX0 Y0\n");
        assert_eq!(sink.commands.len(), 3);
        assert_eq!(
            sink.commands[0].target,
            Position {
                x: 10.0,
                y: 0.0,
                z: 0.0
            }
        );
        assert_eq!(
            sink.commands[1].target,
            Position {
                x: 20.0,
                y: 10.0,
                z: 0.0
            }
        );
        assert_eq!(
            sink.commands[2].target,
            Position {
                x: 0.0,
                y: 0.0,
                z: 0.0
            }
        );
    }

    #[test]
    fn parameter_reference_is_rejected_as_a_syntax_error() {
        // `#1` is NOT surfaced as `Value::Variable` by this gcode 0.7.0
        // release - verified directly: nothing in its lexer/parser ever
        // constructs that variant, despite it existing in the `Value`
        // enum. `#` is simply an unrecognised character, so each of its
        // characters is reported via `emit_unknown_content` and shows up
        // here as `SyntaxError`, not `UnsupportedSyntax`. The
        // `Value::Variable` handling in `record_axis` is kept as
        // forward-compatible dead code for if/when a future `gcode`
        // release actually implements parameter parsing.
        let (sink, _, errors) = run("G1 X#1\n");
        assert!(sink.commands.is_empty());
        assert!(!errors.errors.is_empty());
        assert!(errors
            .errors
            .iter()
            .all(|e| matches!(e, InterpretError::SyntaxError(_))));
    }

    #[test]
    fn spindle_on_with_speed_is_resolved() {
        let (_, commands, _) = run("M3 S1000\n");
        assert_eq!(
            commands.commands,
            std::vec![Command::Spindle(SpindleCommand::Clockwise(1000.0))]
        );
    }

    #[test]
    fn spindle_speed_is_modal_across_lines() {
        let (_, commands, _) = run("S500\nM3\nM5\nM4\n");
        assert_eq!(
            commands.commands,
            std::vec![
                Command::Spindle(SpindleCommand::Clockwise(500.0)),
                Command::Spindle(SpindleCommand::Stop),
                Command::Spindle(SpindleCommand::CounterClockwise(500.0)),
            ]
        );
    }

    #[test]
    fn coolant_commands_are_resolved() {
        let (_, commands, _) = run("M8\nM9\n");
        assert_eq!(
            commands.commands,
            std::vec![
                Command::Coolant(CoolantCommand::Flood),
                Command::Coolant(CoolantCommand::Off),
            ]
        );
    }

    #[test]
    fn dwell_reads_p_word_in_seconds() {
        let (_, commands, _) = run("G4 P1.5\n");
        assert_eq!(
            commands.commands,
            std::vec![Command::Dwell { seconds: 1.5 }]
        );
    }

    #[test]
    fn program_flow_codes_are_resolved() {
        let (_, commands, _) = run("M0\nM1\nM2\nM30\n");
        assert_eq!(
            commands.commands,
            std::vec![
                Command::ProgramFlow(ProgramFlow::Stop),
                Command::ProgramFlow(ProgramFlow::OptionalStop),
                Command::ProgramFlow(ProgramFlow::End),
                Command::ProgramFlow(ProgramFlow::EndAndRewind),
            ]
        );
    }

    #[test]
    fn tool_select_then_change_resolves_with_selected_tool_number() {
        let (_, commands, _) = run("T4\nM6\n");
        assert_eq!(
            commands.commands,
            std::vec![Command::ToolChange { tool: 4 }]
        );
    }

    #[test]
    fn tool_change_without_prior_selection_is_an_error() {
        let (_, commands, errors) = run("M6\n");
        assert!(commands.commands.is_empty());
        assert!(errors.errors.contains(&InterpretError::NoToolSelected));
    }

    #[test]
    fn tool_select_and_change_on_the_same_line_resolves_immediately() {
        let (_, commands, _) = run("T5 M6\n");
        assert_eq!(
            commands.commands,
            std::vec![Command::ToolChange { tool: 5 }]
        );
    }

    #[test]
    fn ccw_arc_by_radius_resolves_the_minor_arc_center() {
        // Worked example (see motion.rs docs): quarter circle from
        // (10,0) to (0,10), r=10. CCW's minor-arc center is the origin.
        let (sink, _, errors) = run("G0 X10 Y0\nG3 X0 Y10 R10\n");
        assert!(errors.errors.is_empty());
        let arc = sink.commands[1].arc.expect("arc geometry");
        assert!((arc.center.x - 0.0).abs() < 1e-9);
        assert!((arc.center.y - 0.0).abs() < 1e-9);
    }

    #[test]
    fn cw_arc_by_radius_resolves_the_other_minor_arc_center() {
        // Same chord, opposite direction: CW's minor-arc center is
        // (10,10), not the origin - see motion.rs's worked example.
        let (sink, _, errors) = run("G0 X10 Y0\nG2 X0 Y10 R10\n");
        assert!(errors.errors.is_empty());
        let arc = sink.commands[1].arc.expect("arc geometry");
        assert!((arc.center.x - 10.0).abs() < 1e-9);
        assert!((arc.center.y - 10.0).abs() < 1e-9);
    }

    #[test]
    fn arc_by_ijk_center_is_start_plus_offset() {
        let (sink, _, errors) = run("G2 X10 Y0 I5 J0\n");
        assert!(errors.errors.is_empty());
        let arc = sink.commands[0].arc.expect("arc geometry");
        assert_eq!(arc.center.x, 5.0);
        assert_eq!(arc.center.y, 0.0);
    }

    #[test]
    fn full_circle_via_ijk_with_no_axis_words_targets_the_start_point() {
        let (sink, _, errors) = run("G0 X10 Y0\nG2 I-10 J0\n");
        assert!(errors.errors.is_empty());
        assert_eq!(sink.commands.len(), 2);
        let full_circle = sink.commands[1];
        assert_eq!(full_circle.target, full_circle.start);
        let arc = full_circle.arc.expect("arc geometry");
        assert!((arc.center.x - 0.0).abs() < 1e-9);
        assert!((arc.center.y - 0.0).abs() < 1e-9);
    }

    #[test]
    fn arc_with_neither_ijk_nor_r_is_a_missing_geometry_error() {
        let (sink, _, errors) = run("G2 X10 Y0\n");
        assert!(sink.commands.is_empty());
        assert!(errors
            .errors
            .contains(&InterpretError::InvalidArc(ArcError::MissingGeometry)));
    }

    #[test]
    fn radius_smaller_than_half_the_chord_is_an_error() {
        // Chord length is 10; no circle of radius 1 passes through
        // points 10 apart.
        let (sink, _, errors) = run("G2 X10 Y0 R1\n");
        assert!(sink.commands.is_empty());
        assert!(errors
            .errors
            .contains(&InterpretError::InvalidArc(ArcError::RadiusTooSmall)));
    }

    #[test]
    fn radius_form_with_coincident_start_and_end_is_an_error() {
        let (sink, _, errors) = run("G2 R5\n");
        assert!(sink.commands.is_empty());
        assert!(errors
            .errors
            .contains(&InterpretError::InvalidArc(ArcError::CoincidentEndpoints)));
    }

    #[test]
    fn ccw_arc_in_zx_plane_uses_the_same_cyclic_convention_as_xy() {
        let (sink, _, errors) = run("G18\nG0 Z10 X0\nG3 Z0 X10 R10\n");
        assert!(errors.errors.is_empty());
        let ccw = sink.commands[1].arc.expect("arc geometry");
        assert!((ccw.center.z - 0.0).abs() < 1e-9);
        assert!((ccw.center.x - 0.0).abs() < 1e-9);
    }

    #[test]
    fn cw_arc_in_zx_plane_uses_the_same_cyclic_convention_as_xy() {
        let (sink, _, errors) = run("G18\nG0 Z10 X0\nG2 Z0 X10 R10\n");
        assert!(errors.errors.is_empty());
        let cw = sink.commands[1].arc.expect("arc geometry");
        assert!((cw.center.z - 10.0).abs() < 1e-9);
        assert!((cw.center.x - 10.0).abs() < 1e-9);
    }

    #[test]
    fn ccw_arc_in_yz_plane_uses_the_same_cyclic_convention_as_xy() {
        let (sink, _, errors) = run("G19\nG0 Y10 Z0\nG3 Y0 Z10 R10\n");
        assert!(errors.errors.is_empty());
        let ccw = sink.commands[1].arc.expect("arc geometry");
        assert!((ccw.center.y - 0.0).abs() < 1e-9);
        assert!((ccw.center.z - 0.0).abs() < 1e-9);
    }

    #[test]
    fn cw_arc_in_yz_plane_uses_the_same_cyclic_convention_as_xy() {
        let (sink, _, errors) = run("G19\nG0 Y10 Z0\nG2 Y0 Z10 R10\n");
        assert!(errors.errors.is_empty());
        let cw = sink.commands[1].arc.expect("arc geometry");
        assert!((cw.center.y - 10.0).abs() < 1e-9);
        assert!((cw.center.z - 10.0).abs() < 1e-9);
    }

    /// Like `run`, but lets a test preload the coordinate system offset
    /// table before parsing - standing in for a host that loads its
    /// work offsets from settings/EEPROM at startup.
    fn run_with_offsets(
        offsets: &[(CoordinateSystem, Position)],
        src: &str,
    ) -> (CollectingSink, CollectingCommands, CollectingErrors) {
        let mut interp = Interpreter::new(
            CollectingSink::default(),
            CollectingCommands::default(),
            CollectingErrors::default(),
        );
        for &(system, offset) in offsets {
            interp.state.set_coordinate_system_offset(system, offset);
        }
        interp.run(src);
        interp.into_sinks()
    }

    #[test]
    fn selecting_a_coordinate_system_applies_its_offset_to_absolute_moves() {
        let (sink, _, errors) = run_with_offsets(
            &[(
                CoordinateSystem::G55,
                Position {
                    x: 100.0,
                    y: 50.0,
                    z: 0.0,
                },
            )],
            "G55 G1 X10 Y0\n",
        );
        assert!(errors.errors.is_empty());
        assert_eq!(sink.commands[0].target.x, 110.0);
        assert_eq!(sink.commands[0].target.y, 50.0);
    }

    #[test]
    fn default_coordinate_system_is_g54_with_no_offset() {
        let (sink, _, errors) = run("G1 X10 Y0\n");
        assert!(errors.errors.is_empty());
        assert_eq!(sink.commands[0].target.x, 10.0);
    }

    #[test]
    fn g92_shifts_subsequent_absolute_moves() {
        // At machine position (10,0), G92 X0 declares "here is X=0" -
        // subsequent absolute moves are shifted by -10 to compensate.
        let (sink, _, errors) = run("G0 X10\nG92 X0\nG1 X5\n");
        assert!(errors.errors.is_empty());
        assert_eq!(sink.commands.len(), 2);
        assert_eq!(sink.commands[1].target.x, 15.0);
    }

    #[test]
    fn g92_does_not_itself_move_the_machine() {
        let (sink, _, errors) = run("G0 X10\nG92 X0\n");
        assert!(errors.errors.is_empty());
        // Only the G0 move produced a command - G92 must not be
        // misread as a bare axis-word move using the carried G0 mode.
        assert_eq!(sink.commands.len(), 1);
        assert_eq!(sink.commands[0].target.x, 10.0);
    }

    #[test]
    fn g92_only_shifts_the_axes_it_names() {
        let (sink, _, errors) = run("G0 X10 Y20\nG92 X0\nG1 X5 Y20\n");
        assert!(errors.errors.is_empty());
        // X was redefined (shift -10); Y was not (shift 0).
        assert_eq!(sink.commands[1].target.x, 15.0);
        assert_eq!(sink.commands[1].target.y, 20.0);
    }

    #[test]
    fn g92_1_cancels_the_offset() {
        let (sink, _, errors) = run("G0 X10\nG92 X0\nG92.1\nG1 X5\n");
        assert!(errors.errors.is_empty());
        assert_eq!(sink.commands.len(), 2);
        // With the G92 shift cancelled, X5 is just X5 again.
        assert_eq!(sink.commands[1].target.x, 5.0);
    }

    /// Like `run`, but lets a test configure `ModalState` (e.g. preload
    /// a G28/G30 reference position, standing in for host-provided
    /// settings) before parsing.
    fn run_configured(
        configure: impl FnOnce(&mut ModalState),
        src: &str,
    ) -> (CollectingSink, CollectingCommands, CollectingErrors) {
        let mut interp = Interpreter::new(
            CollectingSink::default(),
            CollectingCommands::default(),
            CollectingErrors::default(),
        );
        configure(&mut interp.state);
        interp.run(src);
        interp.into_sinks()
    }

    #[test]
    fn g28_with_axis_words_moves_through_intermediate_point_then_home() {
        let (sink, _, errors) = run_configured(
            |state| {
                state.g28_position = Position {
                    x: 0.0,
                    y: 0.0,
                    z: 50.0,
                }
            },
            "G0 X10 Y0\nG28 Z20\n",
        );
        assert!(errors.errors.is_empty());
        assert_eq!(sink.commands.len(), 3);
        // Intermediate leg: rapid to Z20, keeping X/Y from before.
        let intermediate = sink.commands[1];
        assert_eq!(intermediate.motion_mode, MotionMode::Rapid);
        assert_eq!(intermediate.target.x, 10.0);
        assert_eq!(intermediate.target.z, 20.0);
        // Final leg: rapid from the intermediate point to the stored
        // G28 reference position (raw machine coordinates).
        let home = sink.commands[2];
        assert_eq!(home.start, intermediate.target);
        assert_eq!(home.target.x, 0.0);
        assert_eq!(home.target.z, 50.0);
    }

    #[test]
    fn g28_with_no_axis_words_goes_directly_to_reference() {
        let (sink, _, errors) = run_configured(
            |state| {
                state.g28_position = Position {
                    x: 1.0,
                    y: 2.0,
                    z: 3.0,
                }
            },
            "G0 X10 Y0\nG28\n",
        );
        assert!(errors.errors.is_empty());
        // Only two commands: the initial G0, then straight to home - no
        // intermediate leg, since no axis words were given.
        assert_eq!(sink.commands.len(), 2);
        assert_eq!(
            sink.commands[1].target,
            Position {
                x: 1.0,
                y: 2.0,
                z: 3.0
            }
        );
    }

    #[test]
    fn g28_does_not_change_the_carried_motion_mode() {
        // G28/G30 are non-modal: a bare axis-word line afterward should
        // still use G1 (carried from before the G28), not Rapid.
        let (sink, _, errors) = run("G1 X10\nG28\nX20\n");
        assert!(errors.errors.is_empty());
        assert_eq!(sink.commands.len(), 3);
        assert_eq!(sink.commands[2].motion_mode, MotionMode::Linear);
    }

    #[test]
    fn g28_1_records_current_position_as_the_reference() {
        let (sink, _, errors) = run("G0 X7 Y8 Z9\nG28.1\nG28\n");
        assert!(errors.errors.is_empty());
        assert_eq!(sink.commands.len(), 2);
        assert_eq!(
            sink.commands[1].target,
            Position {
                x: 7.0,
                y: 8.0,
                z: 9.0
            }
        );
    }

    #[test]
    fn g30_uses_a_separate_reference_position_from_g28() {
        let (sink, _, errors) = run_configured(
            |state| {
                state.g28_position = Position {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                };
                state.g30_position = Position {
                    x: 2.0,
                    y: 0.0,
                    z: 0.0,
                };
            },
            "G30\n",
        );
        assert!(errors.errors.is_empty());
        assert_eq!(sink.commands[0].target.x, 2.0);
    }

    #[test]
    fn g53_ignores_work_offset_and_uses_machine_coordinates() {
        let (sink, _, errors) = run_with_offsets(
            &[(
                CoordinateSystem::G54,
                Position {
                    x: 100.0,
                    y: 0.0,
                    z: 0.0,
                },
            )],
            "G1 X10\nG53 G1 X10\n",
        );
        assert!(errors.errors.is_empty());
        // Without G53: X10 programmed + 100 work offset = 110.
        assert_eq!(sink.commands[0].target.x, 110.0);
        // With G53: X10 is read as the literal machine coordinate.
        assert_eq!(sink.commands[1].target.x, 10.0);
    }

    #[test]
    fn g53_ignores_incremental_distance_mode() {
        let (sink, _, errors) = run("G91 G1 X10\nG53 G1 X5\n");
        assert!(errors.errors.is_empty());
        // G53's X5 is an absolute machine coordinate even though G91
        // (incremental) is active.
        assert_eq!(sink.commands[1].target.x, 5.0);
    }

    #[test]
    fn g81_drills_a_hole_with_rapid_legs_and_default_g98_retract() {
        // Starting Z (0) is below R (2), so G98's default retracts to R
        // (the higher of the two) - see the dedicated G98/G99 tests
        // below for a case where they actually differ.
        let (sink, _, errors) = run("G81 X5 Y5 Z-10 R2 F100\n");
        assert!(errors.errors.is_empty());
        assert_eq!(sink.commands.len(), 4);
        assert_eq!(sink.commands[0].motion_mode, MotionMode::Rapid);
        assert_eq!(
            sink.commands[0].target,
            Position {
                x: 5.0,
                y: 5.0,
                z: 0.0
            }
        );
        assert_eq!(sink.commands[1].motion_mode, MotionMode::Rapid);
        assert_eq!(
            sink.commands[1].target,
            Position {
                x: 5.0,
                y: 5.0,
                z: 2.0
            }
        );
        assert_eq!(sink.commands[2].motion_mode, MotionMode::Linear);
        assert_eq!(
            sink.commands[2].target,
            Position {
                x: 5.0,
                y: 5.0,
                z: -10.0
            }
        );
        assert_eq!(sink.commands[3].motion_mode, MotionMode::Rapid);
        assert_eq!(
            sink.commands[3].target,
            Position {
                x: 5.0,
                y: 5.0,
                z: 2.0
            }
        );
    }

    #[test]
    fn g81_g98_retracts_to_initial_z_when_higher_than_r() {
        let (sink, _, errors) = run("G0 Z5\nG81 X0 Y0 Z-10 R2 F100\n");
        assert!(errors.errors.is_empty());
        // Initial Z (5) is higher than R (2), so G98 (the default)
        // retracts all the way back to the initial Z, not just to R.
        assert_eq!(sink.commands.last().unwrap().target.z, 5.0);
    }

    #[test]
    fn g99_always_retracts_to_r_even_when_lower_than_initial_z() {
        let (sink, _, errors) = run("G0 Z5\nG99 G81 X0 Y0 Z-10 R2 F100\n");
        assert!(errors.errors.is_empty());
        assert_eq!(sink.commands.last().unwrap().target.z, 2.0);
    }

    #[test]
    fn bare_axis_words_repeat_the_canned_cycle_with_sticky_z_and_r() {
        let (sink, _, errors) = run("G81 X0 Y0 Z-5 R2 F100\nX10 Y0\n");
        assert!(errors.errors.is_empty());
        assert_eq!(sink.commands.len(), 8);
        // Second hole's cut still goes to Z-5 - Z/R were not repeated
        // on the second line but are sticky.
        assert_eq!(
            sink.commands[6].target,
            Position {
                x: 10.0,
                y: 0.0,
                z: -5.0
            }
        );
    }

    #[test]
    fn g82_dwells_at_the_bottom_of_the_hole() {
        let (sink, commands, errors) = run("G82 X0 Y0 Z-5 R2 P0.5 F100\n");
        assert!(errors.errors.is_empty());
        assert_eq!(sink.commands.len(), 4);
        assert_eq!(
            commands.commands,
            std::vec![Command::Dwell { seconds: 0.5 }]
        );
    }

    #[test]
    fn g82_without_p_and_no_prior_value_is_a_missing_parameter_error() {
        let (sink, _, errors) = run("G82 X0 Y0 Z-5 R2 F100\n");
        assert!(sink.commands.is_empty());
        assert!(errors
            .errors
            .contains(&InterpretError::CannedCycleMissingParameter));
    }

    #[test]
    fn g83_pecks_in_q_sized_bites_with_full_retracts_between() {
        // R=2, target Z=-10: total depth 12, in bites of 3 - four
        // pecks (3,6,9,12=full depth), three intermediate retracts.
        let (sink, _, errors) = run("G83 X0 Y0 Z-10 R2 Q3 F100\n");
        assert!(errors.errors.is_empty());
        assert_eq!(sink.commands.len(), 10);
        assert_eq!(sink.commands[2].target.z, -1.0);
        assert_eq!(sink.commands[3].target.z, 2.0);
        assert_eq!(sink.commands[4].target.z, -4.0);
        assert_eq!(sink.commands[5].target.z, 2.0);
        assert_eq!(sink.commands[6].target.z, -7.0);
        assert_eq!(sink.commands[7].target.z, 2.0);
        // Final peck reaches the full programmed depth exactly.
        assert_eq!(sink.commands[8].target.z, -10.0);
        assert_eq!(sink.commands[8].motion_mode, MotionMode::Linear);
    }

    #[test]
    fn g83_without_a_positive_q_is_an_error_with_no_partial_output() {
        let (sink, _, errors) = run("G83 X0 Y0 Z-10 R2 F100\n");
        assert!(sink.commands.is_empty());
        assert!(errors
            .errors
            .contains(&InterpretError::InvalidPeckIncrement));
    }

    #[test]
    fn g85_retracts_at_feed_rate_instead_of_rapid() {
        let (sink, _, errors) = run("G85 X0 Y0 Z-5 R2 F100\n");
        assert!(errors.errors.is_empty());
        assert_eq!(
            sink.commands.last().unwrap().motion_mode,
            MotionMode::Linear
        );
    }

    #[test]
    fn g86_stops_the_spindle_at_the_bottom_and_rapid_retracts() {
        let (sink, commands, errors) = run("G86 X0 Y0 Z-5 R2 F100\n");
        assert!(errors.errors.is_empty());
        assert_eq!(sink.commands.last().unwrap().motion_mode, MotionMode::Rapid);
        assert_eq!(
            commands.commands,
            std::vec![Command::Spindle(SpindleCommand::Stop)]
        );
    }

    #[test]
    fn g89_dwells_then_retracts_at_feed_rate() {
        let (sink, commands, errors) = run("G89 X0 Y0 Z-5 R2 P0.25 F100\n");
        assert!(errors.errors.is_empty());
        assert_eq!(
            sink.commands.last().unwrap().motion_mode,
            MotionMode::Linear
        );
        assert_eq!(
            commands.commands,
            std::vec![Command::Dwell { seconds: 0.25 }]
        );
    }

    #[test]
    fn canned_cycle_without_z_or_r_is_a_missing_parameter_error() {
        let (sink, _, errors) = run("G81 X0 Y0 F100\n");
        assert!(sink.commands.is_empty());
        assert!(errors
            .errors
            .contains(&InterpretError::CannedCycleMissingParameter));
    }

    #[test]
    fn canned_cycle_in_a_non_xy_plane_is_unsupported() {
        let (sink, _, errors) = run("G18\nG81 X0 Y0 Z-5 R2 F100\n");
        assert!(sink.commands.is_empty());
        assert!(errors
            .errors
            .contains(&InterpretError::UnsupportedCannedCyclePlane));
    }

    #[test]
    fn g80_cancels_a_canned_cycle() {
        let (sink, _, errors) = run("G81 X0 Y0 Z-5 R2 F100\nG80\nX10\n");
        assert!(sink.commands.len() == 4);
        assert!(errors.errors.contains(&InterpretError::NoActiveMotionMode));
    }
}
