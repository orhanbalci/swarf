//! The interpreter itself, built directly against `gcode` 0.7's `core`
//! zero-allocation visitor API (`ProgramVisitor` / `BlockVisitor` /
//! `CommandVisitor`) - the architecture this crate originally targeted
//! before a toolchain gap (0.7 requires rust-version 1.85 / edition
//! 2024) forced a temporary fallback to 0.6's flat iterator API. That
//! gap is closed, so this module now matches `lib.rs`'s module docs
//! directly: three borrow levels, each holding a mutable borrow back
//! into the one real state owner (`ModalState`).
//!
//! Unlike the 0.6-based version this replaces, there is no parser bug
//! to work around here: `core::parse` gives us exactly one
//! `BlockVisitor::end_line` call per source line, bare axis words with
//! no G/M/T word on their own line arrive via `BlockVisitor::word_address`,
//! and each command's own arguments arrive scoped to that command via
//! its own `CommandVisitor`. We still pool all axis data (`word_address`
//! and every command's `argument`) into one shared per-line accumulator
//! on `BlockCtx`, because NIST semantics treat a line's axis words as
//! one pool that applies to whichever motion mode is active for that
//! line - not as data belonging to whichever G-word happens to precede
//! it syntactically (e.g. "G1 G91 X10": X is the line's target, not
//! specifically G91's argument).
//!
//! This module (and the crate as a whole) is allocation-free: `gcode`
//! 0.7's `core` module has no `alloc` dependency, and neither do we -
//! per-line state is a handful of `Option<f32>` fields on `BlockCtx`,
//! not a `Vec`.

use core::num::ParseIntError;

use gcode::core::{
    parse, BlockVisitor, CommandVisitor, ControlFlow, Diagnostics as CoreDiagnostics, Noop, Number,
    ProgramVisitor, Span, TokenType, Value,
};

use crate::modal_groups::{
    classify_general_code, classify_miscellaneous_code, ModalGroup, ModalGroupSet,
};
use crate::motion::{MotionMode, ResolvedMotionCommand};
use crate::state::{DistanceMode, ModalState, Plane, Position, Units};

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
/// (`ModalState`), a motion sink, and an error sink.
pub struct Interpreter<M: MotionSink, E: ErrorSink> {
    pub state: ModalState,
    sink: M,
    errors: E,
}

impl<M: MotionSink, E: ErrorSink> Interpreter<M, E> {
    pub fn new(sink: M, errors: E) -> Self {
        Self {
            state: ModalState::new(),
            sink,
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
    pub fn into_sinks(self) -> (M, E) {
        (self.sink, self.errors)
    }

    fn resolve_target_from_values(
        &self,
        x: Option<f32>,
        y: Option<f32>,
        z: Option<f32>,
    ) -> Position {
        let combine = |current: f64, offset: f64, word: Option<f32>| -> f64 {
            match (word, self.state.distance_mode) {
                (Some(v), DistanceMode::Absolute) => self.state.to_mm(v as f64) + offset,
                (Some(v), DistanceMode::Incremental) => current + self.state.to_mm(v as f64),
                (None, _) => current,
            }
        };

        Position {
            x: combine(self.state.position.x, self.state.work_offset.x, x),
            y: combine(self.state.position.y, self.state.work_offset.y, y),
            z: combine(self.state.position.z, self.state.work_offset.z, z),
        }
    }
}

impl<M: MotionSink, E: ErrorSink> CoreDiagnostics for Interpreter<M, E> {
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

impl<M: MotionSink, E: ErrorSink> ProgramVisitor for Interpreter<M, E> {
    fn start_block(&mut self) -> ControlFlow<impl BlockVisitor + '_> {
        ControlFlow::Continue(BlockCtx {
            interp: self,
            seen_groups: ModalGroupSet::new(),
            resolved_mode: None,
            x: None,
            y: None,
            z: None,
            f: None,
        })
    }
}

/// Per-line scratch: which modal groups were touched this line (for
/// conflict detection), which motion mode (if any) this line's G-word
/// resolved to, and the pooled axis/feed values seen anywhere on the
/// line - whether via a bare `word_address` or as an `argument` of some
/// command. Borrows the one real state owner, `Interpreter`.
struct BlockCtx<'a, M: MotionSink, E: ErrorSink> {
    interp: &'a mut Interpreter<M, E>,
    seen_groups: ModalGroupSet,
    resolved_mode: Option<MotionMode>,
    x: Option<f32>,
    y: Option<f32>,
    z: Option<f32>,
    f: Option<f32>,
}

impl<M: MotionSink, E: ErrorSink> BlockCtx<'_, M, E> {
    /// Record an X/Y/Z/F value into this line's shared pool, rejecting
    /// (with `UnsupportedSyntax`) anything that isn't a literal number -
    /// parameters/expressions are out of scope, see `InterpretError`.
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

impl<M: MotionSink, E: ErrorSink> CoreDiagnostics for BlockCtx<'_, M, E> {
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

impl<M: MotionSink, E: ErrorSink> BlockVisitor for BlockCtx<'_, M, E> {
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
        }
        ControlFlow::Continue(Noop)
    }

    fn start_tool_change_code(&mut self, _number: Number) -> ControlFlow<impl CommandVisitor + '_> {
        ControlFlow::Continue(Noop)
    }

    fn end_line(self, _span: Span) {
        let effective_mode = self.resolved_mode.unwrap_or(self.interp.state.motion_mode);
        let has_axis_word = self.x.is_some() || self.y.is_some() || self.z.is_some();

        if !has_axis_word {
            // No axis data on this line: either a bare modal-mode word
            // (e.g. a lone "G1") to remember for later, or a line with
            // nothing motion-relevant at all (comment/M-code/G17-only) -
            // either way, there is no move to resolve.
            if let Some(mode) = self.resolved_mode {
                self.interp.state.motion_mode = mode;
            }
            return;
        }

        if effective_mode == MotionMode::None {
            self.interp.errors.push(InterpretError::NoActiveMotionMode);
            return;
        }

        let start = self.interp.state.position;
        let target = self
            .interp
            .resolve_target_from_values(self.x, self.y, self.z);

        if let Some(f) = self.f {
            self.interp.state.feed_rate = self.interp.state.to_mm(f as f64);
        }

        let command = ResolvedMotionCommand {
            start,
            target,
            motion_mode: effective_mode,
            arc: None,
            feed_rate: self.interp.state.feed_rate,
        };

        self.interp.state.position = target;
        self.interp.state.motion_mode = effective_mode;

        let _ = self.interp.sink.push(command);
    }
}

/// Per-command scratch, sharing the SAME pending-axis accumulator as
/// its parent `BlockCtx` - a command's arguments (e.g. the `X10 Y20`
/// following a `G1`) are pooled into the line's axis data exactly like
/// a bare `word_address` would be, per NIST's line-is-the-unit
/// semantics (see this module's docs).
struct CommandCtx<'a, 'b, M: MotionSink, E: ErrorSink> {
    block: &'a mut BlockCtx<'b, M, E>,
}

impl<M: MotionSink, E: ErrorSink> CoreDiagnostics for CommandCtx<'_, '_, M, E> {
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

impl<M: MotionSink, E: ErrorSink> CommandVisitor for CommandCtx<'_, '_, M, E> {
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
    struct CollectingErrors {
        errors: std::vec::Vec<InterpretError>,
    }

    impl ErrorSink for CollectingErrors {
        fn push(&mut self, error: InterpretError) {
            self.errors.push(error);
        }
    }

    fn run(src: &str) -> (CollectingSink, CollectingErrors) {
        let mut interp = Interpreter::new(CollectingSink::default(), CollectingErrors::default());
        interp.run(src);
        interp.into_sinks()
    }

    #[test]
    fn simple_linear_move_resolves_to_one_command() {
        let (sink, _) = run("G1 X10 Y20\n");
        assert_eq!(sink.commands.len(), 1);
        let cmd = sink.commands[0];
        assert_eq!(cmd.motion_mode, MotionMode::Linear);
        assert_eq!(cmd.target.x, 10.0);
        assert_eq!(cmd.target.y, 20.0);
        assert_eq!(cmd.start, Position::default());
    }

    #[test]
    fn modal_carry_forward_reuses_previous_motion_mode() {
        let (sink, _) = run("G1 X10\nY20\n");
        assert_eq!(sink.commands.len(), 2);
        assert_eq!(sink.commands[1].motion_mode, MotionMode::Linear);
        assert_eq!(sink.commands[1].target.y, 20.0);
        assert_eq!(sink.commands[1].target.x, 10.0);
    }

    #[test]
    fn bare_axis_words_before_any_command_have_no_active_motion_mode() {
        let (sink, errors) = run("X10\n");
        assert_eq!(sink.commands.len(), 0);
        assert!(errors.errors.contains(&InterpretError::NoActiveMotionMode));
    }

    #[test]
    fn incremental_distance_mode_adds_to_current_position() {
        let (sink, _) = run("G1 G91 X10\nX10\n");
        assert_eq!(sink.commands.len(), 2);
        assert_eq!(sink.commands[0].target.x, 10.0);
        assert_eq!(sink.commands[1].target.x, 20.0);
    }

    #[test]
    fn conflicting_motion_words_on_one_line_is_detected() {
        let (_, errors) = run("G0 G1 X10\n");
        assert!(errors.errors.contains(&InterpretError::ModalGroupConflict));
    }

    #[test]
    fn plane_and_motion_words_coexist_without_conflict() {
        let (sink, errors) = run("G17 G1 X10\n");
        assert!(!errors.errors.contains(&InterpretError::ModalGroupConflict));
        assert_eq!(sink.commands.len(), 1);
    }

    #[test]
    fn units_conversion_applies_to_subsequent_literals() {
        let (sink, _) = run("G20 G1 X1\n");
        assert_eq!(sink.commands.len(), 1);
        assert!((sink.commands[0].target.x - 25.4).abs() < 1e-6);
    }

    #[test]
    fn rapid_and_linear_are_distinct_modes_in_output() {
        let (sink, _) = run("G0 X5\nG1 X10\n");
        assert_eq!(sink.commands.len(), 2);
        assert_eq!(sink.commands[0].motion_mode, MotionMode::Rapid);
        assert_eq!(sink.commands[1].motion_mode, MotionMode::Linear);
    }

    #[test]
    fn multiple_lines_accumulate_position_correctly() {
        let (sink, _) = run("G1 X10 Y0\nX20 Y10\nX0 Y0\n");
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
        let (sink, errors) = run("G1 X#1\n");
        assert!(sink.commands.is_empty());
        assert!(!errors.errors.is_empty());
        assert!(errors
            .errors
            .iter()
            .all(|e| matches!(e, InterpretError::SyntaxError(_))));
    }
}
