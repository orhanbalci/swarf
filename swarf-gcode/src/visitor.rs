//! The interpreter itself, built against `gcode` 0.6's REAL public API -
//! verified by reading its actual source after discovering that 0.7's
//! visitor hierarchy (which we designed against on paper, from its
//! docs) cannot compile on this environment's toolchain (0.7 requires
//! rust_version 1.85 / edition 2024; this container has rustc 1.75.0).
//! See Cargo.toml for the full explanation and the action item to port
//! this to 0.7 once a newer toolchain is available.
//!
//! # What changed from the 0.7-based design, and why this is still the
//! same interpreter
//!
//! 0.6 has NO three-level visitor hierarchy. Its model is much simpler:
//!   - `gcode::full_parse_with_callbacks(src, &mut callbacks)` returns an
//!     iterator of `Line`, each holding a slice of already-assembled
//!     `GCode` values (mnemonic + major/minor number + argument words).
//!   - Critically: **the parser itself already does modal carry-forward**.
//!     Reading its `parser.rs` directly (see `handle_arg`), a bare axis
//!     word with no preceding G/M/T word on its own line gets attached to
//!     a SYNTHESIZED `GCode` reusing `last_gcode_type` - a field the
//!     parser tracks internally, persisting across lines, set every time
//!     a real G/M/T word is seen. This means the "BlockCtx accumulates
//!     pending_axes from either word_address or argument, end_line
//!     resolves using carried-forward motion mode" logic we designed
//!     against 0.7 is **already done for us** by 0.6's parser.
//!   - The case of a bare axis word with NO command ever having been
//!     seen calls `Callbacks::argument_without_a_command` - exactly our
//!     `InterpretError::NoActiveMotionMode` case, now detected by the
//!     parser rather than by us.
//!
//! So this version of the interpreter is actually SIMPLER: for each
//! `Line`, for each `GCode` in it, classify by (mnemonic, major, minor),
//! apply simple modal codes immediately, and resolve motion codes using
//! that one `GCode`'s own already-merged argument list. There is no
//! separate per-command sub-visitor needed, because 0.6 hands us a
//! complete `GCode` (with all its arguments already attached) as one
//! unit, not as a stream of individual argument callbacks.
//!
//! The `ModalState`, `modal_groups`, and `motion` modules are UNCHANGED
//! by any of this - this confirms our layering decision was sound: the
//! parser-API boundary is fully absorbed by this one module.

use gcode::{Callbacks, GCode, Mnemonic, Span};

use crate::modal_groups::{
    classify_general_code, classify_miscellaneous_code, ModalGroup, ModalGroupSet,
};
use crate::motion::{MotionMode, ResolvedMotionCommand};
use crate::state::{DistanceMode, ModalState, Plane, Position, Units};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpretError {
    /// A bare axis-word line appeared with no command ever having been
    /// seen at all - 0.6's parser surfaces this via
    /// `Callbacks::argument_without_a_command` rather than us having to
    /// detect it ourselves.
    ArgumentWithoutCommand,
    /// A bare axis-word line appeared, a command WAS seen previously,
    /// but it resolved to `MotionMode::None` (e.g. after a G80) - this
    /// one we DO still have to detect ourselves, since 0.6's parser only
    /// knows about G/M/T numbers, not our semantic MotionMode mapping.
    NoActiveMotionMode,
    /// Two G or M words from the same NIST modal group appeared on one
    /// line (e.g. "G0 G1 X10").
    ModalGroupConflict,
    /// This scaffold does not parse parameter (`#`) references at all -
    /// 0.6 has no concept of them in its grammar; they would surface as
    /// unknown/garbage content via `Callbacks::unknown_content`, which
    /// we surface here under one unified variant for now.
    UnsupportedSyntax,
}

/// Sink for fully resolved motion commands - unchanged from the 0.7
/// design; this trait's shape has nothing to do with which gcode
/// version produced the input.
pub trait MotionSink {
    fn push(&mut self, command: ResolvedMotionCommand) -> Result<(), ()>;
}

/// Splits a single `GCode` (as produced by `gcode::parse`'s flat
/// iterator) into one or more `(GCode, true_line_number)` pairs, based
/// on each argument's own `span.line`. A `GCode` whose arguments all
/// share one line (or has no arguments at all) returns a single pair
/// using that line (or the `GCode`'s own span line, for an empty one -
/// e.g. a bare modal-mode-only command like a lone "G1"). A `GCode`
/// whose arguments straddle multiple lines (the carry-forward-merge
/// defect signature described in `Interpreter::run`'s docs) is split
/// into multiple synthetic `GCode`s, one per distinct line, each
/// carrying only the arguments that truly belong to it - all sharing
/// the original's mnemonic/number, since the carry-forward semantics
/// this works around ARE correct in spirit; we're only re-establishing
/// correct line granularity for our own per-line processing.
fn split_by_argument_line(code: &gcode::GCode) -> std::vec::Vec<(gcode::GCode, usize)> {
    let args = code.arguments();
    if args.is_empty() {
        return std::vec![(code.clone(), code.span().line)];
    }

    let first_line = args[0].span.line;
    let all_same_line = args.iter().all(|a| a.span.line == first_line);
    if all_same_line {
        return std::vec![(code.clone(), first_line)];
    }

    let mut groups: std::vec::Vec<(gcode::GCode, usize)> = std::vec::Vec::new();
    let mut current_line = args[0].span.line;
    let mut current = gcode::GCode::new(code.mnemonic(), gcode_number(code), code.span());

    for arg in args {
        if arg.span.line != current_line {
            groups.push((current, current_line));
            current_line = arg.span.line;
            current = gcode::GCode::new(code.mnemonic(), gcode_number(code), code.span());
        }
        current = current.with_argument(*arg);
    }
    groups.push((current, current_line));
    groups
}

/// Reconstruct the `f32` number `GCode::new` expects from a `GCode`'s
/// already-split major/minor parts - `GCode` does not expose its raw
/// `number` field directly, only the derived major/minor accessors.
fn gcode_number(code: &gcode::GCode) -> f32 {
    code.major_number() as f32 + (code.minor_number() as f32) / 10.0
}

/// The interpreter. Owns the one persistent piece of state
/// (`ModalState`), a motion sink, and an error log accumulated across
/// the whole run.
pub struct Interpreter<S: MotionSink> {
    pub state: ModalState,
    sink: S,
    errors: std::vec::Vec<InterpretError>,
}

impl<S: MotionSink> Interpreter<S> {
    pub fn new(sink: S) -> Self {
        Self {
            state: ModalState::new(),
            sink,
            errors: std::vec::Vec::new(),
        }
    }

    /// Parse and interpret an entire source string.
    ///
    /// # A confirmed defect in `gcode` 0.6.1 this method works around
    ///
    /// While building this against the real crate (not just its docs),
    /// direct testing surfaced a real bug: `gcode::parse()` /
    /// `full_parse_with_callbacks()` merge a bare carry-forward word on
    /// a SUBSEQUENT physical line into the SAME `GCode` as the
    /// preceding command (e.g. "G1 X10\nY20\n" produces one `GCode`
    /// with major=1 whose arguments are X (line 0) and Y (line 1) - NOT
    /// two separate logical lines, contrary to NIST modal semantics
    /// where each line is independently a "did this line have an
    /// explicit command or not" question). Worse: once that merge has
    /// happened, the NEXT command-with-its-own-argument on a new line
    /// (e.g. a following "G0 X5") gets split into a spurious EMPTY
    /// GCode followed by a DUPLICATE GCode that actually holds the
    /// argument. Reproduced minimally and confirmed NOT to occur for
    /// "G1 X10\nY20\nM3\n" or "G1 X10\nY20\nG0\n" (no trailing argument
    /// on the new command) - the defect requires all three conditions:
    /// a carry-forward merge upstream, a new command, AND that new
    /// command having its own argument on the same line.
    ///
    /// Rather than trust this crate's line/command grouping at all, we:
    ///   1. Pull the flat `GCode` iterator (bypassing `Line` entirely).
    ///   2. Re-derive true line boundaries ourselves from each
    ///      argument's own `span.line` - confirmed reliable even when
    ///      the crate's own grouping isn't.
    ///   3. Split any `GCode` whose arguments straddle more than one
    ///      physical line back into per-line argument groups.
    ///   4. Drop any `GCode` with NO mnemonic-bearing number AND no
    ///      arguments at all - the empty-GCode half of the duplication
    ///      defect (a real command always either carries args or
    ///      stands alone meaningfully; an empty one immediately
    ///      followed by a duplicate is the defect's signature).
    ///
    /// This is defensive code working around a third-party bug, not a
    /// design choice - flagged clearly so it can be deleted the moment
    /// either (a) this crate moves to `gcode` 0.7's visitor API on a
    /// newer toolchain, where this whole grouping problem doesn't arise
    /// the same way, or (b) the upstream defect is fixed and we bump
    /// the dependency version.
    pub fn run(&mut self, src: &str) {
        // Pass 1: detect bare-argument-with-no-command via gcode's own
        // Callbacks mechanism - see this method's doc comment for why
        // gcode::parse()'s flat iterator (used below) can't tell us
        // this on its own (it uses Nop callbacks internally).
        {
            let errors = std::rc::Rc::new(std::cell::RefCell::new(std::vec::Vec::new()));
            struct Adapter(std::rc::Rc<std::cell::RefCell<std::vec::Vec<InterpretError>>>);
            impl Callbacks for Adapter {
                fn argument_without_a_command(&mut self, _l: char, _v: f32, _s: Span) {
                    self.0
                        .borrow_mut()
                        .push(InterpretError::ArgumentWithoutCommand);
                }
            }
            let adapter = Adapter(errors.clone());
            for _line in gcode::full_parse_with_callbacks(src, adapter) {}
            self.errors.extend(errors.borrow_mut().drain(..));
        }

        // Pass 2: re-derive correct (GCode, true_line_number) pairs from
        // gcode::parse()'s flat iterator. "True line number" comes from
        // the FIRST argument's span if present, falling back to the
        // GCode's own span otherwise - see split_by_argument_line for
        // how a single GCode whose arguments straddle multiple physical
        // lines (the confirmed carry-forward-merge defect) gets broken
        // back apart into multiple (GCode, line) pairs.
        let codes: std::vec::Vec<_> = gcode::parse(src).collect();
        let mut numbered: std::vec::Vec<(gcode::GCode, usize)> = std::vec::Vec::new();
        for code in codes {
            for (split_code, line_no) in split_by_argument_line(&code) {
                numbered.push((split_code, line_no));
            }
        }

        // Pass 3: collapse the empty-GCode-then-duplicate defect
        // pattern - ONLY when both entries report the SAME true line
        // number AND the same mnemonic/major/minor, which is the exact,
        // narrow signature reproduced and documented above. This must
        // run on the (GCode, line) pairs, not on a naive "consecutive
        // top-level GCode" view - a legitimate multi-command line like
        // "G0 G1 X10" produces an empty G0 followed by a non-empty G1,
        // which must NOT be collapsed (different major numbers, and -
        // just as importantly - this is normal G-code, not a defect).
        let mut cleaned: std::vec::Vec<(gcode::GCode, usize)> = std::vec::Vec::new();
        let mut i = 0;
        while i < numbered.len() {
            let (ref code, line) = numbered[i];
            let is_empty_defect = code.arguments().is_empty()
                && i + 1 < numbered.len()
                && numbered[i + 1].1 == line
                && numbered[i + 1].0.mnemonic() == code.mnemonic()
                && numbered[i + 1].0.major_number() == code.major_number()
                && numbered[i + 1].0.minor_number() == code.minor_number()
                && !numbered[i + 1].0.arguments().is_empty();
            if is_empty_defect {
                i += 1;
                continue;
            }
            cleaned.push(numbered[i].clone());
            i += 1;
        }

        // Pass 4: group by true line number and interpret each group.
        let mut current_line: Option<usize> = None;
        let mut group: std::vec::Vec<gcode::GCode> = std::vec::Vec::new();
        for (code, line) in cleaned {
            if current_line.is_some() && current_line != Some(line) {
                self.interpret_line(&group);
                group.clear();
            }
            current_line = Some(line);
            group.push(code);
        }
        if !group.is_empty() {
            self.interpret_line(&group);
        }
    }

    pub fn take_errors(&mut self) -> std::vec::Vec<InterpretError> {
        core::mem::take(&mut self.errors)
    }

    fn interpret_line(&mut self, gcodes: &[GCode]) {
        let mut seen_groups = ModalGroupSet::new();
        let mut resolved_mode: Option<MotionMode> = None;
        let mut motion_gcode: Option<&GCode> = None;

        for code in gcodes {
            let major = code.major_number();
            let minor_raw = code.minor_number();
            let minor = if minor_raw == 0 { None } else { Some(minor_raw) };

            match code.mnemonic() {
                Mnemonic::General => match classify_general_code(major, minor) {
                    Some(ModalGroup::Motion) => {
                        if seen_groups.contains(ModalGroup::Motion) {
                            self.errors.push(InterpretError::ModalGroupConflict);
                        }
                        seen_groups.insert(ModalGroup::Motion);

                        let mode = match (major, minor) {
                            (0, None) => MotionMode::Rapid,
                            (1, None) => MotionMode::Linear,
                            (2, None) => MotionMode::ArcClockwise,
                            (3, None) => MotionMode::ArcCounterclockwise,
                            (80, None) => MotionMode::None,
                            _ => MotionMode::None,
                        };
                        resolved_mode = Some(mode);
                        motion_gcode = Some(code);
                    }
                    Some(ModalGroup::Plane) => {
                        if seen_groups.contains(ModalGroup::Plane) {
                            self.errors.push(InterpretError::ModalGroupConflict);
                        }
                        seen_groups.insert(ModalGroup::Plane);
                        self.state.plane = match major {
                            17 => Plane::Xy,
                            18 => Plane::Zx,
                            19 => Plane::Yz,
                            _ => self.state.plane,
                        };
                    }
                    Some(ModalGroup::Units) => {
                        if seen_groups.contains(ModalGroup::Units) {
                            self.errors.push(InterpretError::ModalGroupConflict);
                        }
                        seen_groups.insert(ModalGroup::Units);
                        self.state.units = match major {
                            20 => Units::Inches,
                            21 => Units::Millimeters,
                            _ => self.state.units,
                        };
                    }
                    Some(ModalGroup::DistanceMode) => {
                        if seen_groups.contains(ModalGroup::DistanceMode) {
                            self.errors.push(InterpretError::ModalGroupConflict);
                        }
                        seen_groups.insert(ModalGroup::DistanceMode);
                        self.state.distance_mode = match major {
                            90 => DistanceMode::Absolute,
                            91 => DistanceMode::Incremental,
                            _ => self.state.distance_mode,
                        };
                    }
                    Some(group) => {
                        if seen_groups.contains(group) {
                            self.errors.push(InterpretError::ModalGroupConflict);
                        }
                        seen_groups.insert(group);
                    }
                    None => {}
                },
                Mnemonic::Miscellaneous => {
                    if let Some(group) = classify_miscellaneous_code(major, minor) {
                        if seen_groups.contains(group) {
                            self.errors.push(InterpretError::ModalGroupConflict);
                        }
                        seen_groups.insert(group);
                    }
                }
                Mnemonic::ProgramNumber | Mnemonic::ToolChange => {}
            }
        }

        let effective_mode = resolved_mode.unwrap_or(self.state.motion_mode);

        // Axis words (X/Y/Z/F) can attach to ANY G-word on the line, not
        // only the one classified as Motion - e.g. in "G1 G91 X10", the
        // parser attaches X10 to G91 (the nearest preceding command),
        // not to G1. Real G-code semantics treat the whole line's axis
        // data as one pool that applies to the line's resolved motion
        // mode, regardless of which specific word it happened to
        // attach to syntactically. Verified directly against gcode
        // 0.6.1's actual output for this exact case before fixing this.
        let find_value = |letter: char| -> Option<f32> {
            gcodes.iter().find_map(|c| c.value_for(letter))
        };

        let has_axis_word =
            find_value('X').is_some() || find_value('Y').is_some() || find_value('Z').is_some();

        if motion_gcode.is_none() && !has_axis_word {
            // Nothing motion-relevant on this line at all (comment-only,
            // M-code-only, or a non-motion G-word with no axis data).
            return;
        }

        if !has_axis_word {
            // A motion-mode word with no axis data (e.g. a bare "G1") -
            // just update modal state for future lines.
            self.state.motion_mode = effective_mode;
            return;
        }

        if effective_mode == MotionMode::None {
            self.errors.push(InterpretError::NoActiveMotionMode);
            return;
        }

        let start = self.state.position;
        let target = self.resolve_target_from_values(find_value('X'), find_value('Y'), find_value('Z'));

        if let Some(f) = find_value('F') {
            self.state.feed_rate = self.state.to_mm(f as f64);
        }

        let command = ResolvedMotionCommand {
            start,
            target,
            motion_mode: effective_mode,
            arc: None,
            feed_rate: self.state.feed_rate,
        };

        self.state.position = target;
        self.state.motion_mode = effective_mode;

        let _ = self.sink.push(command);
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

    fn run(src: &str) -> Interpreter<CollectingSink> {
        let mut interp = Interpreter::new(CollectingSink::default());
        interp.run(src);
        interp
    }

    #[test]
    fn simple_linear_move_resolves_to_one_command() {
        let interp = run("G1 X10 Y20\n");
        assert_eq!(interp.sink.commands.len(), 1);
        let cmd = interp.sink.commands[0];
        assert_eq!(cmd.motion_mode, MotionMode::Linear);
        assert_eq!(cmd.target.x, 10.0);
        assert_eq!(cmd.target.y, 20.0);
        assert_eq!(cmd.start, Position::default());
    }

    #[test]
    fn modal_carry_forward_reuses_previous_motion_mode() {
        let interp = run("G1 X10\nY20\n");
        assert_eq!(interp.sink.commands.len(), 2);
        assert_eq!(interp.sink.commands[1].motion_mode, MotionMode::Linear);
        assert_eq!(interp.sink.commands[1].target.y, 20.0);
        assert_eq!(interp.sink.commands[1].target.x, 10.0);
    }

    #[test]
    fn bare_axis_words_before_any_command_reported_by_parser_itself() {
        let mut interp = run("X10\n");
        assert_eq!(interp.sink.commands.len(), 0);
        let errors = interp.take_errors();
        assert!(errors.contains(&InterpretError::ArgumentWithoutCommand));
    }

    #[test]
    fn incremental_distance_mode_adds_to_current_position() {
        let interp = run("G1 G91 X10\nX10\n");
        assert_eq!(interp.sink.commands.len(), 2);
        assert_eq!(interp.sink.commands[0].target.x, 10.0);
        assert_eq!(interp.sink.commands[1].target.x, 20.0);
    }

    #[test]
    fn conflicting_motion_words_on_one_line_is_detected() {
        let mut interp = run("G0 G1 X10\n");
        let errors = interp.take_errors();
        assert!(errors.contains(&InterpretError::ModalGroupConflict));
    }

    #[test]
    fn plane_and_motion_words_coexist_without_conflict() {
        let mut interp = run("G17 G1 X10\n");
        let errors = interp.take_errors();
        assert!(!errors.contains(&InterpretError::ModalGroupConflict));
        assert_eq!(interp.sink.commands.len(), 1);
    }

    #[test]
    fn units_conversion_applies_to_subsequent_literals() {
        let interp = run("G20 G1 X1\n");
        assert_eq!(interp.sink.commands.len(), 1);
        assert!((interp.sink.commands[0].target.x - 25.4).abs() < 1e-6);
    }

    #[test]
    fn rapid_and_linear_are_distinct_modes_in_output() {
        let interp = run("G0 X5\nG1 X10\n");
        assert_eq!(interp.sink.commands.len(), 2);
        assert_eq!(interp.sink.commands[0].motion_mode, MotionMode::Rapid);
        assert_eq!(interp.sink.commands[1].motion_mode, MotionMode::Linear);
    }

    #[test]
    fn multiple_lines_accumulate_position_correctly() {
        let interp = run("G1 X10 Y0\nX20 Y10\nX0 Y0\n");
        assert_eq!(interp.sink.commands.len(), 3);
        assert_eq!(
            interp.sink.commands[0].target,
            Position { x: 10.0, y: 0.0, z: 0.0 }
        );
        assert_eq!(
            interp.sink.commands[1].target,
            Position { x: 20.0, y: 10.0, z: 0.0 }
        );
        assert_eq!(
            interp.sink.commands[2].target,
            Position { x: 0.0, y: 0.0, z: 0.0 }
        );
    }
}
