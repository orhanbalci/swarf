//! Step-through 3D toolpath viewer: an `egui`/`wgpu` app that
//! interprets a whole G-code program up front, then lets you step
//! forward and backward through the resolved output one item at a
//! time, watching the toolpath draw itself in a fixed isometric 3D
//! view while the corresponding source line is highlighted.
//!
//! Run with `cargo run --example viewer -- path/to/file.gcode`, or with
//! no argument for the built-in demo. Try it against the real-world
//! samples in `examples/gcode_samples/` (see that directory's
//! `NOTICE.md`), e.g. from the `swarf-gcode/` crate directory:
//!
//! ```text
//! cargo run --example viewer -- examples/gcode_samples/arc_rword_test.gcode
//! ```
//!
//! # Why interpretation is precomputed, not driven live from the UI
//!
//! `swarf-gcode`'s `Interpreter` has no undo - it's a one-way, forward
//! state machine (see its crate docs on real-time/Tier-A design). Going
//! "backward" here doesn't re-run the interpreter in reverse; instead
//! the whole program is interpreted ONCE at load time via repeated
//! `Interpreter::step` calls (one per source line - see `precompute`
//! below), recording every resolved `LineOutput` in order. The 3D
//! scene is built from that complete, static list (`geometry::Scene`).
//! "Stepping" the viewer is then just moving an index into
//! already-computed data - trivial to move forward OR backward, and
//! cheap enough to redraw from scratch every frame (immediate mode, no
//! diffing needed).
//!
//! # Why this needs its own dependencies
//!
//! `eframe`/`wgpu`/`bytemuck` are `[dev-dependencies]` on `swarf-gcode`
//! (see its `Cargo.toml`), not real dependencies - examples always link
//! `std` and can pull in whatever they want without affecting the
//! library's own `no_std`/no-alloc build.

mod geometry;
mod renderer;

use std::cell::Cell;
use std::rc::Rc;
use std::time::Instant;
use std::{env, fs};

use eframe::egui;
use egui_extras::{Column, TableBuilder};
use swarf_gcode::{
    Command, CoolantCommand, DistanceMode, ErrorSink, InterpretError, Interpreter, LineOutput,
    ModalState, MotionMode, OutputSink, Plane, ProgramFlow, ResolvedMotionCommand, SpindleCommand,
    Units,
};

use geometry::{Scene, TraceEntry};
use renderer::{view_projection, OrbitCamera, PathPaintCallback, PathRenderResources};

/// Sink that records every resolved output alongside whichever source
/// line was being fed to the interpreter when it was produced. The
/// current line is a shared `Cell` rather than a field written directly
/// from outside, because `Interpreter` owns its sink by value once
/// constructed - see `precompute`.
struct RecordingSink {
    current_line: Rc<Cell<usize>>,
    entries: Vec<TraceEntry>,
}

impl OutputSink for RecordingSink {
    fn push(&mut self, output: LineOutput) -> Result<(), ()> {
        self.entries.push(TraceEntry {
            line: self.current_line.get(),
            output,
        });
        Ok(())
    }
}

struct RecordingErrors {
    current_line: Rc<Cell<usize>>,
    errors: Vec<(usize, InterpretError)>,
}

impl ErrorSink for RecordingErrors {
    fn push(&mut self, error: InterpretError) {
        self.errors.push((self.current_line.get(), error));
    }
}

/// Whether the spindle is on, and if so which direction and RPM - the
/// RPM the *command* actually carried, not just whichever S word is
/// modally in effect (S is sticky and can change while the spindle sits
/// idle; the status panel should show what's really turning).
#[derive(Clone, Copy, PartialEq)]
enum SpindleStatus {
    Off,
    Clockwise(f64),
    CounterClockwise(f64),
}

#[derive(Clone, Copy, PartialEq)]
enum CoolantStatus {
    Off,
    Mist,
    Flood,
}

/// A snapshot of "what the controller would report right now" as of
/// one resolved output - `ModalState` alone doesn't carry on/off flags
/// for spindle/coolant (deliberately, see its module docs: those are
/// one-shot `Command`s, not persistent modal state), so this folds the
/// `Command` stream on top of a per-line `ModalState` snapshot to
/// reconstruct them.
#[derive(Clone, Copy)]
struct ControllerStatus {
    state: ModalState,
    spindle: SpindleStatus,
    coolant: CoolantStatus,
    loaded_tool: Option<u32>,
}

/// Interpret `source` one line at a time via `Interpreter::step`,
/// recording which source line produced each resolved output, a
/// per-entry `ControllerStatus` snapshot, and any errors.
fn precompute(
    source: &str,
) -> (
    Vec<TraceEntry>,
    Vec<ControllerStatus>,
    Vec<(usize, InterpretError)>,
) {
    let current_line = Rc::new(Cell::new(0));
    let sink = RecordingSink {
        current_line: Rc::clone(&current_line),
        entries: Vec::new(),
    };
    let errors = RecordingErrors {
        current_line: Rc::clone(&current_line),
        errors: Vec::new(),
    };

    let mut interp = Interpreter::new(sink, errors);
    let mut line_states = Vec::new();
    for (line_index, line) in source.lines().enumerate() {
        current_line.set(line_index);
        // `step` doesn't require a trailing newline, but `gcode`'s
        // parser treats it as a line terminator - appending one keeps
        // every call's input shaped exactly like a real serial line.
        interp.run(&format!("{line}\n"));
        line_states.push(interp.state);
    }

    let (sink, errors) = interp.into_sinks();
    let entries = sink.entries;

    let mut spindle = SpindleStatus::Off;
    let mut coolant = CoolantStatus::Off;
    let mut loaded_tool = None;
    let statuses = entries
        .iter()
        .map(|entry| {
            if let LineOutput::Command(cmd) = &entry.output {
                match cmd {
                    Command::Spindle(SpindleCommand::Clockwise(rpm)) => {
                        spindle = SpindleStatus::Clockwise(*rpm);
                    }
                    Command::Spindle(SpindleCommand::CounterClockwise(rpm)) => {
                        spindle = SpindleStatus::CounterClockwise(*rpm);
                    }
                    Command::Spindle(SpindleCommand::Stop) => spindle = SpindleStatus::Off,
                    Command::Coolant(CoolantCommand::Mist) => coolant = CoolantStatus::Mist,
                    Command::Coolant(CoolantCommand::Flood) => coolant = CoolantStatus::Flood,
                    Command::Coolant(CoolantCommand::Off) => coolant = CoolantStatus::Off,
                    Command::ToolChange { tool } => loaded_tool = Some(*tool),
                    Command::ProgramFlow(_) | Command::Dwell { .. } => {}
                }
            }
            ControllerStatus {
                state: line_states[entry.line],
                spindle,
                coolant,
                loaded_tool,
            }
        })
        .collect();

    (entries, statuses, errors.errors)
}

/// This viewer's assumed machine limits, for planning purposes only -
/// a real host would supply its own machine's actual settings. Shared
/// by `plan_motion` (deciding block speeds) and its arc-segment-count
/// bookkeeping (deciding how many `PlannedBlock`s one arc entry yields).
fn assumed_limits() -> swarf_motion::MachineLimits<3> {
    swarf_motion::MachineLimits {
        axes: [
            swarf_motion::AxisLimits { max_velocity: 3000.0, max_acceleration: 500.0 },
            swarf_motion::AxisLimits { max_velocity: 3000.0, max_acceleration: 500.0 },
            swarf_motion::AxisLimits { max_velocity: 600.0, max_acceleration: 100.0 },
        ],
        junction_deviation: 0.01,
        arc_tolerance: 0.002,
    }
}

fn to_motion_position(p: swarf_gcode::Position) -> swarf_motion::Position {
    swarf_motion::Position { x: p.x, y: p.y, z: p.z }
}

/// One entry's aggregated speed profile, as if its whole path (arc
/// segments included) were a single trapezoidal block: real entry speed
/// (respecting look-ahead cornering into this entry), real nominal
/// speed, real exit speed (respecting look-ahead cornering OUT of this
/// entry), and total distance. This is a deliberate simplification for
/// an arc - which actually tessellates into several individually-planned
/// sub-blocks with (usually near-identical, since a smooth arc has no
/// real internal corners) speeds - collapsed into one profile so the
/// rest of this file can reuse the exact same trapezoid math for both
/// straight moves and arcs.
#[derive(Clone, Copy)]
struct PlannedMotion {
    entry_speed: f64,
    nominal_speed: f64,
    exit_speed: f64,
    acceleration: f64,
    distance: f64,
}

impl PlannedMotion {
    fn duration_secs(&self) -> f64 {
        let (v0, vn, v1, a, d) = (self.entry_speed, self.nominal_speed, self.exit_speed, self.acceleration, self.distance);
        if d <= 0.0 || a <= 0.0 {
            return 0.0;
        }
        let accel_dist = (vn * vn - v0 * v0) / (2.0 * a);
        let decel_dist = (vn * vn - v1 * v1) / (2.0 * a);
        if accel_dist + decel_dist <= d {
            let cruise_dist = d - accel_dist - decel_dist;
            (vn - v0) / a + cruise_dist / vn + (vn - v1) / a
        } else {
            let peak_sqr = (2.0 * a * d + v0 * v0 + v1 * v1) / 2.0;
            let peak = peak_sqr.max(0.0).sqrt();
            (peak - v0).max(0.0) / a + (peak - v1).max(0.0) / a
        }
    }

    /// Fraction (0.0-1.0) of `distance` covered after `elapsed` seconds -
    /// the inverse of the timing `duration_secs` implies. Used to
    /// interpolate the tool marker's on-screen position at physically
    /// correct (accelerating/cruising/decelerating) speed, rather than
    /// naive constant-speed-in-time.
    fn distance_fraction_at(&self, elapsed: f64) -> f64 {
        let (v0, vn, v1, a, d) = (self.entry_speed, self.nominal_speed, self.exit_speed, self.acceleration, self.distance);
        if d <= 0.0 {
            return 1.0;
        }
        if a <= 0.0 {
            return (elapsed / self.duration_secs().max(1e-9)).clamp(0.0, 1.0);
        }

        let accel_dist = (vn * vn - v0 * v0) / (2.0 * a);
        let decel_dist = (vn * vn - v1 * v1) / (2.0 * a);

        let pos = if accel_dist + decel_dist <= d {
            let cruise_dist = d - accel_dist - decel_dist;
            let t_accel = (vn - v0) / a;
            let t_cruise = cruise_dist / vn;
            if elapsed < t_accel {
                v0 * elapsed + 0.5 * a * elapsed * elapsed
            } else if elapsed < t_accel + t_cruise {
                accel_dist + vn * (elapsed - t_accel)
            } else {
                let t_decel = (elapsed - t_accel - t_cruise).max(0.0);
                accel_dist + cruise_dist + (vn * t_decel - 0.5 * a * t_decel * t_decel)
            }
        } else {
            let peak_sqr = (2.0 * a * d + v0 * v0 + v1 * v1) / 2.0;
            let peak = peak_sqr.max(0.0).sqrt();
            let t_accel = (peak - v0).max(0.0) / a;
            let dist_to_peak = (peak * peak - v0 * v0) / (2.0 * a);
            if elapsed < t_accel {
                v0 * elapsed + 0.5 * a * elapsed * elapsed
            } else {
                let t_decel = (elapsed - t_accel).max(0.0);
                dist_to_peak + (peak * t_decel - 0.5 * a * t_decel * t_decel)
            }
        };
        (pos / d).clamp(0.0, 1.0)
    }
}

/// How many `PlannedBlock`s pushing `output` is expected to produce -
/// exactly 1 for a `Command` or a straight/rapid move, or however many
/// segments an arc tessellates into (computed independently here via
/// the same chord-tolerance formula `Planner::push_arc` uses
/// internally). Used to keep enough room in the planner queue before
/// each push (see `plan_motion`) and to slice the flat drained stream
/// back into per-entry groups.
fn expected_block_count(output: &LineOutput, arc_tolerance: f64) -> usize {
    match output {
        LineOutput::Command(_) => 1,
        LineOutput::Motion(m) => match m.arc {
            None => 1,
            Some(arc) => {
                let params = swarf_motion::arc::ArcParams::new(
                    to_motion_position(m.start),
                    to_motion_position(m.target),
                    to_motion_position(arc.center),
                    m.motion_mode == MotionMode::ArcClockwise,
                );
                swarf_motion::arc::segment_count(&params, arc_tolerance)
            }
        },
    }
}

/// Collapse one entry's group of drained `PlannedBlock`s (1 for a
/// straight move, many for a tessellated arc) into a single
/// `PlannedMotion` - see that type's doc comment for why this
/// aggregation is a deliberate simplification for arcs.
fn aggregate_planned(group: &[swarf_motion::PlannedBlock<Command>]) -> Option<PlannedMotion> {
    let mut entry_speed = None;
    let mut nominal_speed = 0.0;
    let mut exit_speed = 0.0;
    let mut acceleration = 0.0;
    let mut distance = 0.0;
    for block in group {
        if let swarf_motion::PlannedBlock::Motion {
            entry_speed: e,
            nominal_speed: n,
            exit_speed: x,
            acceleration: a,
            distance: d,
            ..
        } = block
        {
            if entry_speed.is_none() {
                entry_speed = Some(*e);
                acceleration = *a;
            }
            nominal_speed = *n;
            exit_speed = *x;
            distance += *d;
        }
    }
    entry_speed.map(|entry_speed| PlannedMotion {
        entry_speed,
        nominal_speed,
        exit_speed,
        acceleration,
        distance,
    })
}

/// Run every entry through a `swarf_bridge::GcodePlanner` (this
/// viewer's one and only integration point with the motion-planning
/// layer) and collapse the result into one `PlannedMotion` per `Motion`
/// entry (`None` for `Command` entries), aligned 1:1 with `entries`.
///
/// `CAPACITY` is chosen generously (comfortably larger than any single
/// arc's expected segment count at this viewer's tolerance, for any
/// realistic file) specifically so a push is never allowed to start
/// without enough room to complete - `Planner::push_arc` pushes an
/// arc's tessellated segments one at a time internally, so if it were
/// to run out of room PARTWAY through an arc, the segments already
/// pushed couldn't be un-pushed, and simply retrying the whole push
/// would double-count them. Ensuring room BEFORE each push (by draining
/// ready entries until there's enough space) sidesteps that entirely
/// rather than needing transactional/rollback support in `swarf-motion`
/// itself.
///
/// NOTE on the size: `BlockQueue`'s fixed-size array lives on the
/// stack, not the heap (that's the whole point of `swarf-motion` being
/// no-alloc) - discovered the hard way that a much larger CAPACITY here
/// (8192) overflowed a test thread's smaller default stack even though
/// it ran fine on this app's main thread, since debug builds don't
/// guarantee eliding the temporary stack copy `Planner::new()` returns
/// by value. 1024 stays comfortably clear of that while still holding
/// far more than any single arc in these sample files needs.
fn plan_motion(entries: &[TraceEntry]) -> Vec<Option<PlannedMotion>> {
    const CAPACITY: usize = 1024;
    let limits = assumed_limits();
    let arc_tolerance = limits.arc_tolerance;
    let mut planner: swarf_bridge::GcodePlanner<CAPACITY> = swarf_bridge::GcodePlanner::new(limits);

    let expected_counts: Vec<usize> = entries.iter().map(|e| expected_block_count(&e.output, arc_tolerance)).collect();

    // The actual number of blocks produced per entry - equal to
    // `expected_counts` except for the pathological "skipped" case
    // below, where it must become 0 so the final slicing pass doesn't
    // misalign against every entry that follows.
    let mut actual_counts = expected_counts.clone();

    let mut all_planned = Vec::with_capacity(entries.len());
    for (i, entry) in entries.iter().enumerate() {
        let needed = expected_counts[i];
        if needed > CAPACITY {
            // Pathological: a single arc alone needs more segments than
            // the entire buffer (an extremely large radius at a very
            // tight tolerance). Skip planning it rather than risk a
            // partial, uncorrectable push - falls back to naive timing
            // for this one entry.
            actual_counts[i] = 0;
            continue;
        }
        while CAPACITY - planner.len() < needed {
            match planner.pop_ready() {
                Some(p) => all_planned.push(p),
                None => break,
            }
        }
        let _ = planner.push(entry.output);
    }
    while let Some(p) = planner.pop_ready() {
        all_planned.push(p);
    }

    let mut result = Vec::with_capacity(entries.len());
    let mut cursor = 0;
    for (entry, &count) in entries.iter().zip(&actual_counts) {
        let end = (cursor + count).min(all_planned.len());
        let group = &all_planned[cursor..end];
        cursor = end;
        result.push(if matches!(entry.output, LineOutput::Motion(_)) {
            aggregate_planned(group)
        } else {
            None
        });
    }
    result
}

/// Tokenize `line` into `(letter, major, start, end)` words - the
/// upper-cased letter, the integer part of its number (ignoring any
/// `.minor` suffix - none of the patterns `highlight_span_for` looks
/// for need it), and its byte range - skipping `(...)` and
/// `;`-to-end-of-line comments.
fn scan_words(line: &str) -> Vec<(char, u32, usize, usize)> {
    let bytes = line.as_bytes();
    let mut words = Vec::new();
    let mut i = 0;
    let mut in_paren_comment = false;

    while i < bytes.len() {
        let c = bytes[i] as char;
        if in_paren_comment {
            if c == ')' {
                in_paren_comment = false;
            }
            i += 1;
            continue;
        }
        if c == '(' {
            in_paren_comment = true;
            i += 1;
            continue;
        }
        if c == ';' {
            break;
        }
        if c.is_ascii_alphabetic() {
            let letter = c.to_ascii_uppercase();
            let start = i;
            i += 1;
            while i < bytes.len() && (bytes[i] as char).is_whitespace() {
                i += 1;
            }
            if i < bytes.len() && matches!(bytes[i] as char, '+' | '-') {
                i += 1;
            }
            let digits_start = i;
            while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                i += 1;
            }
            let int_end = i;
            if i < bytes.len() && bytes[i] as char == '.' {
                i += 1;
                while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                    i += 1;
                }
            }
            if int_end > digits_start {
                if let Ok(major) = line[digits_start..int_end].parse::<u32>() {
                    words.push((letter, major, start, i));
                    continue;
                }
            }
        }
        i += 1;
    }

    words
}

fn find_word(line: &str, letter: char, majors: &[u32]) -> Option<(usize, usize)> {
    scan_words(line)
        .into_iter()
        .find(|&(l, major, _, _)| l == letter && majors.contains(&major))
        .map(|(_, _, s, e)| (s, e))
}

/// Find the byte range of the specific word in `line` responsible for
/// `output`, or `None` if there isn't one on this line (e.g. a bare
/// axis-word line continuing a motion mode set on an earlier line -
/// nothing on THIS line is "responsible", so nothing is highlighted).
///
/// Matching is done by TYPE (this output's specific meaning ->
/// searching for that specific letter+major-number word), not by
/// counting position. Position-based matching was tried first and had
/// two real bugs, both caught by running real files through the
/// viewer: (1) a line mixing modal-only words with productive ones
/// (`circle.gcode`'s first line, `"G17 G20 G90 G94 G54 M0 M5 M9"`,
/// where G17/G20/G90/G94/G54 never produce output) throws off any
/// simple "Nth word = Nth output" count; (2) even restricted to
/// "productive" words, `end_line`'s M-code resolution order (spindle,
/// then coolant, then program flow - see `visitor.rs`) is FIXED and
/// does not necessarily match the order those words appear in the
/// source text, so e.g. `"M9 M5"` still resolves spindle before
/// coolant. Matching by meaning sidesteps both: it doesn't care how
/// many other words are on the line or what order they're in.
fn highlight_span_for(line: &str, output: &LineOutput) -> Option<(usize, usize)> {
    match output {
        LineOutput::Motion(m) => {
            let majors: &[u32] = match m.motion_mode {
                MotionMode::Rapid => &[0],
                MotionMode::Linear => &[1],
                MotionMode::ArcClockwise => &[2],
                MotionMode::ArcCounterclockwise => &[3],
                MotionMode::Drill => &[81],
                MotionMode::DrillDwell => &[82],
                MotionMode::PeckDrill => &[83],
                MotionMode::BoreFeedOut => &[85],
                MotionMode::BoreSpindleStop => &[86],
                MotionMode::BoreDwellFeedOut => &[89],
                MotionMode::None => &[],
            };
            find_word(line, 'G', majors)
        }
        LineOutput::Command(c) => match c {
            Command::Spindle(SpindleCommand::Clockwise(_)) => find_word(line, 'M', &[3]),
            Command::Spindle(SpindleCommand::CounterClockwise(_)) => find_word(line, 'M', &[4]),
            Command::Spindle(SpindleCommand::Stop) => find_word(line, 'M', &[5]),
            Command::Coolant(CoolantCommand::Mist) => find_word(line, 'M', &[7]),
            Command::Coolant(CoolantCommand::Flood) => find_word(line, 'M', &[8]),
            Command::Coolant(CoolantCommand::Off) => find_word(line, 'M', &[9]),
            Command::ProgramFlow(ProgramFlow::Stop) => find_word(line, 'M', &[0]),
            Command::ProgramFlow(ProgramFlow::OptionalStop) => find_word(line, 'M', &[1]),
            Command::ProgramFlow(ProgramFlow::End) => find_word(line, 'M', &[2]),
            Command::ProgramFlow(ProgramFlow::EndAndRewind) => find_word(line, 'M', &[30]),
            Command::ToolChange { .. } => find_word(line, 'M', &[6]),
            Command::Dwell { .. } => find_word(line, 'G', &[4]),
        },
    }
}

fn describe_output(output: &LineOutput) -> String {
    match output {
        LineOutput::Motion(m) => describe_motion(m),
        LineOutput::Command(c) => describe_command(*c),
    }
}

fn describe_motion(m: &ResolvedMotionCommand) -> String {
    let kind = match m.motion_mode {
        MotionMode::Rapid => "G0 rapid",
        MotionMode::Linear => "G1 linear",
        MotionMode::ArcClockwise => "G2 arc CW",
        MotionMode::ArcCounterclockwise => "G3 arc CCW",
        MotionMode::Drill => "G81 drill",
        MotionMode::DrillDwell => "G82 drill+dwell",
        MotionMode::PeckDrill => "G83 peck drill",
        MotionMode::BoreFeedOut => "G85 bore, feed out",
        MotionMode::BoreSpindleStop => "G86 bore, spindle stop",
        MotionMode::BoreDwellFeedOut => "G89 bore, dwell+feed out",
        MotionMode::None => "(no motion mode)",
    };
    format!(
        "{kind}  ({:.3}, {:.3}, {:.3}) -> ({:.3}, {:.3}, {:.3})",
        m.start.x, m.start.y, m.start.z, m.target.x, m.target.y, m.target.z
    )
}

fn describe_command(c: Command) -> String {
    match c {
        Command::Spindle(SpindleCommand::Clockwise(rpm)) => format!("SPINDLE on, CW, {rpm} RPM"),
        Command::Spindle(SpindleCommand::CounterClockwise(rpm)) => {
            format!("SPINDLE on, CCW, {rpm} RPM")
        }
        Command::Spindle(SpindleCommand::Stop) => "SPINDLE off".to_string(),
        Command::Coolant(CoolantCommand::Mist) => "COOLANT mist on".to_string(),
        Command::Coolant(CoolantCommand::Flood) => "COOLANT flood on".to_string(),
        Command::Coolant(CoolantCommand::Off) => "COOLANT off".to_string(),
        Command::ProgramFlow(ProgramFlow::Stop) => "PROGRAM stop (M0)".to_string(),
        Command::ProgramFlow(ProgramFlow::OptionalStop) => "PROGRAM optional stop (M1)".to_string(),
        Command::ProgramFlow(ProgramFlow::End) => "PROGRAM end (M2)".to_string(),
        Command::ProgramFlow(ProgramFlow::EndAndRewind) => "PROGRAM end + rewind (M30)".to_string(),
        Command::ToolChange { tool } => format!("TOOL change -> T{tool}"),
        Command::Dwell { seconds } => format!("DWELL {seconds}s"),
    }
}

/// Simple on/off text badge: a colored dot plus a label, green-ish when
/// active and gray when idle. Kept deliberately plain (no gauges/lamps)
/// per feedback that a more elaborate "instrument cluster" look didn't
/// fit here and broke the layout.
fn status_badge(ui: &mut egui::Ui, on: bool, active_color: egui::Color32, text: &str) {
    ui.horizontal(|ui| {
        ui.colored_label(
            if on {
                active_color
            } else {
                egui::Color32::from_gray(120)
            },
            "\u{25cf}", // ●
        );
        ui.label(text);
    });
}

/// How to render a status row's value column - most rows are plain
/// text, but spindle/coolant get the on/off dot badge.
enum StatusValue {
    Badge {
        on: bool,
        color: egui::Color32,
        text: String,
    },
    Text(String),
    Monospace(String),
}

/// Status panel using the same `TableBuilder` component as the source
/// listing (striped rows, two columns) rather than a custom grid/canvas
/// widget - keeps this panel's layout behavior consistent with the rest
/// of the sidebar.
fn draw_status_panel(ui: &mut egui::Ui, status: &ControllerStatus) {
    let rows: Vec<(&str, StatusValue)> = vec![
        (
            "Spindle",
            match status.spindle {
                SpindleStatus::Off => StatusValue::Badge {
                    on: false,
                    color: egui::Color32::GREEN,
                    text: "off".to_string(),
                },
                SpindleStatus::Clockwise(rpm) => StatusValue::Badge {
                    on: true,
                    color: egui::Color32::from_rgb(90, 200, 110),
                    text: format!("CW, {rpm:.0} RPM"),
                },
                SpindleStatus::CounterClockwise(rpm) => StatusValue::Badge {
                    on: true,
                    color: egui::Color32::from_rgb(90, 200, 110),
                    text: format!("CCW, {rpm:.0} RPM"),
                },
            },
        ),
        (
            "Coolant",
            match status.coolant {
                CoolantStatus::Off => StatusValue::Badge {
                    on: false,
                    color: egui::Color32::from_rgb(90, 170, 230),
                    text: "off".to_string(),
                },
                CoolantStatus::Mist => StatusValue::Badge {
                    on: true,
                    color: egui::Color32::from_rgb(90, 170, 230),
                    text: "mist".to_string(),
                },
                CoolantStatus::Flood => StatusValue::Badge {
                    on: true,
                    color: egui::Color32::from_rgb(90, 170, 230),
                    text: "flood".to_string(),
                },
            },
        ),
        (
            "Position",
            StatusValue::Monospace(format!(
                "X{:.3}  Y{:.3}  Z{:.3}",
                status.state.position.x, status.state.position.y, status.state.position.z
            )),
        ),
        (
            "Feed rate",
            StatusValue::Text(format!("{:.1} mm/min", status.state.feed_rate)),
        ),
        (
            "Spindle S-word",
            StatusValue::Text(format!("{:.0} RPM", status.state.spindle_speed)),
        ),
        (
            "Units",
            StatusValue::Text(
                match status.state.units {
                    Units::Millimeters => "mm (G21)",
                    Units::Inches => "in (G20)",
                }
                .to_string(),
            ),
        ),
        (
            "Plane",
            StatusValue::Text(
                match status.state.plane {
                    Plane::Xy => "XY (G17)",
                    Plane::Zx => "ZX (G18)",
                    Plane::Yz => "YZ (G19)",
                }
                .to_string(),
            ),
        ),
        (
            "Distance mode",
            StatusValue::Text(
                match status.state.distance_mode {
                    DistanceMode::Absolute => "Absolute (G90)",
                    DistanceMode::Incremental => "Incremental (G91)",
                }
                .to_string(),
            ),
        ),
        (
            "Work offset",
            StatusValue::Text(format!("{:?}", status.state.coordinate_system)),
        ),
        (
            "Tool",
            StatusValue::Text(match (status.loaded_tool, status.state.selected_tool) {
                (Some(loaded), Some(sel)) if loaded == sel => format!("T{loaded}"),
                (Some(loaded), Some(sel)) => format!("T{loaded} (T{sel} selected next)"),
                (Some(loaded), None) => format!("T{loaded}"),
                (None, Some(sel)) => format!("none loaded (T{sel} selected)"),
                (None, None) => "none".to_string(),
            }),
        ),
    ];

    let row_height = ui.text_style_height(&egui::TextStyle::Body);
    TableBuilder::new(ui)
        .id_salt("status_table")
        .striped(true)
        .column(Column::exact(110.0))
        .column(Column::remainder())
        .min_scrolled_height(0.0)
        .body(|body| {
            body.rows(row_height, rows.len(), |mut row| {
                let (label, value) = &rows[row.index()];
                row.col(|ui| {
                    ui.label(*label);
                });
                row.col(|ui| match value {
                    StatusValue::Badge { on, color, text } => status_badge(ui, *on, *color, text),
                    StatusValue::Text(text) => {
                        ui.label(text);
                    }
                    StatusValue::Monospace(text) => {
                        ui.monospace(text);
                    }
                });
            });
        });
}

/// Build one source-line row's text, optionally highlighting a
/// sub-range (the currently-executing command word - see
/// `highlight_span_for`) and dimming non-current lines so the current
/// one stands out even in a long file.
fn line_layout_job(
    ui: &egui::Ui,
    line: &str,
    is_current: bool,
    highlight: Option<(usize, usize)>,
) -> egui::text::LayoutJob {
    let font_id = egui::TextStyle::Monospace.resolve(ui.style());
    let base_color = if is_current {
        egui::Color32::from_rgb(230, 230, 230)
    } else {
        ui.style().visuals.weak_text_color()
    };

    let mut job = egui::text::LayoutJob::default();
    let plain = egui::TextFormat {
        font_id: font_id.clone(),
        color: base_color,
        ..Default::default()
    };

    match highlight {
        Some((start, end)) if !line.is_empty() => {
            job.append(&line[..start], 0.0, plain.clone());
            job.append(
                &line[start..end],
                0.0,
                egui::TextFormat {
                    font_id: font_id.clone(),
                    color: egui::Color32::from_rgb(30, 25, 0),
                    background: egui::Color32::from_rgb(255, 216, 25),
                    ..Default::default()
                },
            );
            job.append(&line[end..], 0.0, plain);
        }
        _ => job.append(line, 0.0, plain),
    }

    job
}

const DEMO: &str = "\
G21 G90 G17
G0 X0 Y0 Z5
M3 S1200
G1 Z-2 F200
G1 X20 Y0
G2 X20 Y20 I0 J10
G1 X0 Y20
G1 X0 Y0
M5
G0 Z5
M2
";

struct ViewerApp {
    source_lines: Vec<String>,
    entries: Vec<TraceEntry>,
    statuses: Vec<ControllerStatus>,
    errors: Vec<(usize, InterpretError)>,
    /// Real speed profile for each `Motion` entry, from running the
    /// whole program through `swarf_bridge::GcodePlanner` once at load
    /// time (see `plan_motion`) - `None` for `Command` entries, or for
    /// a `Motion` entry the planner couldn't fit (see `plan_motion`'s
    /// doc comment). Animation timing and tool-position interpolation
    /// prefer this over the naive feed-rate-only fallback whenever it's
    /// available.
    planned: Vec<Option<PlannedMotion>>,
    scene: Scene,
    axis_vertex_count: u32,
    current_step: usize,
    /// How far through `current_step`'s motion the tool marker has
    /// travelled: 0.0 = `start`, 1.0 = `target`. Manual navigation
    /// (step/reset/end/arrow keys) always leaves this at 1.0 - "settled
    /// at this step's end", matching the viewer's original all-or-
    /// nothing behavior. Only playback (`advance_playback`) produces
    /// values in between, to move the marker smoothly along a move
    /// instead of only at its endpoints. Meaningless (ignored) for
    /// steps whose output isn't a `Motion`.
    progress: f32,
    playing: bool,
    /// Simulation speed as a fraction of real time: 1.0 plays at the
    /// program's actual feed rates, 0.5 half that, 2.0 twice, etc.
    speed: f32,
    /// Wall-clock time of the last `update` call, used to turn frame
    /// time into simulated seconds for `advance_playback`. `None` on
    /// the very first frame, so that frame contributes no elapsed time.
    last_tick: Option<Instant>,
    camera: OrbitCamera,
    /// Which line the table was last force-scrolled to - so
    /// `scroll_to_row` only fires when the highlighted line actually
    /// changes, instead of fighting the user's manual scrolling every
    /// single frame.
    last_scrolled_line: Option<usize>,
}

/// Assumed rapid traverse rate, in mm/min, purely for pacing the
/// play/pause animation - `ResolvedMotionCommand::feed_rate` is
/// meaningless for `Rapid` moves (see its doc comment), so there's no
/// programmed rate to time a rapid's on-screen travel against. Chosen
/// to look "fast" relative to typical feed rates, not to model any real
/// machine's actual rapid speed.
const ASSUMED_RAPID_RATE_MM_PER_MIN: f64 = 6000.0;

/// Floor under a feed move's effective rate for timing purposes, so a
/// line with no `F` word yet (feed rate 0.0, `ModalState`'s default)
/// doesn't produce an infinite/stalled animation duration.
const MIN_FEED_RATE_MM_PER_MIN: f64 = 10.0;

impl ViewerApp {
    fn new(cc: &eframe::CreationContext<'_>, source: String) -> Self {
        let (entries, statuses, errors) = precompute(&source);
        let planned = plan_motion(&entries);
        let scene = Scene::build(&entries);

        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("viewer requires the wgpu backend (see NativeOptions in main())");
        let axis_vertices = geometry::axis_vertices(scene.suggested_axis_length());
        let origin_vertices = geometry::sphere_vertices(scene.suggested_marker_radius());
        let tool_vertices = geometry::spindle_marker_vertices(scene.suggested_tool_marker_radius());
        let axis_vertex_count = axis_vertices.len() as u32;
        let resources = PathRenderResources::new(
            &render_state.device,
            render_state.target_format,
            &scene.vertices,
            &axis_vertices,
            &origin_vertices,
            &tool_vertices,
        );
        render_state
            .renderer
            .write()
            .callback_resources
            .insert(resources);

        Self {
            source_lines: source.lines().map(str::to_string).collect(),
            entries,
            statuses,
            errors,
            planned,
            scene,
            axis_vertex_count,
            current_step: 0,
            progress: 1.0,
            playing: false,
            speed: 1.0,
            last_tick: None,
            camera: OrbitCamera::default_isometric(),
            last_scrolled_line: None,
        }
    }

    /// How long (in simulated seconds) `step`'s motion should take to
    /// animate. Prefers the REAL trapezoidal duration from
    /// `swarf_bridge`/`swarf_motion`'s look-ahead planning
    /// (`self.planned`) - respecting actual acceleration limits and
    /// cornering into/out of this move - falling back to a naive
    /// distance-over-feed-rate estimate only when planning wasn't
    /// available for this entry (see `plan_motion`'s doc comment on
    /// when that happens). `0.0` for anything else (a non-`Motion`
    /// output, or a zero-length move), meaning `advance_playback` treats
    /// it as instantaneous and moves straight on to the next step
    /// without consuming any playback time.
    fn step_duration_secs(&self, step: usize) -> f32 {
        if let Some(Some(planned)) = self.planned.get(step) {
            return planned.duration_secs() as f32;
        }
        match self.entries.get(step).map(|e| &e.output) {
            Some(LineOutput::Motion(m)) => {
                let length = geometry::motion_length(m);
                if length <= 1e-9 {
                    return 0.0;
                }
                let rate = if m.motion_mode == MotionMode::Rapid {
                    ASSUMED_RAPID_RATE_MM_PER_MIN
                } else {
                    m.feed_rate.max(MIN_FEED_RATE_MM_PER_MIN)
                };
                ((length / rate) * 60.0) as f32
            }
            Some(LineOutput::Command(Command::Dwell { seconds })) => *seconds as f32,
            _ => 0.0,
        }
    }

    /// Manual navigation always leaves the marker "settled" at the
    /// resulting step's end - matching the viewer's original
    /// all-or-nothing behavior - and stops any in-progress playback, so
    /// the user's manual jump isn't immediately overridden by the next
    /// animated frame.
    fn goto_step(&mut self, step: usize) {
        self.playing = false;
        self.current_step = step.min(self.entries.len().saturating_sub(1));
        self.progress = 1.0;
    }

    fn step_forward(&mut self) {
        if self.current_step + 1 < self.entries.len() {
            self.goto_step(self.current_step + 1);
        } else {
            self.playing = false;
        }
    }

    fn step_backward(&mut self) {
        self.goto_step(self.current_step.saturating_sub(1));
    }

    /// Advance playback by `seconds` of simulated time (already scaled
    /// by `speed`), moving `progress` within `current_step` and rolling
    /// over into subsequent steps as needed. Steps with zero duration
    /// (non-`Motion` output, dwells aside, or a zero-length move) are
    /// crossed instantly, without consuming any of `seconds` - matching
    /// how a real controller doesn't pause between a spindle command
    /// and the move that follows it.
    fn advance_playback(&mut self, mut seconds: f32) {
        // Bounded rather than a bare `while` so a pathological run of
        // zero-duration steps (shouldn't happen - `step_duration_secs`
        // always returns >0 for a real move - but this is animation
        // code, not safety-critical) can't hang a frame.
        let max_iterations = self.entries.len() + 4;
        for _ in 0..max_iterations {
            if seconds <= 0.0 {
                return;
            }
            let duration = self.step_duration_secs(self.current_step);
            if duration <= f32::EPSILON {
                if self.current_step + 1 >= self.entries.len() {
                    self.playing = false;
                    self.progress = 1.0;
                    return;
                }
                self.current_step += 1;
                self.progress = 0.0;
                continue;
            }
            let time_left_in_step = (1.0 - self.progress) * duration;
            if seconds < time_left_in_step {
                self.progress += seconds / duration;
                return;
            }
            seconds -= time_left_in_step;
            if self.current_step + 1 >= self.entries.len() {
                self.playing = false;
                self.progress = 1.0;
                return;
            }
            self.current_step += 1;
            self.progress = 0.0;
        }
    }
}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.input(|i| {
            if i.key_pressed(egui::Key::ArrowRight) {
                self.step_forward();
            }
            if i.key_pressed(egui::Key::ArrowLeft) {
                self.step_backward();
            }
            if i.key_pressed(egui::Key::Space) {
                self.playing = !self.playing;
            }
        });

        // Turn wall-clock frame time into simulated seconds and advance
        // playback. `last_tick` is updated every frame regardless of
        // play state, so toggling play back on after sitting paused
        // doesn't replay a huge backlog of "elapsed" time in one jump.
        let now = Instant::now();
        let dt = self.last_tick.map_or(0.0, |t| now.duration_since(t).as_secs_f32());
        self.last_tick = Some(now);
        if self.playing {
            self.advance_playback(dt * self.speed);
            // Immediate mode only redraws on input by default - without
            // this, the animation would freeze between user input events.
            ctx.request_repaint();
        }

        if !self.errors.is_empty() {
            egui::TopBottomPanel::top("errors").show(ctx, |ui| {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 80, 80),
                    format!(
                        "{} error(s) during interpretation - see side panel",
                        self.errors.len()
                    ),
                );
            });
        }

        let current_line = self.entries.get(self.current_step).map(|e| e.line);

        // The specific word on `current_line` responsible for the
        // current step's output, if any - see `highlight_span_for`.
        let current_highlight = self
            .entries
            .get(self.current_step)
            .and_then(|entry| highlight_span_for(&self.source_lines[entry.line], &entry.output));

        let status = self
            .statuses
            .get(self.current_step)
            .copied()
            .unwrap_or(ControllerStatus {
                state: ModalState::default(),
                spindle: SpindleStatus::Off,
                coolant: CoolantStatus::Off,
                loaded_tool: None,
            });

        egui::SidePanel::left("source")
            .min_width(420.0)
            .show(ctx, |ui| {
                ui.heading("Controller Status");
                draw_status_panel(ui, &status);
                ui.add_space(4.0);
                ui.separator();

                ui.heading("Source");

                let row_height = ui.text_style_height(&egui::TextStyle::Monospace);

                // Only force-scroll when the highlighted line actually
                // changes - calling `scroll_to_row` every frame regardless
                // would fight any manual scrolling the user does to look
                // around the rest of the file.
                let mut table = TableBuilder::new(ui)
                    .id_salt("source_table")
                    .striped(true)
                    .column(Column::exact(36.0))
                    .column(Column::remainder())
                    .min_scrolled_height(0.0);
                if current_line.is_some() && current_line != self.last_scrolled_line {
                    table = table.scroll_to_row(current_line.unwrap(), Some(egui::Align::Center));
                    self.last_scrolled_line = current_line;
                }

                table.body(|body| {
                    body.rows(row_height, self.source_lines.len(), |mut row| {
                        let i = row.index();
                        let line = &self.source_lines[i];
                        let is_current = Some(i) == current_line;

                        row.col(|ui| {
                            ui.monospace((i + 1).to_string());
                        });
                        row.col(|ui| {
                            let job = line_layout_job(
                                ui,
                                line,
                                is_current,
                                is_current.then_some(current_highlight).flatten(),
                            );
                            ui.add(egui::Label::new(job));
                        });
                    });
                });

                if !self.errors.is_empty() {
                    ui.separator();
                    ui.heading("Errors");
                    egui::ScrollArea::vertical()
                        .max_height(160.0)
                        .show(ui, |ui| {
                            for (line, err) in &self.errors {
                                ui.colored_label(
                                    egui::Color32::from_rgb(220, 80, 80),
                                    format!("line {}: {err:?}", line + 1),
                                );
                            }
                        });
                }
            });

        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(if self.playing { "\u{25b6} Playing" } else { "\u{23f8} Paused" });
                ui.separator();
                ui.label(format!(
                    "Step {} / {}",
                    self.current_step.saturating_add(1).min(self.entries.len()),
                    self.entries.len()
                ));
                if let Some(entry) = self.entries.get(self.current_step) {
                    ui.separator();
                    ui.monospace(describe_output(&entry.output));
                }
                if !self.errors.is_empty() {
                    ui.separator();
                    ui.colored_label(
                        egui::Color32::from_rgb(220, 80, 80),
                        format!("{} error(s)", self.errors.len()),
                    );
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                let (rect, response) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());

                if response.dragged_by(egui::PointerButton::Secondary) {
                    let delta = response.drag_delta();
                    self.camera.pan(
                        [
                            self.scene.bounds_min.x as f32,
                            self.scene.bounds_min.y as f32,
                            self.scene.bounds_min.z as f32,
                        ],
                        [
                            self.scene.bounds_max.x as f32,
                            self.scene.bounds_max.y as f32,
                            self.scene.bounds_max.z as f32,
                        ],
                        rect.width() / rect.height().max(1.0),
                        [rect.width(), rect.height()],
                        [delta.x, delta.y],
                    );
                } else if response.dragged() {
                    const ROTATE_SPEED: f32 = 0.005;
                    let delta = response.drag_delta();
                    self.camera
                        .orbit(delta.x * ROTATE_SPEED, delta.y * ROTATE_SPEED);
                }
                if response.hovered() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll != 0.0 {
                        // Scrolling up (positive delta) zooms in, so the
                        // multiplier on `camera.zoom` (which scales the
                        // projection's half-extents) must shrink then.
                        self.camera.zoom_by((-scroll * 0.002).exp());
                    }
                }

                let aspect = rect.width() / rect.height().max(1.0);

                let mvp = view_projection(
                    [
                        self.scene.bounds_min.x as f32,
                        self.scene.bounds_min.y as f32,
                        self.scene.bounds_min.z as f32,
                    ],
                    [
                        self.scene.bounds_max.x as f32,
                        self.scene.bounds_max.y as f32,
                        self.scene.bounds_max.z as f32,
                    ],
                    aspect,
                    &self.camera,
                );

                let so_far_end = self
                    .scene
                    .step_ranges
                    .get(self.current_step)
                    .map_or(0, |r| r.end);
                let current_range = self
                    .scene
                    .step_ranges
                    .get(self.current_step)
                    .cloned()
                    .unwrap_or(0..0);
                let full_range = 0..self.scene.vertices.len() as u32;

                // Only show the tool-head marker once at least one step
                // has resolved - before that there's no meaningful "tool
                // tip position" yet, just the machine's power-on default.
                // For a `Motion` step, `progress` (1.0 unless playback is
                // mid-move) picks a point along it rather than only ever
                // showing the endpoint - `status.state.position` already
                // IS that endpoint (`ModalState` only moves on a
                // resolved `Motion`), so non-`Motion` steps (spindle/
                // coolant/etc., where `progress` is meaningless) fall
                // back to it unchanged.
                let animated_position = match self.entries.get(self.current_step).map(|e| &e.output) {
                    Some(LineOutput::Motion(m)) => {
                        // `progress` is a fraction of TIME through the
                        // step's duration - for a real (planned) move
                        // that's NOT the same as fraction of DISTANCE,
                        // since the trapezoid accelerates/cruises/
                        // decelerates rather than moving at constant
                        // speed. Convert through the real profile when
                        // available so the marker speeds up/slows down
                        // correctly instead of moving at naive constant
                        // velocity across the whole step.
                        let distance_t = match self.planned.get(self.current_step) {
                            Some(Some(planned)) => {
                                let elapsed = self.progress as f64 * planned.duration_secs();
                                planned.distance_fraction_at(elapsed)
                            }
                            _ => self.progress as f64,
                        };
                        geometry::motion_point_at(m, distance_t)
                    }
                    _ => status.state.position,
                };
                let tool_position = (self.current_step < self.entries.len()).then_some([
                    animated_position.x as f32,
                    animated_position.y as f32,
                    animated_position.z as f32,
                ]);

                ui.painter()
                    .add(eframe::egui_wgpu::Callback::new_paint_callback(
                        rect,
                        PathPaintCallback {
                            mvp,
                            viewport: [rect.width(), rect.height()],
                            full_range,
                            so_far_range: 0..so_far_end,
                            current_range,
                            axis_vertex_count: self.axis_vertex_count,
                            tool_position,
                        },
                    ));

                // A floating overlay drawn directly on the canvas rather
                // than in the top toolbar - `Area` renders in its own
                // layer above whatever `ui.painter()` drew into this
                // `Ui`, so it sits on top of the wgpu callback's output.
                egui::Area::new(egui::Id::new("camera_overlay"))
                    .fixed_pos(rect.left_top() + egui::vec2(8.0, 8.0))
                    .order(egui::Order::Foreground)
                    .show(ui.ctx(), |ui| {
                        egui::Frame::popup(ui.style())
                            .inner_margin(egui::Margin::same(4))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    if ui.button("Iso").clicked() {
                                        self.camera = OrbitCamera::default_isometric();
                                    }
                                    if ui.button("XY").clicked() {
                                        self.camera.view_xy();
                                    }
                                    if ui.button("XZ").clicked() {
                                        self.camera.view_xz();
                                    }
                                    if ui.button("YZ").clicked() {
                                        self.camera.view_yz();
                                    }
                                });
                            });
                    });

                egui::Area::new(egui::Id::new("program_controls_overlay"))
                    .fixed_pos(egui::pos2(rect.center().x, rect.bottom() - 8.0))
                    .pivot(egui::Align2::CENTER_BOTTOM)
                    .order(egui::Order::Foreground)
                    .show(ui.ctx(), |ui| {
                        egui::Frame::popup(ui.style())
                            .inner_margin(egui::Margin::same(4))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    if ui.button("<< Reset").clicked() {
                                        self.goto_step(0);
                                    }
                                    if ui.button("< Step").clicked() {
                                        self.step_backward();
                                    }
                                    if ui.button(if self.playing { "Pause" } else { "Play" })
                                        .clicked()
                                    {
                                        self.playing = !self.playing;
                                    }
                                    if ui.button("Step >").clicked() {
                                        self.step_forward();
                                    }
                                    if ui.button("End >>").clicked() {
                                        self.goto_step(self.entries.len().saturating_sub(1));
                                    }
                                    ui.separator();
                                    ui.label("Speed");
                                    ui.add(
                                        egui::Slider::new(&mut self.speed, 0.1..=4.0)
                                            .fixed_decimals(1)
                                            .suffix("x"),
                                    );
                                    ui.separator();
                                    ui.label("arrow keys/space also work");
                                });
                            });
                    });
            });
        });
    }
}

fn main() -> eframe::Result {
    let source = match env::args().nth(1) {
        Some(path) => fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!("failed to read {path}: {e}");
            std::process::exit(1);
        }),
        None => {
            eprintln!(
                "(no file given - running the built-in demo; pass a path to view your own file)"
            );
            DEMO.to_string()
        }
    };

    let native_options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "swarf-gcode viewer",
        native_options,
        Box::new(|cc| Ok(Box::new(ViewerApp::new(cc, source)))),
    )
}
