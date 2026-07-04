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
    scene: Scene,
    axis_vertex_count: u32,
    current_step: usize,
    camera: OrbitCamera,
    /// Which line the table was last force-scrolled to - so
    /// `scroll_to_row` only fires when the highlighted line actually
    /// changes, instead of fighting the user's manual scrolling every
    /// single frame.
    last_scrolled_line: Option<usize>,
}

impl ViewerApp {
    fn new(cc: &eframe::CreationContext<'_>, source: String) -> Self {
        let (entries, statuses, errors) = precompute(&source);
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
            scene,
            axis_vertex_count,
            current_step: 0,
            camera: OrbitCamera::default_isometric(),
            last_scrolled_line: None,
        }
    }

    fn step_forward(&mut self) {
        if self.current_step + 1 < self.entries.len() {
            self.current_step += 1;
        }
    }

    fn step_backward(&mut self) {
        self.current_step = self.current_step.saturating_sub(1);
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
        });

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

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                let (rect, response) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());

                if response.dragged() {
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
                let tool_position = (self.current_step < self.entries.len()).then_some([
                    status.state.position.x as f32,
                    status.state.position.y as f32,
                    status.state.position.z as f32,
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
                                        self.current_step = 0;
                                    }
                                    if ui.button("< Step").clicked() {
                                        self.step_backward();
                                    }
                                    if ui.button("Step >").clicked() {
                                        self.step_forward();
                                    }
                                    if ui.button("End >>").clicked() {
                                        self.current_step = self.entries.len().saturating_sub(1);
                                    }
                                    ui.separator();
                                    ui.label(format!(
                                        "Step {} / {}  (arrow keys also work)",
                                        self.current_step.saturating_add(1).min(self.entries.len()),
                                        self.entries.len()
                                    ));
                                    if let Some(entry) = self.entries.get(self.current_step) {
                                        ui.separator();
                                        ui.monospace(describe_output(&entry.output));
                                    }
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
