//! Trace tool: interpret a G-code file (or a small built-in demo) and
//! print a human-readable line-by-line trace of every resolved motion,
//! non-motion command, and error - handy for eyeballing what this
//! crate actually does with a piece of G-code while developing against
//! it or debugging unexpected output.
//!
//! Run with `cargo run --example trace -- path/to/file.gcode`, or with
//! no argument to run the built-in demo.
//!
//! `examples/gcode_samples/` has a set of real-world test files (see
//! `examples/gcode_samples/NOTICE.md` for provenance/license - they're
//! third-party GPL-3.0 test data, not part of this crate's own MIT
//! source) to try this against, e.g.:
//!
//! ```text
//! cargo run --example trace -- examples/gcode_samples/arc_rword_test.gcode
//! cargo run --example trace -- examples/gcode_samples/g17-g18-g19.gcode
//! cargo run --example trace -- examples/gcode_samples/ugs.gcode | less
//! ```
//!
//! This lives in `examples/` rather than the crate itself because it
//! needs `std` (`println!`, file I/O) - Cargo examples always link
//! `std` regardless of the library's `#![no_std]`, so this doesn't
//! affect the crate's no_std/no-alloc guarantee at all.

use std::{env, fs, process};

use swarf_gcode::{
    Command, CoolantCommand, ErrorSink, InterpretError, Interpreter, LineOutput, MotionMode,
    OutputSink, ProgramFlow, ResolvedMotionCommand, SpindleCommand,
};

struct PrintingSink;

impl OutputSink for PrintingSink {
    fn push(&mut self, output: LineOutput) -> Result<(), ()> {
        match output {
            LineOutput::Motion(m) => print_motion(m),
            LineOutput::Command(c) => print_command(c),
        }
        Ok(())
    }
}

fn print_motion(m: ResolvedMotionCommand) {
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
        MotionMode::None => "(no motion mode - should never be emitted)",
    };
    print!(
        "MOVE     {kind:<26} ({:>9.3}, {:>9.3}, {:>9.3}) -> ({:>9.3}, {:>9.3}, {:>9.3})",
        m.start.x, m.start.y, m.start.z, m.target.x, m.target.y, m.target.z,
    );
    if let Some(arc) = m.arc {
        print!(
            "  center=({:.3}, {:.3}, {:.3})",
            arc.center.x, arc.center.y, arc.center.z
        );
    }
    if m.motion_mode != MotionMode::Rapid {
        print!("  feed={:.1}mm/min", m.feed_rate);
    }
    println!();
}

fn print_command(c: Command) {
    let line = match c {
        Command::Spindle(SpindleCommand::Clockwise(rpm)) => {
            format!("SPINDLE  ON, clockwise, {rpm} RPM")
        }
        Command::Spindle(SpindleCommand::CounterClockwise(rpm)) => {
            format!("SPINDLE  ON, counter-clockwise, {rpm} RPM")
        }
        Command::Spindle(SpindleCommand::Stop) => "SPINDLE  OFF".to_string(),
        Command::Coolant(CoolantCommand::Mist) => "COOLANT  MIST on".to_string(),
        Command::Coolant(CoolantCommand::Flood) => "COOLANT  FLOOD on".to_string(),
        Command::Coolant(CoolantCommand::Off) => "COOLANT  off".to_string(),
        Command::ProgramFlow(ProgramFlow::Stop) => "PROGRAM  stop (M0)".to_string(),
        Command::ProgramFlow(ProgramFlow::OptionalStop) => {
            "PROGRAM  optional stop (M1)".to_string()
        }
        Command::ProgramFlow(ProgramFlow::End) => "PROGRAM  end (M2)".to_string(),
        Command::ProgramFlow(ProgramFlow::EndAndRewind) => {
            "PROGRAM  end + rewind (M30)".to_string()
        }
        Command::ToolChange { tool } => format!("TOOL     change -> T{tool}"),
        Command::Dwell { seconds } => format!("DWELL    {seconds}s"),
    };
    println!("{line}");
}

struct PrintingErrors;

impl ErrorSink for PrintingErrors {
    fn push(&mut self, error: InterpretError) {
        eprintln!("ERROR    {}", explain(error));
    }
}

fn explain(error: InterpretError) -> String {
    match error {
        InterpretError::NoActiveMotionMode => {
            "axis words given but no motion mode is active (no G0/G1/G2/G3 yet, or cancelled by G80)"
                .to_string()
        }
        InterpretError::ModalGroupConflict => {
            "two G/M words from the same NIST modal group appeared on one line".to_string()
        }
        InterpretError::NoToolSelected => {
            "M6 tool change with no prior T word ever selecting a tool".to_string()
        }
        InterpretError::InvalidArc(e) => format!("invalid arc geometry: {e:?}"),
        InterpretError::CannedCycleMissingParameter => {
            "canned cycle missing a required Z/R/P value (no sticky value set yet)".to_string()
        }
        InterpretError::InvalidPeckIncrement => {
            "G83 peck drilling needs a positive Q value".to_string()
        }
        InterpretError::UnsupportedCannedCyclePlane => {
            "canned cycle attempted outside the supported G17 (XY) plane".to_string()
        }
        InterpretError::UnsupportedSyntax => {
            "unsupported syntax (e.g. a #parameter reference)".to_string()
        }
        InterpretError::SyntaxError(span) => {
            format!("syntax error at byte {}, line {}", span.start, span.line)
        }
        InterpretError::OutputSinkFull => {
            "the output sink rejected a resolved command (sink full)".to_string()
        }
    }
}

const DEMO: &str = "\
G21 G90 G17
G0 X0 Y0 Z5
M3 S1200
G1 Z-2 F200
G1 X20 Y0
G2 X20 Y20 I0 J10
G1 X0 Y20
M5
G0 Z5
M2
";

fn main() {
    let src = match env::args().nth(1) {
        Some(path) => fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!("failed to read {path}: {e}");
            process::exit(1);
        }),
        None => {
            println!(
                "(no file given - running the built-in demo; pass a path to trace your own file)\n"
            );
            DEMO.to_string()
        }
    };

    let mut interp = Interpreter::new(PrintingSink, PrintingErrors);
    interp.run(&src);
}
