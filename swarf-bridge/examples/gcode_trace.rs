//! Trace tool: interpret a G-code file (or a small built-in demo)
//! through `swarf_gcode::Interpreter` piped, via this crate's
//! `GcodePlanner`, into a `swarf_motion::Planner`, then print each
//! finalized block's entry/nominal/exit speed and computed duration -
//! handy for eyeballing that look-ahead is actually working (a block
//! approaching a sharp corner should visibly show a lower exit speed
//! than its own nominal speed).
//!
//! For tracing the bare planner with NO G-code involved at all, see
//! `swarf-motion/examples/planner_trace.rs` instead.
//!
//! Run with `cargo run --example gcode_trace -- path/to/file.gcode`, or
//! with no argument for the built-in demo. Try it against the real
//! files in `swarf-gcode/examples/gcode_samples/`:
//!
//! ```text
//! # from the swarf-bridge/ crate directory:
//! cargo run --example gcode_trace -- ../swarf-gcode/examples/gcode_samples/circle.gcode
//! cargo run --example gcode_trace -- ../swarf-gcode/examples/gcode_samples/rust_logo.gcode
//! ```

use std::{env, fs, process};

use swarf_bridge::GcodePlanner;
use swarf_gcode::{ErrorSink, Interpreter, InterpretError};
use swarf_motion::{AxisLimits, MachineLimits, PlannedBlock};

struct PrintingErrors;

impl ErrorSink for PrintingErrors {
    fn push(&mut self, error: InterpretError) {
        eprintln!("ERROR    {error:?}");
    }
}

/// Trapezoid (or triangle, if the block is too short to reach nominal
/// speed) move duration from entry/nominal/exit speed, distance, and
/// acceleration - kept local to this example rather than part of any
/// crate's own public API: an executor stage would need this kind of
/// timing, but the planner's own job stops at handing over the
/// entry/nominal/exit speeds themselves.
fn duration_secs(entry: f64, nominal: f64, exit: f64, accel: f64, distance: f64) -> f64 {
    if distance <= 0.0 || accel <= 0.0 {
        return 0.0;
    }
    let accel_dist = (nominal * nominal - entry * entry) / (2.0 * accel);
    let decel_dist = (nominal * nominal - exit * exit) / (2.0 * accel);

    if accel_dist + decel_dist <= distance {
        let cruise_dist = distance - accel_dist - decel_dist;
        (nominal - entry) / accel + cruise_dist / nominal + (nominal - exit) / accel
    } else {
        // Triangle profile: never reaches `nominal`, solve for the
        // actual peak speed reached partway through the block.
        let peak_sqr = (2.0 * accel * distance + entry * entry + exit * exit) / 2.0;
        let peak = peak_sqr.max(0.0).sqrt();
        (peak - entry).max(0.0) / accel + (peak - exit).max(0.0) / accel
    }
}

fn print_planned(block: PlannedBlock<swarf_gcode::Command>) {
    match block {
        PlannedBlock::Motion {
            start,
            target,
            distance,
            entry_speed,
            nominal_speed,
            exit_speed,
            acceleration,
            is_rapid,
        } => {
            let kind = if is_rapid { "RAPID" } else { "FEED " };
            let duration = duration_secs(entry_speed, nominal_speed, exit_speed, acceleration, distance);
            println!(
                "MOVE {kind} ({:>8.3}, {:>8.3}, {:>8.3}) -> ({:>8.3}, {:>8.3}, {:>8.3})  d={distance:>8.3}mm  entry={:>7.1} nominal={:>7.1} exit={:>7.1} mm/min  t={:>6.3}s",
                start.x, start.y, start.z, target.x, target.y, target.z,
                entry_speed * 60.0, nominal_speed * 60.0, exit_speed * 60.0, duration,
            );
        }
        PlannedBlock::Command(c) => println!("CMD  {c:?}"),
    }
}

const DEMO: &str = "\
G21 G90 G17
G0 X0 Y0 Z5
M3 S1200
G1 Z-2 F200
G1 X20 Y0 F3000
G1 X20 Y0.5
G1 X40 Y0.5
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

    // Reasonable generic-mill defaults for this demo tool - a real host
    // would supply its own machine's actual settings.
    let limits = MachineLimits {
        axes: [
            AxisLimits { max_velocity: 3000.0, max_acceleration: 500.0 },
            AxisLimits { max_velocity: 3000.0, max_acceleration: 500.0 },
            AxisLimits { max_velocity: 600.0, max_acceleration: 100.0 },
        ],
        junction_deviation: 0.01,
        arc_tolerance: 0.002,
    };
    // Generously sized: `interp.run(&src)` processes the whole file in
    // one call with no chance to drain the queue in between (the real
    // streaming pattern - interleaving `Interpreter::step` per line with
    // `GcodePlanner::pop_ready` - needs the sink to be reachable between
    // steps, which `Interpreter` doesn't expose; not a planner bug, just
    // this demo's simplicity). A fixed array this size costs nothing (no
    // heap) and comfortably fits any of the real files under
    // `swarf-gcode/examples/gcode_samples/`.
    let planner: GcodePlanner<4096> = GcodePlanner::new(limits);

    let mut interp = Interpreter::new(planner, PrintingErrors);
    interp.run(&src);
    let (mut planner, _errors) = interp.into_sinks();

    while let Some(block) = planner.pop_ready() {
        print_planned(block);
    }
}
