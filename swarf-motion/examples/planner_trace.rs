//! Trace tool: push a small hardcoded synthetic toolpath directly
//! through a `swarf_motion::Planner` (no G-code involved at all - this
//! crate doesn't depend on `swarf-gcode`), then print each finalized
//! block's entry/nominal/exit speed and computed duration. Handy for
//! eyeballing that look-ahead is actually working (a block approaching
//! a sharp corner should visibly show a lower exit speed than its own
//! nominal speed).
//!
//! For tracing a REAL G-code file through this same planner, see
//! `swarf-bridge/examples/gcode_trace.rs` instead - that crate is the
//! thin adapter between `swarf-gcode`'s `Interpreter` and this crate's
//! plain `Planner` API.
//!
//! Run with `cargo run --example planner_trace`.

use swarf_motion::{AxisLimits, MachineLimits, PlannedBlock, Planner, Position};

/// Trapezoid (or triangle, if the block is too short to reach nominal
/// speed) move duration from entry/nominal/exit speed, distance, and
/// acceleration - kept local to this example rather than part of the
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

fn print_planned(block: PlannedBlock<&str>) {
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
        PlannedBlock::Command(label) => println!("CMD  {label}"),
    }
}

fn main() {
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
    // `&str` stands in for whatever a real caller's non-motion command
    // type is - this crate doesn't care, see `queue.rs`'s doc comment.
    let mut planner: Planner<256, &str> = Planner::new(limits);

    let pos = |x: f64, y: f64, z: f64| Position { x, y, z };

    planner.push_linear(pos(0.0, 0.0, 0.0), pos(0.0, 0.0, 5.0), 3000.0, true).unwrap();
    planner.push_command("spindle on").unwrap();
    planner.push_linear(pos(0.0, 0.0, 5.0), pos(0.0, 0.0, -2.0), 200.0, false).unwrap();
    planner.push_linear(pos(0.0, 0.0, -2.0), pos(20.0, 0.0, -2.0), 3000.0, false).unwrap();
    // A sharp near-90-degree corner - look-ahead should visibly slow the
    // PREVIOUS block's exit speed down well before it, not just cap this
    // block's own entry.
    planner.push_linear(pos(20.0, 0.0, -2.0), pos(20.0, 0.5, -2.0), 3000.0, false).unwrap();
    planner.push_linear(pos(20.0, 0.5, -2.0), pos(40.0, 0.5, -2.0), 3000.0, false).unwrap();
    // A full-circle arc, to demonstrate tessellation.
    planner
        .push_arc(pos(40.0, 0.5, -2.0), pos(40.0, 0.5, -2.0), pos(35.0, 0.5, -2.0), true, 1000.0, false)
        .unwrap();
    planner.push_command("spindle off").unwrap();
    planner.push_linear(pos(40.0, 0.5, -2.0), pos(40.0, 0.5, 5.0), 3000.0, true).unwrap();
    planner.flush(); // program done - decelerate to rest

    while let Some(block) = planner.pop_ready() {
        print_planned(block);
    }
}
