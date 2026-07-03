//! `Command` - Interface 3, the non-motion output boundary.
//!
//! Parallel to `ResolvedMotionCommand` (Interface 2): a downstream
//! consumer of moves shouldn't have to special-case "this thing has no
//! target position" for spindle/coolant/tool-change/dwell/program-flow
//! effects, so those effects get their own enum entirely, rather than
//! being folded into `ResolvedMotionCommand` with a bunch of `Option`
//! fields nobody but M-code handling needs.
//!
//! `Command` and `ResolvedMotionCommand` share one sink
//! (`visitor::OutputSink`, via `visitor::LineOutput`) rather than each
//! having their own - see that module's docs for why: a downstream
//! real-time consumer needs to know exactly where a `Command` (e.g.
//! "spindle on") falls relative to the surrounding moves, which two
//! independent sinks can't express.
//!
//! Same invariant as `ResolvedMotionCommand`: every value here is an
//! owned copy taken at the moment the line resolves, never a reference
//! back into `ModalState`.

/// M0/M1/M2/M30 - what to do with program execution. This crate has no
/// notion of an "optional stop enabled" switch (that's a controller
/// setting, not G-code state), so `OptionalStop` is reported as-is and
/// it's up to the caller to decide whether to honor it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramFlow {
    /// M0 - unconditional stop.
    Stop,
    /// M1 - stop only if the controller's optional-stop switch is on.
    OptionalStop,
    /// M2 - end of program.
    End,
    /// M30 - end of program, and rewind to the start.
    EndAndRewind,
}

/// M3/M4/M5 - spindle state. RPM is whatever S was last set to
/// (modally, like feed rate) at the moment the spindle command runs -
/// see `ModalState::spindle_speed`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpindleCommand {
    /// M3 - spindle on, clockwise, at the given RPM.
    Clockwise(f64),
    /// M4 - spindle on, counterclockwise, at the given RPM.
    CounterClockwise(f64),
    /// M5 - spindle stop.
    Stop,
}

/// M7/M8/M9 - coolant state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoolantCommand {
    /// M7 - mist coolant on.
    Mist,
    /// M8 - flood coolant on.
    Flood,
    /// M9 - all coolant off.
    Off,
}

/// One fully resolved non-motion command - the output of interpreting a
/// line whose effect isn't a move.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Command {
    ProgramFlow(ProgramFlow),
    Spindle(SpindleCommand),
    Coolant(CoolantCommand),
    /// M6 - execute a tool change to whichever tool number the most
    /// recent T word selected (see `ModalState::selected_tool`).
    ToolChange {
        tool: u32,
    },
    /// G4 - dwell for the given number of seconds, taken from its P
    /// word (`0.0` if P was omitted).
    Dwell {
        seconds: f64,
    },
}
