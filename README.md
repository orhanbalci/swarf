# Swarf

A family of Rust crates filling the gaps in the ecosystem for **CNC firmware
development**. The Rust world has mature G-code *parsers*, but little above
them: nothing that tracks modal state, plans motion, or drives steppers as
reusable, `no_std`-friendly building blocks. Swarf aims to be that missing
middle.

The project is being built bottom-up, one layer at a time. The first crate —
and the current focus — is the **interpreter**: the layer that turns parsed
G-code into concrete motion commands.

## Architecture

The intended end-to-end pipeline, from serial bytes to real motion:

```
Serial bytes
  │
  ▼
[Line assembler]        byte-by-byte; watches for \n / \r → one complete line
  │
  ▼
[Parser]                gcode crate → one Block (codes + word addresses)
  │
  ▼
[Interpreter]    ★      swarf-gcode (this repo): mutates persistent modal state,
  │                     validates modal-group conflicts, emits ONE frozen
  │                     ResolvedMotionCommand (an owned snapshot, no live refs)
  ▼
[Ring buffer]           bounded; backpressure stalls the assembler when full;
  │                     full backward/forward replan on every push/pop
  ▼
[Executor]              pops from the front, drives real motion
  ▲
  │
[Real-time channel]     feed hold / flush / overrides — bypasses the buffer
```

★ = the layer implemented today.

## Crates

| Crate          | Status      | Role                                                        |
| -------------- | ----------- | ----------------------------------------------------------- |
| `swarf-gcode`  | in progress | G-code interpreter: parsed G-code → `ResolvedMotionCommand` |
| `swarf-motion` | planned     | Trajectory / acceleration planner (the ring-buffer stage)   |
| `swarf-kinematics` | planned | Cartesian / CoreXY / delta / lathe coordinate transforms    |
| `swarf-step`   | planned     | Step/dir pulse generation and stepper timing                |
| `swarf-hal`    | planned     | Board / MCU hardware abstraction                            |

## `swarf-gcode`

A NIST RS274NGC-compatible, modal-state G-code interpreter. It is the
equivalent of what grblHAL calls `gc_state` and what the NIST reference
implementation calls the interpreter proper — verified against:

- the NIST RS274NGC Interpreter Version 3 spec (modal groups, Table 4),
- grblHAL's `gc_state` / `gcode.c` as a live reference,
- the `gcode` crate's actual parsing API.

It tracks the persistent modal state a motion planner needs (position, active
motion mode, plane, units, distance mode, work offset, feed rate), detects
modal-group conflicts the parser won't catch (e.g. `G0 G1` on one line), and
resolves G90/G91 distance mode and G20/G21 units into absolute,
millimetre-space targets — handing each line off as one immutable
`ResolvedMotionCommand`.

### Example

```rust
use swarf_gcode::{Interpreter, MotionSink, ResolvedMotionCommand};

// The interpreter pushes each resolved move into a sink you provide.
// In real firmware this is the ring buffer; here it just prints.
struct Echo;
impl MotionSink for Echo {
    fn push(&mut self, cmd: ResolvedMotionCommand) -> Result<(), ()> {
        println!("{:?} -> {:?} @ {} mm/min", cmd.motion_mode, cmd.target, cmd.feed_rate);
        Ok(())
    }
}

fn main() {
    let mut interp = Interpreter::new(Echo);
    interp.run("G21 G90\nG0 X10 Y10\nG1 X20 Y20 F300\n");

    for err in interp.take_errors() {
        eprintln!("error: {err:?}");
    }
}
```

### Module map

| Module          | Responsibility                                                       |
| --------------- | -------------------------------------------------------------------- |
| `state`         | `ModalState` — persistent, motion-scoped interpreter state           |
| `modal_groups`  | NIST Table 4 modal-group classification + allocation-free conflict set |
| `motion`        | `ResolvedMotionCommand` — the planner-facing output (owned snapshots) |
| `visitor`       | `Interpreter` + the `MotionSink` boundary trait                      |

### Scope

In scope now: core motion semantics — modal state, modal-group conflict
detection, G0/G1/G2/G3 motion modes, plane/units/distance modes, work offsets,
feed rate.

Deliberately **not** yet implemented (staged for later): parameter and
expression evaluation (`#1`, `#<expr>`), canned cycles (G73, G81–G89), cutter
compensation, tool-length offsets, arc center resolution from I/J/K/R, and
rotary axes (A/B/C).

## Status & roadmap

- Builds and passes its full test suite (21 tests) on a current Rust toolchain.
- The interpreter is currently pinned to `gcode` 0.6 with a workaround for an
  upstream line-grouping defect. The next major step is porting to `gcode`
  0.7's zero-allocation visitor API, which removes that workaround, restores a
  `no_std` build, and reshapes the entry point into the per-line *step
  function* the architecture above calls for.

## Building

```bash
cargo build
cargo test
```

## License

Licensed under the [MIT License](LICENSE).
