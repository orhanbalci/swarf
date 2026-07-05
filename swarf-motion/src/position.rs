//! A plain 3D position - this crate's own type, deliberately NOT
//! `swarf_gcode::Position`. `swarf-motion` has no dependency on
//! `swarf-gcode` at all: a junction-deviation motion planner is a
//! generic capability, useful for any source of coordinated multi-axis
//! moves, not just G-code. Translating from `swarf-gcode`'s own types
//! into this crate's plain API is `swarf-bridge`'s job, a separate
//! crate - see its docs for why that split exists.

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Position {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
