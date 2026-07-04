# Third-party G-code sample files

The files in this directory are test fixtures taken from
[Universal-G-Code-Sender](https://github.com/winder/Universal-G-Code-Sender)
(`test_files/`), fetched from commit
[`54d5c80`](https://github.com/winder/Universal-G-Code-Sender/tree/54d5c806242ae4861453da2ace26705c8147389c/test_files)
on 2026-07-04.

Universal-G-Code-Sender is licensed under the
[GNU General Public License v3.0](https://github.com/winder/Universal-G-Code-Sender/blob/master/LICENSE).
That license applies to these files, **not** to `swarf-gcode`'s own MIT
license (see the repository root `LICENSE`) - they are included here only
as plain-text test data for exercising this crate's interpreter (see
`examples/trace.rs`), not as part of the crate's own source or
distributed binary.

Files included, and roughly what each one is useful for testing:

| File | Exercises |
|---|---|
| `circle.gcode` | Basic G17 arcs, inch units (G20) |
| `comments.gcode` | `;` and `(...)` comment styles, inline comments |
| `g17-g18-g19.gcode` | Plane selection (G17/G18/G19) with R-word arcs in each |
| `spiral.gcode` | Full-circle arcs (I/J only, explicit same-point X/Y), helical Z |
| `line_skip_test.gcode` | Work coordinate selection (G55), dwell (G4), mixed axis updates |
| `no_spaces.gcode` | Words packed with no whitespace between letter and number |
| `square.nc` | Large coordinates, repeated absolute/incremental (G90/G91) switching |
| `ruler.gcode` | A long real-world-shaped straight-line/drill program |
| `buffer_stress_test.gcode` | Many short lines in quick succession |
| `arc_rword_test.gcode` | Real CAM-generated (SketchUcam) toolpath, heavy R-word arc usage |
| `ugs.gcode` | A large, realistic multi-thousand-line job |
| `serial_stress_test.gcode` | Another large multi-line throughput test |

See `examples/trace.rs`'s module docs for how to run these through the
interpreter's trace tool.

## `rust_logo.gcode`

Not from UGS. Generated with [svg2gcode](https://github.com/sameer/svg2gcode)
from the Rust logo SVG in
[simple-icons](https://github.com/simple-icons/simple-icons/blob/develop/icons/rust.svg),
which is released under
[CC0 1.0](https://github.com/simple-icons/simple-icons/blob/develop/LICENSE.md)
(public domain, no attribution required). Exercises G2/G3 R-word arcs on a
real closed-contour toolpath (gear ring, R cutout, mounting bolts) with no
fill/relief artifacts.

svg2gcode's raw output is single-depth (pen-plotter/laser style: one M3/M5
per contour, no Z motion at all), which isn't a real cut - so each of the
10 contours was reworked into 2 full-depth routing passes (Z-2.5, Z-5.2mm)
to actually cut through 5mm stock, with a distinct plunge feed rate (300
mm/min) from the cutting feed rate (1000 mm/min).

Contours are also ordered interior-features-first, outer-perimeter-last:
the outer gear ring is the only contour whose through-cut fully separates
the part from the surrounding stock, so it runs last - cutting it first
would leave the part loose (free to shift/vibrate under the bit) while
the interior features (R cutout, mounting bolt holes) still had cutting
left to do.
