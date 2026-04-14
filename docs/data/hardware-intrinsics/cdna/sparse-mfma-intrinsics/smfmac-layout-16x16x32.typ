// SMFMAC 16×16×32 (16-bit inputs) register layout
//
// Four sub-figures, left to right:
//   1. A (sparse, 2:4): rows = matrix row i (0..15), cols = compressed k (0..31)
//   2. Compression index: rows = i (0..15), cols = compressed k (0..31)
//   3. B (dense):  rows = k (0..31), cols = matrix col j (0..15)
//   4. D (output): rows = i (0..15), cols = j (0..15)
//
// Lane mapping:
//   A / Idx : lane = 16·⌊k/8⌋  + i      (block of 16 lanes per 8 k-steps)
//   B       : lane = 16·⌊k/16⌋ + j      (block of 16 lanes per 16 k-steps)
//   D       : lane = 16·⌊i/4⌋  + j
//
// VGPR mapping:
//   A   : VGPR = ⌊k/4⌋ mod 2              → v0 or v1
//   Idx : nibble = ⌊k/4⌋ mod 2            → [3:0] or [7:4] of v0
//   B   : VGPR = ⌊k/2⌋ mod 4, half = k mod 2  → v{0..3}{l|h}
//   D   : VGPR = i mod 4                  → v0..v3

#set page(width: auto, height: auto, margin: 0pt)
#set text(font: "New Computer Modern", size: 7pt, fill: rgb("#1e1e1e"))

// ── Palette ──────────────────────────────────────────────────────────────────
#let teal-light = rgb("#cceef5")   // even groups
#let cool-grey  = rgb("#bfbfbf")   // odd  groups
#let white      = rgb("#ffffff")
#let ink        = rgb("#1e1e1e")
#let dimmed     = rgb("#777777")

// ── Geometry ─────────────────────────────────────────────────────────────────
#let sz        = 14pt
#let gap       = 0pt
#let rule-maj  = 0.8pt + rgb("#777777")
#let rule-min  = 0.3pt + rgb("#bfbfbf")

// ── Colour helper ─────────────────────────────────────────────────────────────
#let group-fill(g) = if calc.rem(g, 2) == 0 { teal-light } else { cool-grey }

// ── Base cell ────────────────────────────────────────────────────────────────
#let cell(label, fill) = box(
  width: sz, height: sz, fill: fill,
  stroke: rule-min,
  align(center + horizon, text(size: 5.5pt, label)))

// ─────────────────────────────────────────────────────────────────────────────
// A (sparse) sub-figure
//   rows = i (0..15), cols = compressed k (0..31)
//   VGPR = ⌊k/4⌋ mod 2;  colour group = ⌊k/8⌋
// ─────────────────────────────────────────────────────────────────────────────
#let a-cell(i, k) = {
  let v    = calc.rem(calc.floor(k / 4), 2)
  let g    = calc.floor(k / 8)
  cell("v" + str(v), group-fill(g))
}

#let a-data-grid = grid(
  columns: 32, rows: 16, gutter: gap,
  ..range(16).map(i =>
    range(32).map(k => a-cell(i, k))
  ).flatten()
)

// ─────────────────────────────────────────────────────────────────────────────
// Compression index sub-figure
//   Same row/col layout as A; label = nibble position within v0
//   nibble = ⌊k/4⌋ mod 2 → [3:0] or [7:4]
// ─────────────────────────────────────────────────────────────────────────────
#let ci-cell(i, k) = {
  let n    = calc.rem(calc.floor(k / 4), 2)
  let g    = calc.floor(k / 8)
  let lbl  = if n == 0 { "[3:0]" } else { "[7:4]" }
  cell(lbl, group-fill(g))
}

#let ci-data-grid = grid(
  columns: 32, rows: 16, gutter: gap,
  ..range(16).map(i =>
    range(32).map(k => ci-cell(i, k))
  ).flatten()
)

// ─────────────────────────────────────────────────────────────────────────────
// B (dense) sub-figure
//   rows = k (0..31), cols = j (0..15)
//   VGPR = ⌊k/2⌋ mod 4;  half = k mod 2 → l or h
//   colour group = ⌊k/8⌋
// ─────────────────────────────────────────────────────────────────────────────
#let b-cell(k, j) = {
  let v    = calc.rem(calc.floor(k / 2), 4)
  let half = if calc.rem(k, 2) == 0 { "l" } else { "h" }
  let g    = calc.floor(k / 8)
  cell("v" + str(v) + half, group-fill(g))
}

#let b-data-grid = grid(
  columns: 16, rows: 32, gutter: gap,
  ..range(32).map(k =>
    range(16).map(j => b-cell(k, j))
  ).flatten()
)

// ─────────────────────────────────────────────────────────────────────────────
// D (output) sub-figure
//   rows = i (0..15), cols = j (0..15)
//   VGPR = i mod 4;  colour group = ⌊i/4⌋
// ─────────────────────────────────────────────────────────────────────────────
#let d-cell(i, j) = {
  let v = calc.rem(i, 4)
  let g = calc.floor(i / 4)
  cell("v" + str(v), group-fill(g))
}

#let d-data-grid = grid(
  columns: 16, rows: 16, gutter: gap,
  ..range(16).map(i =>
    range(16).map(j => d-cell(i, j))
  ).flatten()
)

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

// Column header row
#let col-hdr(n, start: 0) = grid(
  columns: n, gutter: gap,
  ..range(n).map(c =>
    box(width: sz, height: sz * 0.9,
      align(center + bottom, text(size: 5pt, fill: dimmed, str(c + start))))
  )
)

// Row-label column
#let row-labels(labels) = grid(
  columns: 1, gutter: gap,
  ..labels.map(lbl =>
    box(width: sz * 2.2, height: sz,
      align(right + horizon, text(size: 5pt, fill: dimmed, lbl + " ")))
  )
)

// Horizontal separator between header and grid
#let hrule(w) = line(length: w, stroke: rule-maj)

// Sub-figure: title + col header + rule + (row labels | data grid)
#let sub-fig(title, col-header, label-col, data-grid, total-width) = {
  stack(dir: ttb, spacing: 2pt,
    align(center, text(weight: "bold", title)),
    pad(left: sz * 2.2, col-header),
    hrule(total-width),
    stack(dir: ltr, label-col, data-grid),
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Assemble sub-figures
// ─────────────────────────────────────────────────────────────────────────────

// A: 16 rows (i) × 32 cols (k)
#let a-label-w  = sz * 2.2
#let a-grid-w   = sz * 32
#let a-total-w  = a-label-w + a-grid-w

#let fig-a = sub-fig(
  "A (sparse, 2:4)",
  col-hdr(32),
  row-labels(range(16).map(i => "i=" + str(i))),
  a-data-grid,
  a-total-w,
)

// Compression index: 16 rows (i) × 32 cols (k); same dimensions as A
#let fig-ci = sub-fig(
  "Compression index (v0)",
  col-hdr(32),
  row-labels(range(16).map(i => "i=" + str(i))),
  ci-data-grid,
  a-total-w,
)

// B: 32 rows (k) × 16 cols (j)
#let b-label-w = sz * 2.2
#let b-grid-w  = sz * 16
#let b-total-w = b-label-w + b-grid-w

#let fig-b = sub-fig(
  "B (dense)",
  col-hdr(16),
  row-labels(range(32).map(k => "k=" + str(k))),
  b-data-grid,
  b-total-w,
)

// D: 16 rows (i) × 16 cols (j)
#let d-label-w = sz * 2.2
#let d-grid-w  = sz * 16
#let d-total-w = d-label-w + d-grid-w

#let fig-d = sub-fig(
  "D / C (output / input)",
  col-hdr(16),
  row-labels(range(16).map(i => "i=" + str(i))),
  d-data-grid,
  d-total-w,
)

// ─────────────────────────────────────────────────────────────────────────────
// Caption
// ─────────────────────────────────────────────────────────────────────────────
#let caption-text = [
  SMFMAC 16×16, 16-bit input elements — operand register layout. \
  *A* (sparse): 2 VGPRs (v0–v1), 2:4 structured sparsity along the k-dimension;
  VGPR = ⌊k/4⌋ mod 2, lane = 16·⌊k/8⌋ + i. \
  *Compression index*: 1 VGPR (v0), nibble [3:0] for k∈{0..3,8..11,…},
  nibble [7:4] for k∈{4..7,12..15,…}; same lane assignment as A. \
  *B* (dense): 4 VGPRs (v0–v3), lo/hi 16-bit half per k-step;
  VGPR = ⌊k/2⌋ mod 4, half = k mod 2, lane = 16·⌊k/16⌋ + j. \
  *D*: 4 VGPRs (v0–v3); VGPR = i mod 4, lane = 16·⌊i/4⌋ + j.
  Colour groups of 8 k-rows (A/B) or 4 i-rows (D) alternate teal/grey.
]

// ─────────────────────────────────────────────────────────────────────────────
// Top-level composition
// ─────────────────────────────────────────────────────────────────────────────
#pad(top: 14pt, rest: 14pt,
  stack(dir: ttb, spacing: 14pt,
    grid(
      columns: (a-total-w, a-total-w, b-total-w, d-total-w),
      column-gutter: 20pt,
      align: horizon,
      fig-a, fig-ci, fig-b, fig-d,
    ),
    block(width: (a-total-w + a-total-w + b-total-w + d-total-w + 3 * 20pt),
      align(left, text(size: 6pt, fill: dimmed, caption-text))),
  )
)
