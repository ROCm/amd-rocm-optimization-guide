// SMFMAC 32×32×32 (8-bit inputs) register layout
//
// Four sub-figures, left to right:
//   1. A (sparse, 2:4): rows = matrix row i (0..31), cols = compressed k (0..31)
//   2. Compression index: rows = i (0..31), cols = compressed k (0..31)
//   3. B (dense):  rows = k (0..31), cols = matrix col j (0..31)
//   4. D (output): rows = i (0..31), cols = j (0..31)
//
// Lane mapping:
//   A / Idx : lane = i + 32·⌊k/16⌋   (two k-blocks of 16)
//   B       : lane = j + 32·⌊k/16⌋   (two k-blocks of 16, j = 0..31)
//   D       : lane = (32·⌊i/4⌋) mod 64 + j
//
// VGPR mapping:
//   A   : VGPR = ⌊k/8⌋ mod 2               → v0 (k mod 16 < 8) or v1 (k mod 16 ≥ 8)
//   Idx : nibble = ⌊(k mod 16)/4⌋           → n0 [3:0], n1 [7:4], n2 [11:8], n3 [15:12] of v0
//   B   : VGPR = ⌊k/4⌋ mod 4               → v0..v3; byte = k mod 4
//   D   : VGPR = 4·⌊i/8⌋ + (i mod 4)      → v0..v15

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
//   rows = i (0..31), cols = compressed k (0..31)
//   VGPR = ⌊k/8⌋ mod 2;  colour group = ⌊k/16⌋  (2 groups)
// ─────────────────────────────────────────────────────────────────────────────
#let a-cell(i, k) = {
  let v = calc.rem(calc.floor(k / 8), 2)
  let g = calc.floor(k / 16)
  cell("v" + str(v), group-fill(g))
}

#let a-data-grid = grid(
  columns: 32, rows: 32, gutter: gap,
  ..range(32).map(i =>
    range(32).map(k => a-cell(i, k))
  ).flatten()
)

// ─────────────────────────────────────────────────────────────────────────────
// Compression index sub-figure
//   Same row/col layout as A; nibble = ⌊(k mod 16)/4⌋ within v0
//   n0=[3:0], n1=[7:4], n2=[11:8], n3=[15:12]
// ─────────────────────────────────────────────────────────────────────────────
#let ci-cell(i, k) = {
  let n   = calc.rem(calc.floor(k / 4), 4)
  let g   = calc.floor(k / 16)
  cell("n" + str(n), group-fill(g))
}

#let ci-data-grid = grid(
  columns: 32, rows: 32, gutter: gap,
  ..range(32).map(i =>
    range(32).map(k => ci-cell(i, k))
  ).flatten()
)

// ─────────────────────────────────────────────────────────────────────────────
// B (dense) sub-figure
//   rows = k (0..31), cols = j (0..31)
//   VGPR = ⌊k/4⌋ mod 4  → v0..v3
//   colour group = ⌊k/16⌋  (2 groups)
// ─────────────────────────────────────────────────────────────────────────────
#let b-cell(k, j) = {
  let v = calc.rem(calc.floor(k / 4), 4)
  let g = calc.floor(k / 16)
  cell("v" + str(v), group-fill(g))
}

#let b-data-grid = grid(
  columns: 32, rows: 32, gutter: gap,
  ..range(32).map(k =>
    range(32).map(j => b-cell(k, j))
  ).flatten()
)

// ─────────────────────────────────────────────────────────────────────────────
// D (output) sub-figure
//   rows = i (0..31), cols = j (0..31)
//   VGPR = 4·⌊i/8⌋ + (i mod 4)  (0..15)
//   colour group = ⌊i/8⌋  (4 groups)
// ─────────────────────────────────────────────────────────────────────────────
#let d-cell(i, j) = {
  let v = 4 * calc.floor(i / 8) + calc.rem(i, 4)
  let g = calc.floor(i / 8)
  cell("v" + str(v), group-fill(g))
}

#let d-data-grid = grid(
  columns: 32, rows: 32, gutter: gap,
  ..range(32).map(i =>
    range(32).map(j => d-cell(i, j))
  ).flatten()
)

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

// Column header row
#let col-hdr(n) = grid(
  columns: n, gutter: gap,
  ..range(n).map(c =>
    box(width: sz, height: sz * 0.9,
      align(center + bottom, text(size: 5pt, fill: dimmed, str(c))))
  )
)

// Row-label column
#let row-labels(labels) = grid(
  columns: 1, gutter: gap,
  ..labels.map(lbl =>
    box(width: sz * 2.2, height: sz,
      align(right + horizon, text(size: 5pt, fill: dimmed, lbl + " "))))
)

// Horizontal separator
#let hrule(w) = line(length: w, stroke: rule-maj)

// Sub-figure
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

#let label-w  = sz * 2.2

// A: 32 rows (i) × 32 cols (k)
#let a-grid-w = sz * 32
#let a-total  = label-w + a-grid-w

#let fig-a = sub-fig(
  "A (sparse, 2:4)",
  col-hdr(32),
  row-labels(range(32).map(i => "i=" + str(i))),
  a-data-grid,
  a-total,
)

// CI: same dimensions as A
#let fig-ci = sub-fig(
  "Compression index (v0)",
  col-hdr(32),
  row-labels(range(32).map(i => "i=" + str(i))),
  ci-data-grid,
  a-total,
)

// B: 32 rows (k) × 32 cols (j)
#let b-grid-w = sz * 32
#let b-total  = label-w + b-grid-w

#let fig-b = sub-fig(
  "B (dense)",
  col-hdr(32),
  row-labels(range(32).map(k => "k=" + str(k))),
  b-data-grid,
  b-total,
)

// D: 32 rows (i) × 32 cols (j)
#let d-grid-w = sz * 32
#let d-total  = label-w + d-grid-w

#let fig-d = sub-fig(
  "D / C (output / input)",
  col-hdr(32),
  row-labels(range(32).map(i => "i=" + str(i))),
  d-data-grid,
  d-total,
)

// ─────────────────────────────────────────────────────────────────────────────
// Caption
// ─────────────────────────────────────────────────────────────────────────────
#let caption-text = [
  SMFMAC 32×32, 8-bit input elements — operand register layout. \
  *A* (sparse): 2 VGPRs (v0–v1), 2:4 structured sparsity along the k-dimension;
  VGPR = ⌊k/8⌋ mod 2, lane = i + 32·⌊k/16⌋. \
  *Compression index*: 1 VGPR (v0), nibble n0–n3 packed in bits [15:0];
  nibble = ⌊(k mod 16)/4⌋; same lane assignment as A. \
  *B* (dense): 4 VGPRs (v0–v3), one byte per k-step;
  VGPR = ⌊k/4⌋ mod 4, lane = j + 32·⌊k/16⌋. \
  *D*: 16 VGPRs (v0–v15); VGPR = 4·⌊i/8⌋ + (i mod 4),
  lane = (32·⌊i/4⌋) mod 64 + j.
  Colour groups of 16 k-columns (A/B) or 8 i-rows (D) alternate teal/grey.
]

// ─────────────────────────────────────────────────────────────────────────────
// Top-level composition
// ─────────────────────────────────────────────────────────────────────────────
#let total-w = a-total + a-total + b-total + d-total + 3 * 20pt

#pad(top: 14pt, rest: 14pt,
  stack(dir: ttb, spacing: 14pt,
    stack(dir: ltr, spacing: 20pt,
      fig-a,
      fig-ci,
      fig-b,
      fig-d,
    ),
    block(width: total-w,
      align(left, text(size: 6pt, fill: dimmed, caption-text))),
  )
)
