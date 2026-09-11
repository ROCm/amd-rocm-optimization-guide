// Per-lane assignment for RDNA4's GLOBAL_LOAD_TR_B128 (v8f16 overload)
//
// Hardware-verified (not derived from an ISA-manual diagram): sourced from
// a standalone correctness test (global_load_tr_wmma.hip) that checks the
// kernel's output against a CPU reference on real gfx1200 hardware.
//
// One call covers an 8(row)x32(col) sub-tile -- half the K-depth of a
// 16-wide WMMA K-step, so two calls (second shifted by 8 rows) fill a
// full 16x32 B strip. Contrast with CDNA5's tr16_b128, which covers a
// full 16x16 tile in a single call.
//
// Verified formula (lane tx, return-vector element e, both 0-indexed):
//   in_r = (tx/8)*2 + e/4          -- row (K) position, 0-7
//   in_c = (e%4)*8 + tx%8          -- column (N) position, 0-31
//
// Inverted for this grid (row = in_r, col = in_c):
//   lane = (in_r/2)*8 + (in_c%8)
//   e    = (in_r%2)*4 + (in_c/8)

// ── Page ──────────────────────────────────────────────────────────────────
#set page(width: auto, height: auto, margin: 0pt)
#set text(font: "New Computer Modern", fill: rgb("#1e1e1e"), size: 7pt)

// ── Palette ───────────────────────────────────────────────────────────────
#let rose-light = rgb("#f5cccd")   // even rows
#let grey       = rgb("#bfbfbf")   // odd rows
#let white      = rgb("#ffffff")
#let ink        = rgb("#1e1e1e")
#let dimmed     = rgb("#777777")

// ── Cell geometry ─────────────────────────────────────────────────────────
#let cw = 20pt
#let ch = 24pt
#let gap = 0pt
#let lw  = 46pt   // row-label width

// ── Formulas ──────────────────────────────────────────────────────────────
#let lane-of(row, col) = calc.floor(row / 2) * 8 + calc.rem(col, 8)
#let e-of(row, col)    = calc.rem(row, 2) * 4 + calc.floor(col / 8)

// ── Data grid: 8 rows (K) x 32 columns (N) ─────────────────────────────────
#let data-grid = grid(
  columns: (cw,) * 32,
  rows:    (ch,) * 8,
  column-gutter: gap,
  row-gutter:    gap,
  ..range(8).map(row =>
    range(32).map(col => {
      let bg = if calc.rem(row, 2) == 0 { rose-light } else { grey }
      rect(
        width: cw, height: ch, fill: bg, stroke: 0.4pt + ink,
        align(center + horizon,
          stack(dir: ttb, spacing: 1pt,
            text(size: 6pt, weight: "bold", fill: ink)[#lane-of(row, col)],
            text(size: 4.5pt, fill: dimmed)[e=#e-of(row, col)],
          )
        )
      )
    })
  ).flatten()
)

#let col-header = grid(
  columns: (cw,) * 32,
  column-gutter: gap,
  ..range(32).map(col =>
    align(center, text(size: 5pt, fill: dimmed)[#col])
  )
)

#let row-labels = stack(dir: ttb, spacing: 0pt,
  ..range(8).map(row =>
    box(width: lw, height: ch,
      align(right + horizon,
        text(size: 5.5pt, style: "italic", fill: dimmed)[K=#row #h(2pt)]
      )
    )
  )
)

// ── Legend ────────────────────────────────────────────────────────────────
#let swatch(col, label) = grid(
  columns: (10pt, 3pt, auto),
  column-gutter: 0pt,
  rect(width: 10pt, height: 8pt, fill: col, stroke: 0.4pt + ink),
  [],
  align(left + horizon, text(size: 5.5pt, fill: dimmed)[ #label]),
)
#let legend = grid(
  columns: (auto, 10pt, auto),
  column-gutter: 0pt,
  swatch(rose-light, [K even]),
  [],
  swatch(grey, [K odd]),
)

// ── Title and caption ───────────────────────────────────────────────────────
#let title = text(size: 8pt, weight: "bold")[
  GLOBAL_LOAD_TR_B128 (v8f16) per-lane assignment -- wave32 (RDNA4)
]

#let caption-text = text(size: 6pt, fill: dimmed)[
  Cell shows lane index (bold) and return-vector element _e_ (small). One \
  call covers K = 0--7 across all N = 0--31; a second call shifted by 8 \
  rows fills K = 8--15. Formula: lane = ⌊K/2⌋·8 + (N mod 8), \
  e = (K mod 2)·4 + ⌊N/8⌋.
]

// ── Final layout ──────────────────────────────────────────────────────────
#pad(top: 14pt, bottom: 14pt, left: 10pt, right: 14pt,
  stack(dir: ttb, spacing: 4pt,
    title,
    v(2pt),
    grid(columns: (lw, auto), column-gutter: gap,
      [], align(center, text(size: 5.5pt, fill: dimmed)[N])),
    grid(columns: (lw, auto), column-gutter: gap, [], col-header),
    grid(columns: (lw, auto), column-gutter: gap, row-labels, data-grid),
    v(4pt),
    legend,
    v(2pt),
    caption-text,
  )
)
