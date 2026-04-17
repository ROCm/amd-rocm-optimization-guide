// V_MFMA_F32_32X32X1F32 — C/D accumulator layout (CDNA)
//
// Visual encoding
//   Warm cells  → lane group 0, lanes  0–31  (rows where ⌊i/4⌋ is even)
//   Cool cells  → lane group 1, lanes 32–63  (rows where ⌊i/4⌋ is odd)
//   Cell number → accVGPR index, block 0  (block 1 adds 16)
//   Column j    → lane offset within the group  (lane = base + j)
//
// Formulas
//   lane    = (32 · ⌊i/4⌋) mod 64 + j
//   accVGPR = 4 · ⌊i/8⌋ + (i mod 4)

// ── Page ──────────────────────────────────────────────────────────────────
#set page(width: auto, height: auto, margin: 0pt)
#set text(font: "New Computer Modern", fill: rgb("#1e1e1e"), size: 7pt)

// ── Palette ───────────────────────────────────────────────────────────────
#let warm    = rgb("#cceef5")   // lane group 0  (lanes  0–31) — light teal tint
#let cool    = rgb("#bfbfbf")   // lane group 1  (lanes 32–63) — grey tint
#let white   = rgb("#ffffff")
#let ink     = rgb("#1e1e1e")
#let dimmed  = rgb("#444444")
#let rule-major = 0.8pt + rgb("#777777")   // between accVGPR groups (every 8 rows)
#let rule-minor = none                      // between lane-group bands — colour contrast suffices

// ── Cell geometry ─────────────────────────────────────────────────────────
#let sz  = 14pt    // cell side
#let gap = 0pt     // no gap — colour contrast provides separation

// ── Layout functions ──────────────────────────────────────────────────────
#let acc-vgpr(i) = 4 * calc.floor(i / 8) + calc.rem(i, 4)
#let lane-grp(i) = calc.rem(calc.floor(i / 4), 2)   // 0 → warm, 1 → cool

// ── Single cell ───────────────────────────────────────────────────────────
#let mfma-cell(i, j) = {
  let r  = acc-vgpr(i)
  let bg = if lane-grp(i) == 0 { warm } else { cool }
  // Major horizontal rule drawn via the top stroke of the first row in each 8-row group
  let top-rule = if i > 0 and calc.rem(i, 8) == 0 { rule-major } else { none }
  rect(
    width: sz, height: sz, fill: bg,
    stroke: (top: top-rule, bottom: none, left: none, right: none),
    align(center + horizon,
      text(size: 6pt, weight: "bold", fill: ink)[#r]
    )
  )
}

// ── Data grid (32 × 32) ───────────────────────────────────────────────────
#let data-grid = grid(
  columns: (sz,) * 32,
  rows:    (sz,) * 32,
  column-gutter: gap,
  row-gutter:    gap,
  ..range(32).map(i => range(32).map(j => mfma-cell(i, j))).flatten()
)

// ── Column header (j = 0 … 31) ───────────────────────────────────────────
#let col-header = grid(
  columns: (sz,) * 32,
  column-gutter: gap,
  ..range(32).map(j =>
    align(center, text(size: 6pt, fill: dimmed)[#j])
  )
)

// ── Left margin: one label per 4-row band ─────────────────────────────────
// At each band start: row index i on the left, lane range + swatch on the right.
#let margin-w = 80pt
#let margin-grid = grid(
  rows: (sz,) * 32,
  ..range(32).map(i => {
    let top-rule = if i > 0 and calc.rem(i, 8) == 0 { rule-major } else { none }
    let label = if calc.rem(i, 4) == 0 {
      let g    = lane-grp(i)
      let base = if g == 0 { 0 } else { 32 }
      let last = base + 31
      let swatch = box(
        width: 6pt, height: 6pt,
        fill: if g == 0 { warm } else { cool },
        stroke: 0.3pt + dimmed,
      )
      box(
        height: sz,
        align(right + horizon,
          grid(
            columns: (auto, 8pt, auto, 4pt, auto),
            // row index
            align(right + horizon, text(size: 6pt, style: "italic", fill: dimmed)[i=#i]),
            [],
            // lane range
            align(right + horizon, text(size: 6pt, fill: dimmed)[lanes #base–#last]),
            [],
            align(horizon, swatch),
          )
        )
      )
    } else {
      box(height: sz)
    }
    rect(
      width: margin-w, height: sz, fill: white,
      stroke: (top: top-rule, bottom: none, left: none, right: none),
      label
    )
  })
)

// ── Final layout ──────────────────────────────────────────────────────────
#pad(
  top: 14pt, bottom: 14pt, left: 10pt, right: 14pt,
  stack(
    dir: ttb, spacing: 6pt,
    // Column index row
    grid(
      columns: (margin-w, 4pt, auto),
      // label above margin: italic "i"
      align(right + bottom, text(size: 6pt, style: "italic", fill: dimmed)[_i_ ╲ _j_]),
      [],
      col-header,
    ),
    // Main body: margin + data
    grid(
      columns: (margin-w, 4pt, auto),
      margin-grid,
      [],
      data-grid,
    ),
  )
)
