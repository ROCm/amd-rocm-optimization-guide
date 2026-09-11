// 16×16 WMMA Accumulator layout (RDNA 4, wave32)
//
// Visual encoding
//   Rose cells → rows  0– 7 (lanes  0–15)
//   Grey cells → rows  8–15 (lanes 16–31)
//   Cell value → VGPR index g (0–7)
//   Column j   → matrix column = lane % 16
//
// Formulas (lane L, VGPR g)
//   row = (L / 16) * 8 + g
//   col = L % 16
//
// Inverse (row i, col j)
//   lane = (i / 8) * 16 + j
//   VGPR = i % 8

// -- Page ------------------------------------------------------------------
#set page(width: auto, height: auto, margin: 0pt)
#set text(font: "New Computer Modern", fill: rgb("#1e1e1e"), size: 7pt)

// -- Palette ---------------------------------------------------------------
#let rose-light = rgb("#f5cccd")   // lanes  0–15 (rows 0–7)
#let grey       = rgb("#bfbfbf")   // lanes 16–31 (rows 8–15)
#let white      = rgb("#ffffff")
#let ink        = rgb("#1e1e1e")
#let dimmed     = rgb("#777777")
#let rule-major = 0.8pt + rgb("#777777")   // between the two lane groups

// -- Cell geometry ---------------------------------------------------------
#let sz  = 14pt
#let gap = 0pt

// -- Layout functions ------------------------------------------------------
// VGPR index: position within the lane's 8-element output vector
#let vgpr(i) = calc.rem(i, 8)
// Lane group: 0 → rose (rows 0–7), 1 → grey (rows 8–15)
#let lane-grp(i) = calc.floor(i / 8)

// -- Single cell -----------------------------------------------------------
#let d-cell(i, j) = {
  let g  = vgpr(i)
  let bg = if lane-grp(i) == 0 { rose-light } else { grey }
  let top-rule = if i == 8 { rule-major } else { none }
  rect(
    width: sz, height: sz, fill: bg,
    stroke: (top: top-rule, bottom: none, left: none, right: none),
    align(center + horizon,
      text(size: 6pt, weight: "bold", fill: ink)[#g]
    )
  )
}

// -- Data grid (16 rows × 16 cols) -----------------------------------------
#let data-grid = grid(
  columns: (sz,) * 16,
  rows:    (sz,) * 16,
  column-gutter: gap,
  row-gutter:    gap,
  ..range(16).map(i => range(16).map(j => d-cell(i, j))).flatten()
)

// -- Column header (j = 0 … 15) -------------------------------------------
#let col-header = grid(
  columns: (sz,) * 16,
  column-gutter: gap,
  ..range(16).map(j =>
    align(center, text(size: 5pt, fill: dimmed)[#j])
  )
)

// -- Left margin: one label per 8-row band ---------------------------------
#let margin-w = 80pt
#let margin-grid = grid(
  rows: (sz,) * 16,
  ..range(16).map(i => {
    let top-rule = if i == 8 { rule-major } else { none }
    let label = if calc.rem(i, 8) == 0 {
      let g    = lane-grp(i)
      let base = g * 16
      let last = base + 15
      let row0 = g * 8
      let row1 = row0 + 7
      let swatch = box(
        width: 6pt, height: 6pt,
        fill: if g == 0 { rose-light } else { grey },
        stroke: 0.3pt + dimmed,
      )
      box(
        height: sz,
        align(right + horizon,
          grid(
            columns: (auto, 8pt, auto, 4pt, auto),
            align(right + horizon, text(size: 5.5pt, style: "italic", fill: dimmed)[i=#row0–#row1]),
            [],
            align(right + horizon, text(size: 5.5pt, fill: dimmed)[lanes #base–#last]),
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

// -- Caption ---------------------------------------------------------------
#let caption-text = [
  #set text(size: 6pt, fill: dimmed)
  Accumulator for all 16×16 WMMA intrinsics on RDNA4. \
  Cell value = VGPR index _g_ (0–7). Column _j_ = matrix column = lane % 16. \
  Formulas: lane = ⌊_i_/8⌋ · 16 + _j_ ;  VGPR = _i_ mod 8.
]

// -- Title -----------------------------------------------------------------
#let title = text(size: 8pt, weight: "bold")[
  16×16 WMMA accumulator layout - wave32 (RDNA4)
]

// -- Final layout ----------------------------------------------------------
#pad(
  top: 14pt, bottom: 14pt, left: 10pt, right: 14pt,
  stack(
    dir: ttb, spacing: 6pt,
    grid(
      columns: (margin-w, 4pt, auto),
      [], [],
      title,
    ),
    grid(
      columns: (margin-w, 4pt, auto),
      align(right + bottom, text(size: 5.5pt, style: "italic", fill: dimmed)[_i_ ╲ _j_]),
      [],
      col-header,
    ),
    grid(
      columns: (margin-w, 4pt, auto),
      margin-grid,
      [],
      data-grid,
    ),
    v(2pt),
    grid(
      columns: (margin-w, 4pt, auto),
      [], [],
      caption-text,
    ),
  )
)
