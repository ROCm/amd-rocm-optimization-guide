// 16×16 MFMA C/D accumulator layout (CDNA1)
//
// Visual encoding
//   Teal cells  → rows 0–3  (lanes  0–15)  and  8–11  (lanes 32–47)
//   Grey cells  → rows 4–7  (lanes 16–31)  and 12–15  (lanes 48–63)
//   Cell number → accVGPR index for block 0  (block k adds 4k)
//   Column j    → lane offset within the group  (lane = base + j)
//
// Formulas
//   lane    = 16 · ⌊i/4⌋ + j
//   accVGPR = 4b + (i mod 4)

// -- Page ------------------------------------------------------------------
#set page(width: auto, height: auto, margin: 0pt)
#set text(font: "New Computer Modern", fill: rgb("#1e1e1e"), size: 7pt)

// -- Palette ---------------------------------------------------------------
#let teal-light    = rgb("#cceef5")   // even lane groups - light teal tint
#let grey    = rgb("#bfbfbf")   // odd  lane groups - grey tint
#let white   = rgb("#ffffff")
#let ink     = rgb("#1e1e1e")
#let dimmed  = rgb("#444444")
#let rule-major = 0.8pt + rgb("#777777")   // between 4-row bands

// -- Cell geometry ---------------------------------------------------------
#let sz  = 18pt
#let gap = 0pt

// -- Layout functions ------------------------------------------------------
// accVGPR for block 0 is i mod 4
#let acc-vgpr(i) = calc.rem(i, 4)
// lane group alternates every 4 rows: 0 → teal-light, 1 → teal
#let lane-grp(i) = calc.rem(calc.floor(i / 4), 2)

// -- Single cell -----------------------------------------------------------
#let mfma-cell(i, j) = {
  let r  = acc-vgpr(i)
  let bg = if lane-grp(i) == 0 { teal-light } else { grey }
  let top-rule = if i > 0 and calc.rem(i, 4) == 0 { rule-major } else { none }
  rect(
    width: sz, height: sz, fill: bg,
    stroke: (top: top-rule, bottom: none, left: none, right: none),
    align(center + horizon,
      text(size: 9pt, weight: "bold", fill: ink)[#r]
    )
  )
}

// -- Data grid (16 × 16) ---------------------------------------------------
#let data-grid = grid(
  columns: (sz,) * 16,
  rows:    (sz,) * 16,
  column-gutter: gap,
  row-gutter:    gap,
  ..range(16).map(i => range(16).map(j => mfma-cell(i, j))).flatten()
)

// -- Column header (j = 0 … 15) -------------------------------------------
#let col-header = grid(
  columns: (sz,) * 16,
  column-gutter: gap,
  ..range(16).map(j =>
    align(center, text(size: 9pt, fill: dimmed)[#j])
  )
)

// -- Left margin: one label per 4-row band ---------------------------------
#let margin-w = 105pt
#let margin-grid = grid(
  rows: (sz,) * 16,
  ..range(16).map(i => {
    let top-rule = if i > 0 and calc.rem(i, 4) == 0 { rule-major } else { none }
    let label = if calc.rem(i, 4) == 0 {
      let g    = lane-grp(i)
      let base = calc.floor(i / 4) * 16
      let last = base + 15
      let swatch = box(
        width: 7pt, height: 7pt,
        fill: if g == 0 { teal-light } else { grey },
        stroke: 0.3pt + dimmed,
      )
      box(
        width: margin-w, height: sz,
        align(right + horizon,
          grid(
            columns: (24pt, 4pt, 60pt, 4pt, 9pt),
            align(right + horizon, text(size: 9pt, style: "italic", fill: dimmed)[i=#i]),
            [],
            align(right + horizon, text(size: 9pt, fill: dimmed)[lanes #base–#last]),
            [],
            align(center + horizon, swatch),
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

// -- Final layout ----------------------------------------------------------
#pad(
  top: 14pt, bottom: 14pt, left: 10pt, right: 14pt,
  stack(
    dir: ttb, spacing: 6pt,
    grid(
      columns: (margin-w, 10pt, auto),
      align(right + bottom, text(size: 9pt, style: "italic", fill: dimmed)[_i_ ╲ _j_]),
      [],
      col-header,
    ),
    grid(
      columns: (margin-w, 10pt, auto),
      margin-grid,
      [],
      data-grid,
    ),
  )
)
