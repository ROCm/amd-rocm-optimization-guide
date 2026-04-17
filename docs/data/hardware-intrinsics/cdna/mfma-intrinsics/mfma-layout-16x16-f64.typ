// V_MFMA_F64_16X16X4F64 — C/D accumulator layout (CDNA2)
//
// Visual encoding
//   Teal cells  → rows 0, 1, 2, 3  (i mod 4 = 0,1,2,3) — lane groups 0–15, 16–31, 32–47, 48–63
//   Grey cells  → same rows but at odd accVGPR-pair indices
//   Cell number → accVGPR pair index k = ⌊i/4⌋  (physical regs v[2k+1:2k])
//   Column j    → lane offset within the 16-lane group  (lane = 16·(i mod 4) + j)
//
// The grid is 16 rows × 16 columns, mirroring the matrix shape.
// Each group of 4 consecutive rows shares the same 16 lanes but uses a different accVGPR pair.
// Colour alternates between teal (accVGPR pair 0, 2) and grey (accVGPR pair 1, 3).
//
// Formulas
//   lane           = 16 · (i mod 4) + j
//   accVGPR pair k = ⌊i/4⌋   (physical regs v[2k+1:2k])

// ── Page ──────────────────────────────────────────────────────────────────
#set page(width: auto, height: auto, margin: 0pt)
#set text(font: "New Computer Modern", fill: rgb("#1e1e1e"), size: 7pt)

// ── Palette ───────────────────────────────────────────────────────────────
#let teal-light  = rgb("#cceef5")   // even accVGPR pairs (0, 2)
#let grey        = rgb("#bfbfbf")   // odd  accVGPR pairs (1, 3)
#let white       = rgb("#ffffff")
#let ink         = rgb("#1e1e1e")
#let dimmed      = rgb("#444444")
#let rule-major  = 0.8pt + rgb("#777777")   // between accVGPR-pair groups (every 4 rows)

// ── Cell geometry ─────────────────────────────────────────────────────────
#let sz  = 18pt
#let gap = 0pt

// ── Layout functions ──────────────────────────────────────────────────────
// accVGPR pair index (= the label shown in each cell)
#let acc-pair(i) = calc.floor(i / 4)
// colour group: alternates per accVGPR pair
#let colour-grp(i) = calc.rem(calc.floor(i / 4), 2)

// ── Single cell ───────────────────────────────────────────────────────────
#let mfma-cell(i, j) = {
  let k  = acc-pair(i)
  let bg = if colour-grp(i) == 0 { teal-light } else { grey }
  let top-rule = if i > 0 and calc.rem(i, 4) == 0 { rule-major } else { none }
  rect(
    width: sz, height: sz, fill: bg,
    stroke: (top: top-rule, bottom: none, left: none, right: none),
    align(center + horizon,
      text(size: 9pt, weight: "bold", fill: ink)[#k]
    )
  )
}

// ── Data grid (16 × 16) ───────────────────────────────────────────────────
#let data-grid = grid(
  columns: (sz,) * 16,
  rows:    (sz,) * 16,
  column-gutter: gap,
  row-gutter:    gap,
  ..range(16).map(i => range(16).map(j => mfma-cell(i, j))).flatten()
)

// ── Column header (j = 0 … 15, = lane offset within group) ──────────────
#let col-header = grid(
  columns: (sz,) * 16,
  column-gutter: gap,
  ..range(16).map(j =>
    align(center, text(size: 9pt, fill: dimmed)[#j])
  )
)

// ── Left margin: one label per row ────────────────────────────────────────
// Each row i maps to a distinct 16-lane group: lanes 16·(i mod 4) … 16·(i mod 4)+15.
// A horizontal rule separates accVGPR-pair groups (every 4 rows).
#let margin-w = 105pt
#let margin-grid = grid(
  rows: (sz,) * 16,
  ..range(16).map(i => {
    let top-rule = if i > 0 and calc.rem(i, 4) == 0 { rule-major } else { none }
    let grp = colour-grp(i)
    let lb  = calc.rem(i, 4) * 16
    let le  = lb + 15
    let swatch = box(
      width: 9pt, height: 9pt,
      fill: if grp == 0 { teal-light } else { grey },
      stroke: 0.3pt + dimmed,
    )
    rect(
      width: margin-w, height: sz, fill: white,
      stroke: (top: top-rule, bottom: none, left: none, right: none),
      align(right + horizon,
        grid(
          columns: (24pt, 4pt, 60pt, 4pt, 9pt),
          align(right + horizon, text(size: 9pt, style: "italic", fill: dimmed)[i=#i]),
          [],
          align(right + horizon, text(size: 9pt, fill: dimmed)[lanes #lb–#le]),
          [],
          align(horizon, swatch),
        )
      )
    )
  })
)

// ── Final layout ──────────────────────────────────────────────────────────
#pad(
  top: 14pt, bottom: 14pt, left: 10pt, right: 14pt,
  stack(
    dir: ttb, spacing: 6pt,
    grid(
      columns: (margin-w, 4pt, auto),
      align(right + bottom, text(size: 9pt, style: "italic", fill: dimmed)[_i_ ╲ _j_]),
      [],
      col-header,
    ),
    grid(
      columns: (margin-w, 4pt, auto),
      margin-grid,
      [],
      data-grid,
    ),
  )
)
