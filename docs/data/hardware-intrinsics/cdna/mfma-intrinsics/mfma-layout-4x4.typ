// 4×4 MFMA C/D accumulator layout (CDNA1)
//
// The wavefront holds 16 independent 4×4 output tiles simultaneously.
// Each group of 4 consecutive lanes (lanes 4b … 4b+3) holds one tile.
//
// Layout: rows = accVGPR index (= tile row i, 0–3)
//         cols = lane 0–63 (16 groups of 4, one per block)
//
// Formulas
//   lane    = 4b + j      (j = column within the tile, 0–3)
//   accVGPR = i           (= tile row, 0–3)

// ── Page ──────────────────────────────────────────────────────────────────
#set page(width: auto, height: auto, margin: 0pt)
#set text(font: "New Computer Modern", fill: rgb("#1e1e1e"), size: 7pt)

// ── Palette ───────────────────────────────────────────────────────────────
#let teal-light   = rgb("#cceef5")   // even blocks — light teal tint
#let grey   = rgb("#bfbfbf")   // odd  blocks — grey tint
#let white  = rgb("#ffffff")
#let ink    = rgb("#1e1e1e")
#let dimmed = rgb("#444444")
#let rule-block = 0.8pt + rgb("#777777")   // between blocks (every 4 cols)

// ── Cell geometry ─────────────────────────────────────────────────────────
#let sz  = 18pt
#let gap = 0pt

// ── Derived quantities ────────────────────────────────────────────────────
// lane (= column of the full grid) → block index
#let block-of(lane) = calc.floor(lane / 4)

// ── Single cell: row = accVGPR index i (0–3), col = lane (0–63) ──────────
#let mfma-cell(i, lane) = {
  let b  = block-of(lane)
  let bg = if calc.rem(b, 2) == 0 { teal-light } else { grey }
  // Vertical rule at the left edge of each block (except block 0)
  let left-rule = if lane > 0 and calc.rem(lane, 4) == 0 { rule-block } else { none }
  rect(
    width: sz, height: sz, fill: bg,
    stroke: (left: left-rule, top: none, bottom: none, right: none),
    align(center + horizon,
      text(size: 9pt, weight: "bold", fill: ink)[#i]
    )
  )
}

// ── Data grid (4 rows × 64 cols) ──────────────────────────────────────────
#let data-grid = grid(
  columns: (sz,) * 64,
  rows:    (sz,) * 4,
  column-gutter: gap,
  row-gutter:    gap,
  ..range(4).map(i => range(64).map(lane => mfma-cell(i, lane))).flatten()
)

// ── Column header: block number above each 4-lane group ──────────────────
// Show block index centred over each 4-column group
#let block-header = grid(
  columns: (sz,) * 64,
  column-gutter: gap,
  ..range(64).map(lane => {
    let b = block-of(lane)
    // Only label the first lane of each block
    if calc.rem(lane, 4) == 0 {
      // span 4 cells by using a box of width 4*sz
      box(
        width: 4 * sz,
        align(center, text(size: 9pt, fill: dimmed)[b=#b])
      )
    } else {
      // empty cells already consumed by the box above — emit nothing
      box(width: 0pt)
    }
  })
)

// ── Left margin: row index label (= accVGPR = tile row i) ─────────────────
#let margin-w = 80pt
#let margin-grid = grid(
  rows: (sz,) * 4,
  ..range(4).map(i => {
    rect(
      width: margin-w, height: sz, fill: white,
      stroke: none,
      align(right + horizon,
        text(size: 9pt, fill: dimmed)[
          accVGPR #i #h(4pt)
        ]
      )
    )
  })
)

// ── Lane axis label row (below block header, above data) ──────────────────
// Print lane number every 4 columns
#let lane-header = grid(
  columns: (sz,) * 64,
  column-gutter: gap,
  ..range(64).map(lane => {
    if calc.rem(lane, 4) == 0 {
      box(
        width: 4 * sz,
        align(center, text(size: 9pt, fill: dimmed)[#lane])
      )
    } else {
      box(width: 0pt)
    }
  })
)

// ── Final layout ──────────────────────────────────────────────────────────
#pad(
  top: 14pt, bottom: 14pt, left: 10pt, right: 14pt,
  stack(
    dir: ttb, spacing: 4pt,
    // Block index row
    grid(
      columns: (margin-w, 4pt, auto),
      align(right + bottom, text(size: 9pt, style: "italic", fill: dimmed)[block]),
      [],
      block-header,
    ),
    // Lane index row
    grid(
      columns: (margin-w, 4pt, auto),
      align(right + bottom, text(size: 9pt, style: "italic", fill: dimmed)[lane]),
      [],
      lane-header,
    ),
    // Data
    grid(
      columns: (margin-w, 4pt, auto),
      margin-grid,
      [],
      data-grid,
    ),
  )
)
