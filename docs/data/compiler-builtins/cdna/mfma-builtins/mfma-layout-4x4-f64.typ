// V_MFMA_F64_4X4X4F64 - C/D accumulator layout (CDNA2)
//
// Visual encoding
//   Layout: 4 rows (= matrix rows i 0–3) × 64 columns (= lanes 0–63)
//   Each group of 4 consecutive lanes holds one complete 4×4 output tile
//   for one block (b = ⌊lane/4⌋ mod 4, row group = ⌊lane/16⌋).
//   Cell number → accVGPR pair index (always 0: physical regs v[1:0])
//   Colour      → alternates per block (even = teal, odd = grey)
//
// The 4×4 FP64 instruction uses 4 blocks.  With 4 rows and 64 lanes:
//   rows 0–3 → all 64 lanes, 16 per row group × 4 blocks per row group.
//   Within each 16-lane row group: 4 sub-groups of 4 lanes = 4 blocks.
//
// Formulas
//   lane = 16·i + 4·b + j
//   accVGPR pair = 0  (single pair v[1:0] per lane)

// -- Page ------------------------------------------------------------------
#set page(width: auto, height: auto, margin: 0pt)
#set text(font: "New Computer Modern", fill: rgb("#1e1e1e"), size: 7pt)

// -- Palette ---------------------------------------------------------------
#let teal-light   = rgb("#cceef5")   // even blocks - light teal tint
#let grey         = rgb("#bfbfbf")   // odd  blocks - grey tint
#let white        = rgb("#ffffff")
#let ink          = rgb("#1e1e1e")
#let dimmed       = rgb("#444444")
#let rule-block   = 0.8pt + rgb("#777777")   // between blocks (every 4 cols)
#let rule-group   = 1.2pt + rgb("#444444")   // between 16-lane row groups (every 16 cols)

// -- Cell geometry ---------------------------------------------------------
#let sz  = 18pt
#let gap = 0pt

// -- Derived quantities ----------------------------------------------------
// Given a lane (0–63), the block index within its 16-lane row group:
//   block-within-group = ⌊(lane mod 16) / 4⌋
// Colour by this local block index (so the teal/grey pattern resets every 16 lanes).
#let local-block(lane) = calc.floor(calc.rem(lane, 16) / 4)

// -- Single cell: row = matrix row i (0–3), col = lane (0–63) -------------
#let mfma-cell(i, lane) = {
  let lb = local-block(lane)
  let bg = if calc.rem(lb, 2) == 0 { teal-light } else { grey }
  // Vertical rule: thin between blocks (every 4 lanes), thicker between row groups (every 16)
  let left-rule = if lane > 0 and calc.rem(lane, 16) == 0 {
    rule-group
  } else if lane > 0 and calc.rem(lane, 4) == 0 {
    rule-block
  } else {
    none
  }
  rect(
    width: sz, height: sz, fill: bg,
    stroke: (left: left-rule, top: none, bottom: none, right: none),
    align(center + horizon,
      text(size: 9pt, weight: "bold", fill: ink)[0]
    )
  )
}

// -- Data grid (4 rows × 64 cols) ------------------------------------------
#let data-grid = grid(
  columns: (sz,) * 64,
  rows:    (sz,) * 4,
  column-gutter: gap,
  row-gutter:    gap,
  ..range(4).map(i => range(64).map(lane => mfma-cell(i, lane))).flatten()
)

// -- Column header: block number above each 4-lane group ------------------
#let block-header = grid(
  columns: (sz,) * 64,
  column-gutter: gap,
  ..range(64).map(lane => {
    let b = local-block(lane)
    if calc.rem(lane, 4) == 0 {
      box(
        width: 4 * sz,
        align(center, text(size: 9pt, fill: dimmed)[b=#b])
      )
    } else {
      box(width: 0pt)
    }
  })
)

// -- Row-group header: row group above each 16-lane group -----------------
#let group-header = grid(
  columns: (sz,) * 64,
  column-gutter: gap,
  ..range(64).map(lane => {
    let g = calc.floor(lane / 16)
    if calc.rem(lane, 16) == 0 {
      box(
        width: 16 * sz,
        align(center, text(size: 9pt, fill: dimmed)[row #g, lanes #(g*16)–#(g*16+15)])
      )
    } else {
      box(width: 0pt)
    }
  })
)

// -- Lane axis label row ----------------------------------------------------
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

// -- Left margin: row index label ------------------------------------------
#let margin-w = 60pt
#let margin-grid = grid(
  rows: (sz,) * 4,
  ..range(4).map(i => {
    rect(
      width: margin-w, height: sz, fill: white,
      stroke: none,
      align(right + horizon,
        text(size: 9pt, fill: dimmed)[
          row _i_=#i #h(4pt)
        ]
      )
    )
  })
)

// -- Final layout ----------------------------------------------------------
#pad(
  top: 14pt, bottom: 14pt, left: 10pt, right: 14pt,
  stack(
    dir: ttb, spacing: 4pt,
    // Row-group header
    grid(
      columns: (margin-w, 4pt, auto),
      align(right + bottom, text(size: 9pt, style: "italic", fill: dimmed)[row group]),
      [],
      group-header,
    ),
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
