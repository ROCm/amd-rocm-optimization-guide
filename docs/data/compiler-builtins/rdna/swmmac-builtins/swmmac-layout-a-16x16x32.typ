// 16×16×32 SWMMAC srcA (sparse) fragment layout -- RDNA 4, wave32
//
// The A matrix after 2:4 pruning has 16 rows × 16 compressed-K columns.
// Each lane covers one matrix row:  row = lane % 16.
//
// The 8 compressed FP16 elements (v8fp16, 4 VGPRs × 2 FP16) are loaded
// from NON-CONTIGUOUS compressed-K positions:
//
//   Lane group 0 (lanes  0–15, rose):
//     VGPR 0 → compressed K {0, 1}    VGPR 2 → compressed K { 8,  9}
//     VGPR 1 → compressed K {2, 3}    VGPR 3 → compressed K {10, 11}
//
//   Lane group 1 (lanes 16–31, grey):
//     VGPR 0 → compressed K {4, 5}    VGPR 2 → compressed K {12, 13}
//     VGPR 1 → compressed K {6, 7}    VGPR 3 → compressed K {14, 15}
//
// Cell value = VGPR index (0–3); column color = lane group.

// -- Page ------------------------------------------------------------------
#set page(width: auto, height: auto, margin: 0pt)
#set text(font: "New Computer Modern", fill: rgb("#1e1e1e"), size: 7pt)

// -- Palette ---------------------------------------------------------------
#let rose-light = rgb("#f5cccd")   // lane group 0 (lanes  0–15)
#let grey       = rgb("#bfbfbf")   // lane group 1 (lanes 16–31)
#let white      = rgb("#ffffff")
#let ink        = rgb("#1e1e1e")
#let dimmed     = rgb("#777777")
#let rule-major = 0.8pt + rgb("#777777")   // between compressed-K band groups

// -- Cell geometry ---------------------------------------------------------
#let sz  = 14pt
#let gap = 0pt

// -- Column classification functions ---------------------------------------
// Lane group that owns compressed-K column c (0–15):
//   group 0 (rose): c ∈ {0-3, 8-11}   group 1 (grey): c ∈ {4-7, 12-15}
#let a-grp(c) = if c < 4 or (c >= 8 and c < 12) { 0 } else { 1 }

// VGPR index (0–3) for compressed-K column c:
//   grp 0: cols 0,1→V0  cols 2,3→V1  cols  8, 9→V2  cols 10,11→V3
//   grp 1: cols 4,5→V0  cols 6,7→V1  cols 12,13→V2  cols 14,15→V3
#let a-vgpr(c) = {
  if c < 4       { calc.floor(c / 2) }
  else if c < 8  { calc.floor((c - 4)  / 2) }
  else if c < 12 { 2 + calc.floor((c - 8)  / 2) }
  else           { 2 + calc.floor((c - 12) / 2) }
}

// -- Single cell: row i, compressed-K column c -----------------------------
#let a-cell(i, c) = {
  let bg  = if a-grp(c) == 0 { rose-light } else { grey }
  // Vertical major rule at the left edge of each 4-column band (except col 0)
  let lft = if c == 4 or c == 8 or c == 12 { rule-major } else { none }
  rect(
    width: sz, height: sz, fill: bg,
    stroke: (left: lft, top: none, bottom: none, right: none),
    align(center + horizon,
      text(size: 6pt, weight: "bold", fill: ink)[#a-vgpr(c)]
    )
  )
}

// -- Data grid (16 rows × 16 compressed-K cols) ----------------------------
#let data-grid = grid(
  columns: (sz,) * 16,
  rows:    (sz,) * 16,
  column-gutter: gap,
  row-gutter:    gap,
  ..range(16).map(i => range(16).map(c => a-cell(i, c))).flatten()
)

// -- Band header: lane-group label spanning each 4-column group ------------
// Uses the box(width: 4*sz) / box(width: 0pt) trick to span cells.
#let band-header = grid(
  columns: (sz,) * 16,
  column-gutter: gap,
  ..range(16).map(c => {
    let g   = a-grp(c)
    let bg  = if g == 0 { rose-light } else { grey }
    let lft = if c == 4 or c == 8 or c == 12 { rule-major } else { none }
    if c == 0 or c == 4 or c == 8 or c == 12 {
      // Emit a box that spans the 4-column band
      box(
        width: 4 * sz,
        rect(
          width: 4 * sz, height: 9pt, fill: bg,
          stroke: (left: lft, top: none, bottom: 0.3pt + dimmed, right: none),
          align(center + horizon,
            text(size: 4.5pt, fill: ink)[lanes #if g == 0 { [0–15] } else { [16–31] }]
          )
        )
      )
    } else {
      box(width: 0pt)
    }
  })
)

// -- Column header: compressed-K index (0–15) with left-edge major rules ---
#let col-header = grid(
  columns: (sz,) * 16,
  column-gutter: gap,
  ..range(16).map(c => {
    let lft = if c == 4 or c == 8 or c == 12 { rule-major } else { none }
    rect(
      width: sz, height: 10pt, fill: white,
      stroke: (left: lft, top: none, bottom: none, right: none),
      align(center + horizon, text(size: 5pt, fill: dimmed)[#c])
    )
  })
)

// -- Left margin: one label per 4 rows -------------------------------------
#let margin-w = 48pt
#let margin-grid = grid(
  rows: (sz,) * 16,
  ..range(16).map(i => {
    let label = if calc.rem(i, 4) == 0 {
      box(height: sz,
        align(right + horizon,
          text(size: 5.5pt, style: "italic", fill: dimmed)[i=#i]
        )
      )
    } else { box(height: sz) }
    rect(width: margin-w, height: sz, fill: white, stroke: none, label)
  })
)

// -- Caption ---------------------------------------------------------------
#let caption-text = [
  #set text(size: 6pt, fill: dimmed)
  srcA sparse fragment for 16×16×32 SWMMAC (FP16/BF16 inputs), RDNA4. \
  Cell value = VGPR index (0–3); each VGPR holds 2 packed FP16 values. \
  Teal columns = lane group 0 (lanes 0–15); grey = lane group 1 (lanes 16–31). \
  Both groups cover all 16 rows (row = lane % 16). Column _c_ = compressed-K index.
]

// -- Title -----------------------------------------------------------------
#let title = text(size: 8pt, weight: "bold")[
  16×16×32 SWMMAC srcA layout - wave32 (compressed K, FP16)
]

// -- Final layout ----------------------------------------------------------
#pad(
  top: 14pt, bottom: 14pt, left: 10pt, right: 14pt,
  stack(
    dir: ttb, spacing: 0pt,
    grid(
      columns: (margin-w, 4pt, auto),
      [], [],
      title,
    ),
    v(6pt),
    // Band header (lane group spans)
    grid(
      columns: (margin-w, 4pt, auto),
      align(right + bottom, text(size: 5.5pt, style: "italic", fill: dimmed)[lane grp]),
      [],
      band-header,
    ),
    // Compressed-K column indices
    grid(
      columns: (margin-w, 4pt, auto),
      align(right + bottom, text(size: 5.5pt, style: "italic", fill: dimmed)[_c_]),
      [],
      col-header,
    ),
    // Data
    grid(
      columns: (margin-w, 4pt, auto),
      margin-grid,
      [],
      data-grid,
    ),
    v(4pt),
    grid(
      columns: (margin-w, 4pt, auto),
      [], [],
      caption-text,
    ),
  )
)
