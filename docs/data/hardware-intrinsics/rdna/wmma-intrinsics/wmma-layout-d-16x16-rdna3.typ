// 16×16 WMMA Accumulator layout (RDNA 3/3.5, wave32)
//
// Split into two 8×16 grids — one per lane group — so labels are clean.
//
// Visual encoding
//   Rose grid → even rows (0,2,4,…,14) held by lanes  0–15
//   Grey grid → odd  rows (1,3,5,…,15) held by lanes 16–31
//   Cell value → VGPR index g (0–7)
//   Column j   → matrix column = lane % 16
//
// Formulas (lane L, VGPR g)
//   row = g * 2 + floor(L / 16)
//   col = L % 16
//
// Inverse (row i, col j)
//   lane = (i % 2) * 16 + j
//   VGPR = floor(i / 2)

// ── Page ──────────────────────────────────────────────────────────────────
#set page(width: auto, height: auto, margin: 0pt)
#set text(font: "New Computer Modern", fill: rgb("#1e1e1e"), size: 7pt)

// ── Palette ───────────────────────────────────────────────────────────────
#let rose-light = rgb("#f5cccd")   // lanes  0–15 (even rows)
#let grey       = rgb("#bfbfbf")   // lanes 16–31 (odd  rows)
#let white      = rgb("#ffffff")
#let ink        = rgb("#1e1e1e")
#let dimmed     = rgb("#777777")

// ── Cell geometry ─────────────────────────────────────────────────────────
#let sz  = 14pt
#let gap = 0pt

// ── Single cell (uniform color) ──────────────────────────────────────────
#let d-cell(vgpr, bg) = {
  rect(
    width: sz, height: sz, fill: bg,
    stroke: none,
    align(center + horizon,
      text(size: 6pt, weight: "bold", fill: ink)[#vgpr]
    )
  )
}

// ── Build an 8-row × 16-col grid for one lane group ─────────────────────
#let half-grid(bg, rows-list) = grid(
  columns: (sz,) * 16,
  rows:    (sz,) * 8,
  column-gutter: gap,
  row-gutter:    gap,
  ..rows-list.map(i => {
    let g = calc.floor(i / 2)
    range(16).map(j => d-cell(g, bg))
  }).flatten()
)

// ── Column header (j = 0 … 15) ──────────────────────────────────────────
#let col-header = grid(
  columns: (sz,) * 16,
  column-gutter: gap,
  ..range(16).map(j =>
    align(center, text(size: 5pt, fill: dimmed)[#j])
  )
)

// ── Left margin for one half-grid ────────────────────────────────────────
#let margin-w = 105pt
#let half-margin(rows-list, bg, lane-base) = {
  let lane-last = lane-base + 15
  let swatch = box(
    width: 6pt, height: 6pt,
    fill: bg,
    stroke: 0.3pt + dimmed,
  )
  grid(
    rows: (sz,) * 8,
    ..rows-list.map(i => {
      box(
        width: margin-w, height: sz,
        align(right + horizon,
          grid(
            columns: (24pt, 4pt, 50pt, 4pt, 9pt),
            align(right + horizon, text(size: 5.5pt, style: "italic", fill: dimmed)[i=#i]),
            [],
            align(right + horizon, text(size: 5.5pt, fill: dimmed)[lanes #lane-base–#lane-last]),
            [],
            align(center + horizon, swatch),
          )
        )
      )
    })
  )
}

// ── Caption ──────────────────────────────────────────────────────────────
#let caption-text = [
  #set text(size: 6pt, fill: dimmed)
  Accumulator for all 16×16 WMMA intrinsics on RDNA3/3.5. \
  Cell value = VGPR index _g_ (0–7). Column _j_ = matrix column = lane % 16. \
  Even rows → lanes 0–15; odd rows → lanes 16–31 (matrix replication). \
  Formulas: lane = (_i_ mod 2) · 16 + _j_ ;  VGPR = ⌊_i_/2⌋.
]

// ── Title ────────────────────────────────────────────────────────────────
#let title = text(size: 8pt, weight: "bold")[
  16×16 WMMA accumulator layout — wave32 (RDNA3)
]

// ── Row sets ─────────────────────────────────────────────────────────────
#let even-rows = (0, 2, 4, 6, 8, 10, 12, 14)
#let odd-rows  = (1, 3, 5, 7, 9, 11, 13, 15)

// ── Final layout ─────────────────────────────────────────────────────────
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
      half-margin(even-rows, rose-light, 0),
      [],
      half-grid(rose-light, even-rows),
    ),
    v(4pt),
    grid(
      columns: (margin-w, 4pt, auto),
      half-margin(odd-rows, grey, 16),
      [],
      half-grid(grey, odd-rows),
    ),
    v(2pt),
    grid(
      columns: (margin-w, 4pt, auto),
      [], [],
      caption-text,
    ),
  )
)
