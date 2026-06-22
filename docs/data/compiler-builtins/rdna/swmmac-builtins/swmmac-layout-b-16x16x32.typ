// 16×16×32 SWMMAC srcB (dense) fragment layout -- RDNA 4, wave32
//
// The B matrix is K_dense=32 rows × N=16 columns.
// Each lane covers one N-column:  col = lane % 16.
//
// The 16 dense FP16 elements (v16fp16, 8 VGPRs × 2 FP16) are loaded
// from NON-CONTIGUOUS K-rows:
//
//   Lane group 0 (lanes  0–15, rose):
//     VGPRs 0–3 → K rows  0– 7    VGPRs 4–7 → K rows 16–23
//
//   Lane group 1 (lanes 16–31, grey):
//     VGPRs 0–3 → K rows  8–15    VGPRs 4–7 → K rows 24–31
//
// Within each 8-row block, two consecutive K-rows share a VGPR:
//   K rows 0,1 → VGPR 0 ;  K rows 2,3 → VGPR 1 ;  etc.
//
// Cell value = VGPR index (0–7); row color = lane group.

// ── Page ──────────────────────────────────────────────────────────────────
#set page(width: auto, height: auto, margin: 0pt)
#set text(font: "New Computer Modern", fill: rgb("#1e1e1e"), size: 7pt)

// ── Palette ───────────────────────────────────────────────────────────────
#let rose-light = rgb("#f5cccd")   // lane group 0 (lanes  0–15)
#let grey       = rgb("#bfbfbf")   // lane group 1 (lanes 16–31)
#let white      = rgb("#ffffff")
#let ink        = rgb("#1e1e1e")
#let dimmed     = rgb("#777777")
#let rule-major = 0.8pt + rgb("#777777")   // between K-row bands

// ── Cell geometry ─────────────────────────────────────────────────────────
#let sz  = 12pt   // slightly smaller to keep a 32-row grid compact
#let gap = 0pt

// ── Row classification functions ──────────────────────────────────────────
// Lane group that covers K-row k:
//   group 0 (rose): k ∈ { 0– 7, 16–23}
//   group 1 (grey): k ∈ { 8–15, 24–31}
#let b-grp(k) = if k < 8 or (k >= 16 and k < 24) { 0 } else { 1 }

// VGPR index (0–7) for K-row k:
//   k  0– 7: VGPR = floor(k / 2)           → 0–3 (grp 0 lower block)
//   k  8–15: VGPR = floor((k-8) / 2)       → 0–3 (grp 1 lower block)
//   k 16–23: VGPR = 4 + floor((k-16) / 2)  → 4–7 (grp 0 upper block)
//   k 24–31: VGPR = 4 + floor((k-24) / 2)  → 4–7 (grp 1 upper block)
#let b-vgpr(k) = {
  if k < 8       { calc.floor(k / 2) }
  else if k < 16 { calc.floor((k - 8)  / 2) }
  else if k < 24 { 4 + calc.floor((k - 16) / 2) }
  else           { 4 + calc.floor((k - 24) / 2) }
}

// Major rule positions: at the start of each 8-row band (k = 8, 16, 24)
#let b-top-rule(k) = if k == 8 or k == 16 or k == 24 { rule-major } else { none }

// ── Single cell: K-row k, N-column j ──────────────────────────────────────
#let b-cell(k, j) = {
  let bg = if b-grp(k) == 0 { rose-light } else { grey }
  rect(
    width: sz, height: sz, fill: bg,
    stroke: (top: b-top-rule(k), bottom: none, left: none, right: none),
    align(center + horizon,
      text(size: 5.5pt, weight: "bold", fill: ink)[#b-vgpr(k)]
    )
  )
}

// ── Data grid (32 K-rows × 16 N-cols) ────────────────────────────────────
#let data-grid = grid(
  columns: (sz,) * 16,
  rows:    (sz,) * 32,
  column-gutter: gap,
  row-gutter:    gap,
  ..range(32).map(k => range(16).map(j => b-cell(k, j))).flatten()
)

// ── Column header: N index (0–15) ────────────────────────────────────────
#let col-header = grid(
  columns: (sz,) * 16,
  column-gutter: gap,
  ..range(16).map(j =>
    align(center, text(size: 5pt, fill: dimmed)[#j])
  )
)

// ── Left margin: one label per 8-row K band ───────────────────────────────
#let margin-w = 90pt
#let margin-grid = grid(
  rows: (sz,) * 32,
  ..range(32).map(k => {
    let top-rule = b-top-rule(k)
    let label = if calc.rem(k, 8) == 0 {
      let g     = b-grp(k)
      let klast = k + 7
      let lbase = g * 16
      let llast = lbase + 15
      let vbase = if k < 16 { 0 } else { 4 }
      let vlast = vbase + 3
      let swatch = box(
        width: 6pt, height: 6pt,
        fill: if g == 0 { rose-light } else { grey },
        stroke: 0.3pt + dimmed,
      )
      box(
        height: sz,
        align(right + horizon,
          grid(
            columns: (auto, 6pt, auto, 4pt, auto),
            align(right + horizon, text(size: 5pt, fill: dimmed)[k=#k–#klast, V#vbase–#vlast]),
            [],
            align(right + horizon, text(size: 5pt, fill: dimmed)[lanes #lbase–#llast]),
            [],
            align(horizon, swatch),
          )
        )
      )
    } else { box(height: sz) }
    rect(
      width: margin-w, height: sz, fill: white,
      stroke: (top: top-rule, bottom: none, left: none, right: none),
      label
    )
  })
)

// ── Caption ───────────────────────────────────────────────────────────────
#let caption-text = [
  #set text(size: 6pt, fill: dimmed)
  srcB dense fragment for 16×16×32 SWMMAC (FP16/BF16 inputs), RDNA4. \
  Cell value = VGPR index (0–7); each VGPR holds 2 packed FP16 values. \
  Teal rows = lane group 0 (lanes 0–15); grey = lane group 1 (lanes 16–31). \
  Both groups cover all 16 columns (col = lane % 16). Row _k_ = dense-K index.
]

// ── Title ─────────────────────────────────────────────────────────────────
#let title = text(size: 8pt, weight: "bold")[
  16×16×32 SWMMAC srcB layout — wave32 (dense K, FP16)
]

// ── Final layout ──────────────────────────────────────────────────────────
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
      align(right + bottom, text(size: 5.5pt, style: "italic", fill: dimmed)[_k_ ╲ _j_]),
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
