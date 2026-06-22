// SMFMAC 32×32×16 — A (sparse, 2:4) operand register layout
//
//   rows = i (0..31), cols = compressed k (0..15)
//   VGPR = ⌊k/4⌋ mod 2 → v0 or v1
//   lane  = i + 32·⌊k/8⌋

#set page(width: auto, height: auto, margin: 0pt)
#set text(font: "New Computer Modern", size: 7pt, fill: rgb("#1e1e1e"))

#let teal-light = rgb("#cceef5")
#let cool-grey  = rgb("#bfbfbf")
#let ink        = rgb("#1e1e1e")
#let dimmed     = rgb("#777777")

#let sz       = 14pt
#let gap      = 0pt
#let rule-maj = 0.8pt + rgb("#777777")
#let rule-min = 0.3pt + rgb("#bfbfbf")

#let group-fill(g) = if calc.rem(g, 2) == 0 { teal-light } else { cool-grey }

#let cell(label, fill) = box(
  width: sz, height: sz, fill: fill,
  stroke: rule-min,
  align(center + horizon, text(size: 5.5pt, label)))

#let a-cell(i, k) = {
  let v = calc.rem(calc.floor(k / 4), 2)
  let g = calc.floor(k / 8)
  cell("v" + str(v), group-fill(g))
}

#let a-data-grid = grid(
  columns: 16, rows: 32, gutter: gap,
  ..range(32).map(i =>
    range(16).map(k => a-cell(i, k))
  ).flatten()
)

#let col-hdr(n) = grid(
  columns: n, gutter: gap,
  ..range(n).map(c =>
    box(width: sz, height: sz * 0.9,
      align(center + bottom, text(size: 5pt, fill: dimmed, str(c))))
  )
)

#let row-labels(labels) = grid(
  columns: 1, gutter: gap,
  ..labels.map(lbl =>
    box(width: sz * 2.2, height: sz,
      align(right + horizon, text(size: 5pt, fill: dimmed, lbl + " ")))
  )
)

#let hrule(w) = line(length: w, stroke: rule-maj)

#let sub-fig(title, col-header, label-col, data-grid, total-width) = {
  stack(dir: ttb, spacing: 2pt,
    align(center, text(weight: "bold", title)),
    pad(left: sz * 2.2, col-header),
    hrule(total-width),
    stack(dir: ltr, label-col, data-grid),
  )
}

#let label-w = sz * 2.2
#let grid-w  = sz * 16
#let total-w = label-w + grid-w

#let fig-a = sub-fig(
  "A (sparse, 2:4)",
  col-hdr(16),
  row-labels(range(32).map(i => "i=" + str(i))),
  a-data-grid,
  total-w,
)

#let caption-text = [
  *A* (sparse): 2 VGPRs (v0–v1), 2:4 structured sparsity along the k-dimension;
  VGPR = ⌊k/4⌋ mod 2, lane = i + 32·⌊k/8⌋.
  Colour groups of 8 k-columns alternate teal/grey.
]

#pad(top: 14pt, rest: 14pt,
  stack(dir: ttb, spacing: 14pt,
    fig-a,
    block(width: total-w,
      align(left, text(size: 6pt, fill: dimmed, caption-text))),
  )
)
