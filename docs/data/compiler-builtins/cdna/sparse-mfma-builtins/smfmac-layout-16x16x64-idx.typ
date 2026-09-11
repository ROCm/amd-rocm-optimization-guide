// SMFMAC 16×16×64 - Compression index operand register layout
//
//   rows = i (0..15), cols = compressed k (0..63)
//   nibble = ⌊(k mod 16)/4⌋ → n0 [3:0], n1 [7:4], n2 [11:8], n3 [15:12] of v0
//   lane   = i + 16·⌊k/16⌋

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

#let ci-cell(i, k) = {
  let n = calc.rem(calc.floor(k / 4), 4)
  let g = calc.floor(k / 16)
  cell("n" + str(n), group-fill(g))
}

#let ci-data-grid = grid(
  columns: 64, rows: 16, gutter: gap,
  ..range(16).map(i =>
    range(64).map(k => ci-cell(i, k))
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
#let grid-w  = sz * 64
#let total-w = label-w + grid-w

#let fig-ci = sub-fig(
  "Compression index (v0)",
  col-hdr(64),
  row-labels(range(16).map(i => "i=" + str(i))),
  ci-data-grid,
  total-w,
)

#let caption-text = [
  *Compression index*: 1 VGPR (v0), nibble n0–n3 packed in bits [15:0];
  nibble = ⌊(k mod 16)/4⌋; same lane assignment as A.
  Colour groups of 16 k-columns alternate teal/grey.
]

#pad(top: 14pt, rest: 14pt,
  stack(dir: ttb, spacing: 14pt,
    fig-ci,
    block(width: total-w,
      align(left, text(size: 6pt, fill: dimmed, caption-text))),
  )
)
