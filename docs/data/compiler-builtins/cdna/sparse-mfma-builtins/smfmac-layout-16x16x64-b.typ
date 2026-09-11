// SMFMAC 16×16×64 - B (dense) operand register layout
//
//   rows = k (0..63), cols = j (0..15)
//   VGPR = ⌊(k mod 16)/4⌋ → v0..v3
//   lane  = j + 16·⌊k/16⌋

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

#let b-cell(k, j) = {
  let v = calc.rem(calc.floor(k / 4), 4)
  let g = calc.floor(k / 16)
  cell("v" + str(v), group-fill(g))
}

#let b-data-grid = grid(
  columns: 16, rows: 64, gutter: gap,
  ..range(64).map(k =>
    range(16).map(j => b-cell(k, j))
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

#let fig-b = sub-fig(
  "B (dense)",
  col-hdr(16),
  row-labels(range(64).map(k => "k=" + str(k))),
  b-data-grid,
  total-w,
)

#let caption-text = [
  *B* (dense): 4 VGPRs (v0–v3), one byte per k-step;
  VGPR = ⌊(k mod 16)/4⌋, lane = j + 16·⌊k/16⌋.
  Colour groups of 16 k-rows alternate teal/grey.
]

#pad(top: 14pt, rest: 14pt,
  stack(dir: ttb, spacing: 14pt,
    fig-b,
    block(width: total-w,
      align(left, text(size: 6pt, fill: dimmed, caption-text))),
  )
)
