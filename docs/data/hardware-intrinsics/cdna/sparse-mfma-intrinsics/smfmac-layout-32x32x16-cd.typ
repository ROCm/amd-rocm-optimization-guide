// SMFMAC 32×32×16 — C/D (output/input) accumulator register layout
//
//   rows = i (0..31), cols = j (0..31)
//   VGPR = 4·⌊i/8⌋ + (i mod 4) → v0..v15
//   lane  = (32·⌊i/4⌋) mod 64 + j

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

#let d-cell(i, j) = {
  let v = 4 * calc.floor(i / 8) + calc.rem(i, 4)
  let g = calc.floor(i / 8)
  cell("v" + str(v), group-fill(g))
}

#let d-data-grid = grid(
  columns: 32, rows: 32, gutter: gap,
  ..range(32).map(i =>
    range(32).map(j => d-cell(i, j))
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
#let grid-w  = sz * 32
#let total-w = label-w + grid-w

#let fig-d = sub-fig(
  "C / D (input / output)",
  col-hdr(32),
  row-labels(range(32).map(i => "i=" + str(i))),
  d-data-grid,
  total-w,
)

#let caption-text = [
  *C/D*: 16 VGPRs (v0–v15); VGPR = 4·⌊i/8⌋ + (i mod 4),
  lane = (32·⌊i/4⌋) mod 64 + j.
  Colour groups of 8 i-rows alternate teal/grey.
]

#pad(top: 14pt, rest: 14pt,
  stack(dir: ttb, spacing: 14pt,
    fig-d,
    block(width: total-w,
      align(left, text(size: 6pt, fill: dimmed, caption-text))),
  )
)
