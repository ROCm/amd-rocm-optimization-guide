// Race condition in histogram bin update
//
// Two threads read the same bin value before either writes back.
// The second write overwrites the first, losing one increment.

// ── Page ──────────────────────────────────────────────────────────────────
#set page(width: auto, height: auto, margin: 12pt, fill: rgb("#262626"))
#set text(font: "New Computer Modern", fill: rgb("#ffffff"), size: 9pt)

// ── Palette (matches reduction / matrix-multiply diagrams) ───────────────
#let bg-dark    = rgb("#262626")
#let accent     = rgb("#c23555")
#let grey       = rgb("#5e5b61")
#let dark-red   = rgb("#4f1623")
#let mauve      = rgb("#523e43")
#let white      = rgb("#ffffff")
#let dimmed     = rgb("#9e9e9e")

// ── Geometry ──────────────────────────────────────────────────────────────
#let col-w = 140pt
#let row-h = 28pt
#let time-w = 28pt
#let mem-w = 140pt

#let step-cell(body, bg: grey) = {
  rect(
    width: col-w, height: row-h, fill: bg,
    stroke: 0.5pt + rgb("#444444"),
    radius: 3pt,
    align(center + horizon, text(fill: white)[#body])
  )
}

#let time-label(t) = {
  rect(
    width: time-w, height: row-h, fill: bg-dark,
    stroke: none,
    align(center + horizon, text(size: 8pt, fill: dimmed, weight: "bold")[#t])
  )
}

#let mem-cell(body, bg: grey) = {
  rect(
    width: mem-w, height: row-h, fill: bg,
    stroke: 0.5pt + rgb("#444444"),
    radius: 3pt,
    align(center + horizon, text(fill: white)[#body])
  )
}

// ── Layout ────────────────────────────────────────────────────────────────
#grid(
  columns: (time-w, col-w, 8pt, col-w, 12pt, mem-w),
  rows: (row-h,) * 6,
  row-gutter: 6pt,
  column-gutter: 0pt,

  // Header row
  rect(width: time-w, height: row-h, stroke: none, fill: bg-dark, []),
  rect(width: col-w, height: row-h, stroke: none, fill: bg-dark,
    align(center + horizon, text(weight: "bold", fill: white)[Thread A])),
  [],
  rect(width: col-w, height: row-h, stroke: none, fill: bg-dark,
    align(center + horizon, text(weight: "bold", fill: white)[Thread B])),
  [],
  rect(width: mem-w, height: row-h, stroke: none, fill: bg-dark,
    align(center + horizon, text(weight: "bold", fill: white)[histogram\[bin\]])),

  // t1: Both threads read
  time-label[t1],
  step-cell(bg: mauve)[read: val = 5],
  [],
  step-cell(bg: dark-red)[read: val = 5],
  [],
  mem-cell(bg: grey)[5],

  // t2: Both threads compute
  time-label[t2],
  step-cell(bg: mauve)[compute: 5 + 1 = 6],
  [],
  step-cell(bg: dark-red)[compute: 5 + 1 = 6],
  [],
  mem-cell(bg: grey)[5],

  // t3: Thread A writes
  time-label[t3],
  step-cell(bg: mauve)[write: 6],
  [],
  step-cell(bg: rgb("#363538"))[ ],
  [],
  mem-cell(bg: grey)[6],

  // t4: Thread B writes (overwrites)
  time-label[t4],
  step-cell(bg: rgb("#363538"))[ ],
  [],
  step-cell(bg: dark-red)[write: 6],
  [],
  mem-cell(bg: accent)[6 #text(fill: white, size: 8pt)[ (expected 7)]],

  // Result annotation
  time-label[],
  grid.cell(colspan: 5,
    align(center + horizon,
      text(fill: white, size: 9pt, weight: "bold")[
        One increment lost: both threads read 5, both write 6
      ]
    )
  ),
)
