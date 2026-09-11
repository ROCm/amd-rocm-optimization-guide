// Partial histogram: two-pass approach
//
// Pass 1: Each block writes its local histogram to a private slice
//         of a temporary buffer (no atomics needed).
// Pass 2: A reduction kernel sums across all block slices for each bin.

// -- Page ------------------------------------------------------------------
#set page(width: auto, height: auto, margin: 14pt, fill: rgb("#262626"))
#set text(font: "New Computer Modern", fill: rgb("#ffffff"), size: 9pt)

// -- Palette (matches reduction / matrix-multiply diagrams) ---------------
#let bg-dark   = rgb("#262626")
#let accent    = rgb("#c23555")
#let grey      = rgb("#5e5b61")
#let dark-red  = rgb("#4f1623")
#let mauve     = rgb("#523e43")
#let dark-grey = rgb("#363538")
#let white     = rgb("#ffffff")
#let dimmed    = rgb("#9e9e9e")

// -- Geometry --------------------------------------------------------------
#let cell-w = 48pt
#let cell-h = 22pt
#let label-w = 70pt

// Block row colors - cycle through palette shades
#let block-colors = (dark-red, mauve, grey, dark-grey)

#let data-cell(body, bg: grey) = {
  rect(
    width: cell-w, height: cell-h, fill: bg,
    stroke: 0.5pt + rgb("#444444"),
    align(center + horizon, text(size: 8pt, fill: white)[#body])
  )
}

#let row-label(body) = {
  rect(
    width: label-w, height: cell-h, fill: bg-dark,
    stroke: none,
    align(right + horizon, text(size: 8pt, fill: dimmed)[#body #h(4pt)])
  )
}

// -- Parameters ------------------------------------------------------------
#let num-blocks = 4
#let num-bins = 5

// -- Pass 1: Partial histogram buffer layout ------------------------------
#align(center,
  stack(dir: ttb, spacing: 14pt,
    text(weight: "bold", size: 11pt, fill: white)[Pass 1: Each block stores its histogram (plain writes, no atomics)],

    // Column header: bin indices
    grid(
      columns: (label-w, ..(cell-w,) * num-bins),
      rows: (cell-h,),
      rect(width: label-w, height: cell-h, stroke: none, fill: bg-dark,
        align(right + horizon, text(size: 8pt, fill: dimmed, weight: "bold")[bin #h(4pt)])),
      ..range(num-bins).map(b =>
        rect(width: cell-w, height: cell-h, stroke: none, fill: bg-dark,
          align(center + horizon, text(size: 8pt, fill: dimmed)[#b]))
      ),
    ),

    // Rows: one per block
    grid(
      columns: (label-w, ..(cell-w,) * num-bins),
      rows: (cell-h,) * num-blocks,
      row-gutter: 2pt,
      ..range(num-blocks).map(blk =>
        (
          row-label[Block #blk],
          ..range(num-bins).map(bin =>
            data-cell(bg: block-colors.at(blk))[H#sub[#blk]\[#bin\]]
          ),
        )
      ).flatten(),
    ),

    v(4pt),
    text(size: 8pt, fill: dimmed, weight: "bold")[
      partial_histogram\[blockIdx.x × num_bins + bin\]
    ],

    // -- Arrow ----------------------------------------------------------
    v(4pt),
    text(size: 16pt, fill: dimmed)[↓],
    text(size: 8pt, fill: dimmed, weight: "bold")[kernel launch boundary],
    v(4pt),

    // -- Pass 2: Reduction across block slices ------------------------
    text(weight: "bold", size: 11pt, fill: white)[Pass 2: Sum across all block slices per bin],

    // Show the reduction for each bin
    grid(
      columns: (label-w, ..(cell-w,) * num-bins),
      rows: (cell-h,),

      row-label[Output],
      ..range(num-bins).map(bin =>
        data-cell(bg: accent)[
          #text(size: 7pt, fill: white)[
            #range(num-blocks).map(blk => [H#sub[#blk]]).join([+])
          ]
        ]
      ),
    ),

    v(2pt),
    grid(
      columns: (label-w, ..(cell-w,) * num-bins),
      rows: (cell-h,),

      rect(width: label-w, height: cell-h, stroke: none, fill: bg-dark,
        align(right + horizon, text(size: 8pt, fill: dimmed, weight: "bold")[bin #h(4pt)])),
      ..range(num-bins).map(b =>
        rect(width: cell-w, height: cell-h, stroke: none, fill: bg-dark,
          align(center + horizon, text(size: 8pt, fill: dimmed)[#b]))
      ),
    ),

    v(2pt),
    text(size: 8pt, fill: dimmed, weight: "bold")[
      One block per bin; tree reduction across num_blocks partial results
    ],
  )
)
