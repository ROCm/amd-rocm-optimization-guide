// Shared-memory histogram: three-phase kernel structure
//
// Phase 1: Initialize per-block LDS histogram to zero
// Phase 2: Threads atomically increment LDS bins (on-chip, fast)
// Phase 3: One thread per bin merges LDS count into global histogram

// ── Page ──────────────────────────────────────────────────────────────────
#set page(width: auto, height: auto, margin: 14pt, fill: rgb("#262626"))
#set text(font: "New Computer Modern", fill: rgb("#ffffff"), size: 9pt)

// ── Palette (matches reduction / matrix-multiply diagrams) ───────────────
#let bg-dark  = rgb("#262626")
#let accent   = rgb("#c23555")
#let grey     = rgb("#5e5b61")
#let dark-red = rgb("#4f1623")
#let mauve    = rgb("#523e43")
#let dark-grey = rgb("#363538")
#let white    = rgb("#ffffff")
#let dimmed   = rgb("#9e9e9e")

// ── Geometry ──────────────────────────────────────────────────────────────
#let phase-w = 180pt
#let phase-h = 100pt
#let arrow-gap = 20pt

#let phase-box(title, body-content, bg: grey) = {
  rect(
    width: phase-w, height: phase-h, fill: bg,
    stroke: 0.8pt + rgb("#444444"),
    radius: 4pt,
    inset: 8pt,
    stack(
      dir: ttb, spacing: 6pt,
      align(center, text(weight: "bold", size: 10pt, fill: white)[#title]),
      line(length: 100%, stroke: 0.5pt + rgb("#666666")),
      align(center, body-content),
    )
  )
}

#let arrow-label(label) = {
  rect(
    width: arrow-gap, height: phase-h,
    stroke: none, fill: bg-dark,
    align(center + horizon,
      stack(dir: ttb, spacing: 2pt,
        text(size: 14pt, fill: dimmed)[→],
        text(size: 7pt, fill: dimmed, weight: "bold")[sync],
      )
    )
  )
}

// ── Layout ────────────────────────────────────────────────────────────────
#align(center,
  stack(dir: ttb, spacing: 12pt,
    // Block label
    text(size: 10pt, fill: dimmed, weight: "bold")[Per block (× num_blocks)],

    // Three phases in a row
    grid(
      columns: (phase-w, arrow-gap, phase-w, arrow-gap, phase-w),
      rows: (phase-h,),

      phase-box("Phase 1: Initialize", bg: dark-grey)[
        #text(size: 8pt, fill: white)[
          Each thread zeros one or more \
          LDS bins \
          \
          `shared_hist[tid] = 0`
        ]
      ],

      arrow-label("sync"),

      phase-box("Phase 2: Accumulate", bg: mauve)[
        #text(size: 8pt, fill: white)[
          Each thread processes \
          ITEMS_PER_THREAD elements, \
          atomically incrementing LDS bins \
          \
          `atomicAdd(&shared_hist[bin], 1)`
        ]
      ],

      arrow-label("sync"),

      phase-box("Phase 3: Merge", bg: dark-red)[
        #text(size: 8pt, fill: white)[
          One thread per bin adds the \
          LDS count to the global histogram \
          \
          `atomicAdd(&global_hist[tid], ...)`
        ]
      ],
    ),

    // Memory labels
    grid(
      columns: (phase-w, arrow-gap, phase-w, arrow-gap, phase-w),
      rows: (auto,),

      align(center, text(size: 8pt, fill: dimmed)[LDS (on-chip)]),
      [],
      align(center, text(size: 8pt, fill: dimmed)[LDS (on-chip)]),
      [],
      align(center, text(size: 8pt, fill: white)[Global memory]),
    ),
  )
)
