// K-position split for ds_read_tr* intrinsics
//
// For each intrinsic a horizontal strip of 64 cells shows which K positions
// the first call (teal) and the second call (grey) cover.
// Cells beyond the intrinsic's K range are shown as inactive.
//
// FP4 and FP6:        K = 0..63  (64 total)
// FP8 and BF8:        K = 0..63  (64 total)
// INT8, FP16, BF16:   K = 0..15  (16 total)

// ── Page ──────────────────────────────────────────────────────────────────
#set page(width: auto, height: auto, margin: 0pt)
#set text(font: "New Computer Modern", fill: rgb("#1e1e1e"), size: 7pt)

// ── Palette ───────────────────────────────────────────────────────────────
#let call1  = rgb("#cceef5")   // first call  -- light teal
#let call2  = rgb("#bfbfbf")   // second call -- grey
#let none-c = rgb("#f4f4f4")   // inactive (beyond K range)
#let white  = rgb("#ffffff")
#let ink    = rgb("#1e1e1e")
#let dimmed = rgb("#777777")

// ── Cell geometry ─────────────────────────────────────────────────────────
#let cell-w = 8pt
#let cell-h = 12pt
#let gap    = 0pt

// ── Helper: classify a K position ─────────────────────────────────────────
#let k-color(k, k-max, c1-ranges, c2-ranges) = {
  if k >= k-max {
    none-c
  } else if c1-ranges.any(r => k >= r.first() and k <= r.last()) {
    call1
  } else if c2-ranges.any(r => k >= r.first() and k <= r.last()) {
    call2
  } else {
    none-c
  }
}

// ── One 64-cell K strip ────────────────────────────────────────────────────
#let k-strip(k-max, c1-ranges, c2-ranges) = grid(
  columns: (cell-w,) * 64,
  column-gutter: gap,
  ..range(64).map(k => {
    let bg  = k-color(k, k-max, c1-ranges, c2-ranges)
    let str = if k >= k-max { 0.4pt + rgb("#cccccc") } else { 0.4pt + ink }
    rect(width: cell-w, height: cell-h, fill: bg, stroke: str)
  })
)

// ── Row label (right-aligned, fixed width) ─────────────────────────────────
#let lw = 115pt
#let row-label(name) = rect(
  width: lw, height: cell-h, fill: white, stroke: none,
  align(right + horizon,
    text(size: 5pt, fill: dimmed)[#name #h(4pt)]
  )
)

// ── K-axis tick labels (every 8 positions) ────────────────────────────────
#let k-axis = grid(
  columns: (cell-w,) * 64,
  column-gutter: gap,
  ..range(64).map(k =>
    if calc.rem(k, 8) == 0 {
      box(width: 8 * cell-w,
        align(left, text(size: 5pt, fill: dimmed)[#k])
      )
    } else {
      box(width: 0pt)
    }
  )
)

// ── Intrinsic strips ──────────────────────────────────────────────────────
#let fp4 = k-strip(64,
  ((0, 15), (32, 47)),
  ((16, 31), (48, 63)),
)
#let fp6 = k-strip(64,
  ((0, 15), (32, 47)),
  ((16, 31), (48, 63)),
)
#let fp8 = k-strip(64,
  ((0, 7), (16, 23), (32, 39), (48, 55)),
  ((8, 15), (24, 31), (40, 47), (56, 63)),
)
#let i16 = k-strip(16,
  ((0, 3), (8, 11)),
  ((4, 7), (12, 15)),
)
#let f16 = k-strip(16,
  ((0, 3), (8, 11)),
  ((4, 7), (12, 15)),
)
#let bf16 = k-strip(16,
  ((0, 3), (8, 11)),
  ((4, 7), (12, 15)),
)

// ── One labelled row ──────────────────────────────────────────────────────
#let data-row(label, strip) = grid(
  columns: (lw, auto),
  column-gutter: gap,
  row-label(label),
  strip,
)

// ── Legend ────────────────────────────────────────────────────────────────
#let swatch(col, label) = grid(
  columns: (10pt, 3pt, auto),
  column-gutter: 0pt,
  rect(width: 10pt, height: 8pt, fill: col, stroke: 0.4pt + ink),
  [],
  align(left + horizon, text(size: 5.5pt, fill: dimmed)[ #label]),
)
#let legend = grid(
  columns: (auto, 10pt, auto),
  column-gutter: 0pt,
  swatch(call1, [Call 1]),
  [],
  swatch(call2, [Call 2]),
)

// ── Title ─────────────────────────────────────────────────────────────────
#let title = text(size: 8pt, weight: "bold")[
  K-position split -- call 1 vs. call 2
]

// ── Caption ───────────────────────────────────────────────────────────────
#let caption-text = text(size: 6pt, fill: dimmed)[
  Teal cells are covered by the first call; grey cells by the second. \
  White cells indicate K positions beyond the intrinsic's total K depth. \
  FP4 and FP6 use K = 0--63 (64 total); \
  FP8 and BF8 use K = 0--63 (64 total); \
  INT8, FP16, and BF16 use K = 0--15 (16 total).
]

// ── Final layout ──────────────────────────────────────────────────────────
#pad(top: 14pt, bottom: 14pt, left: 10pt, right: 14pt,
  stack(dir: ttb, spacing: 3pt,
    title,
    v(2pt),
    grid(columns: (lw, auto), column-gutter: gap, [], k-axis),
    data-row([ds_read_tr4_b64_v2i32],  fp4),
    data-row([ds_read_tr6_b96_v3i32],  fp6),
    data-row([ds_read_tr8_b64_v2i32],  fp8),
    data-row([ds_read_tr16_b64_v4i16], i16),
    data-row([ds_read_tr16_b64_v4f16], f16),
    data-row([ds_read_tr16_b64_v4bf16], bf16),
    v(4pt),
    grid(columns: (lw, auto), column-gutter: gap, [], legend),
    v(2pt),
    caption-text,
  )
)
