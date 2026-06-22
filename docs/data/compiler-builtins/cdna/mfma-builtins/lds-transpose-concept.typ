// LDS transpose concept
//
// Left panel:  B matrix stored row-major in LDS (K=4 rows × N=8 cols)
// Right panel: Per-lane VGPRs after ds_read_tr (lane n holds column n)
//
// Both panels are coloured by N-index so matching elements share a colour,
// making the column→lane correspondence easy to follow visually.

// ── Page ──────────────────────────────────────────────────────────────────
#set page(width: auto, height: auto, margin: 0pt)
#set text(font: "New Computer Modern", fill: rgb("#1e1e1e"), size: 7pt)

// ── Palette ───────────────────────────────────────────────────────────────
#let teal-light = rgb("#cceef5")
#let grey       = rgb("#bfbfbf")
#let white      = rgb("#ffffff")
#let ink        = rgb("#1e1e1e")
#let dimmed     = rgb("#777777")

// ── Parameters ────────────────────────────────────────────────────────────
#let K       = 4
#let N       = 8
#let cw      = 22pt
#let ch      = 18pt
#let gap     = 0pt
#let lm-lds  = 24pt   // row-label margin for the LDS panel
#let lm-vgpr = 28pt   // row-label margin for the VGPR panel
#let arrow-w = 70pt

// ── Colour by N-index ─────────────────────────────────────────────────────
#let n-bg(n) = if calc.rem(n, 2) == 0 { teal-light } else { grey }

// ── LDS panel ─────────────────────────────────────────────────────────────
#let lds-cell(k, n) = rect(
  width: cw, height: ch, fill: n-bg(n),
  stroke: 0.4pt + ink,
  align(center + horizon,
    text(size: 5pt, fill: ink)[B[#k,#n]]
  )
)

#let lds-row-labels = stack(dir: ttb, spacing: 0pt,
  ..range(K).map(k =>
    box(width: lm-lds, height: ch,
      align(right + horizon,
        text(size: 5pt, fill: dimmed)[k=#k #h(2pt)]
      )
    )
  )
)

#let lds-col-labels = grid(
  columns: (cw,) * N,
  column-gutter: gap,
  ..range(N).map(n =>
    align(center, text(size: 5pt, fill: dimmed)[n=#n])
  )
)

#let lds-grid = grid(
  columns: (cw,) * N,
  rows:    (ch,) * K,
  column-gutter: gap,
  row-gutter:    gap,
  ..range(K).map(k => range(N).map(n => lds-cell(k, n))).flatten()
)

// ── VGPR panel ────────────────────────────────────────────────────────────
// lane n holds B[0,n] through B[K-1,n] — one complete column of B
#let vgpr-cell(k, n) = rect(
  width: cw, height: ch, fill: n-bg(n),
  stroke: 0.4pt + ink,
  align(center + horizon,
    text(size: 5pt, fill: ink)[B[#k,#n]]
  )
)

#let vgpr-row-labels = stack(dir: ttb, spacing: 0pt,
  ..range(K).map(k =>
    box(width: lm-vgpr, height: ch,
      align(right + horizon,
        text(size: 5pt, fill: dimmed)[vgpr #k #h(2pt)]
      )
    )
  )
)

#let vgpr-col-labels = grid(
  columns: (cw,) * N,
  column-gutter: gap,
  ..range(N).map(n =>
    align(center, text(size: 5pt, fill: dimmed)[lane #n])
  )
)

#let vgpr-grid = grid(
  columns: (cw,) * N,
  rows:    (ch,) * K,
  column-gutter: gap,
  row-gutter:    gap,
  ..range(K).map(k => range(N).map(n => vgpr-cell(k, n))).flatten()
)

// ── Arrow ─────────────────────────────────────────────────────────────────
#let panel-h = K * ch
#let the-arrow = box(width: arrow-w, height: panel-h,
  align(center + horizon,
    stack(dir: ttb, spacing: 3pt,
      text(size: 20pt, fill: rgb("#444444"))[→],
      text(size: 5.5pt, fill: dimmed)[ds_read_tr],
    )
  )
)

// ── Panel headers ─────────────────────────────────────────────────────────
#let lds-header  = align(center,
  text(size: 6pt, weight: "bold", fill: dimmed)[LDS (shared)]
)
#let vgpr-header = align(center,
  text(size: 6pt, weight: "bold", fill: dimmed)[Per-lane VGPRs (private)]
)

// ── Title ─────────────────────────────────────────────────────────────────
#let title = text(size: 8pt, weight: "bold")[
  LDS transpose: row-major B into per-lane VGPRs
]

// ── Caption ───────────────────────────────────────────────────────────────
#let caption-text = text(size: 6pt, fill: dimmed)[
  B is stored row-major in LDS (left). After ds_read_tr, lane n holds \
  B[0,n] through B[K-1,n] in its VGPRs: one complete column of B (right). \
  Colours identify each N-column. Example: K = 4, N = 8.
]

// ── Column widths (reused across three rows) ───────────────────────────────
#let cols = (lm-lds, N * cw, arrow-w, lm-vgpr, N * cw)

// ── Final layout ──────────────────────────────────────────────────────────
#pad(top: 14pt, bottom: 14pt, left: 10pt, right: 14pt,
  stack(dir: ttb, spacing: 4pt,
    title,
    v(2pt),
    // Panel headers
    grid(columns: cols, column-gutter: gap,
      [], lds-header, [], [], vgpr-header,
    ),
    // Column-axis labels
    grid(columns: cols, column-gutter: gap,
      [], lds-col-labels, [], [], vgpr-col-labels,
    ),
    // Data rows with row-axis labels and arrow
    grid(columns: cols, column-gutter: gap,
      lds-row-labels, lds-grid, the-arrow, vgpr-row-labels, vgpr-grid,
    ),
    v(4pt),
    caption-text,
  )
)
