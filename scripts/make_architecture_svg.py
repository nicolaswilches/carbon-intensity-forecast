"""Generate the framework architecture diagram as a clean SVG.

Recreates the CarbonCast two-tier architecture in the poster's visual language:
Inter font, warm off-white background, poster palette, no drop shadows.

    .venv/bin/python scripts/make_architecture_svg.py
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT  = os.path.join(ROOT, "outputs", "figures", "poster", "architecture.svg")

# ── palette ──────────────────────────────────────────────────────────────────
BG        = "#FAF8F5"
INK       = "#1A1A1A"
GRAY      = "#6B6B6B"
DIVIDER   = "#C8C4BE"
T1_STROKE = "#0072B2"
T1_FILL   = "#EEF5FB"
T2_STROKE = "#E05312"
T2_FILL   = "#FEF2EE"
BOX_FILL  = "#F0EFED"
BOX_STR   = "#B0ADA8"
ARROW     = "#6B6B6B"

W, H = 900, 380


def rect(x, y, w, h, fill, stroke, stroke_w=1.2, rx=6, dash="") -> str:
    d = f'stroke-dasharray="{dash}" ' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}" {d}/>')


def text(x, y, content, size=14, weight=400, color=INK, anchor="middle",
         dy=0) -> str:
    return (f'<text x="{x}" y="{y}" dy="{dy}" font-family="Inter" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}" '
            f'text-anchor="{anchor}">{content}</text>')


def arrow(x1, y1, x2, y2, color=ARROW) -> str:
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="1.4" '
            f'marker-end="url(#arrowhead)"/>')


def multi(x, y, lines, size=13, weight=400, color=INK, anchor="middle",
          leading=17) -> str:
    out = []
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else leading
        out.append(f'<tspan x="{x}" dy="{dy}">{line}</tspan>')
    return (f'<text font-family="Inter" font-size="{size}" font-weight="{weight}" '
            f'fill="{color}" text-anchor="{anchor}" x="{x}" y="{y}">'
            + "".join(out) + "</text>")


svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"
     width="{W}" height="{H}">

  <defs>
    <marker id="arrowhead" markerWidth="7" markerHeight="5"
            refX="6" refY="2.5" orient="auto">
      <polygon points="0 0, 7 2.5, 0 5" fill="{ARROW}"/>
    </marker>
  </defs>

  <!-- background -->
  <rect width="{W}" height="{H}" fill="{BG}"/>

  <!-- ── INPUT BOX (left) ───────────────────────────────────── -->
  {rect(18, 100, 90, 80, BOX_FILL, BOX_STR, rx=5)}
  {multi(63, 125, ["Historical", "source", "mix"], size=12, color=GRAY, leading=15)}

  <!-- arrow: input → tier1 -->
  {arrow(108, 140, 148, 140)}

  <!-- ── TIER 1 DASHED BOX ─────────────────────────────────── -->
  {rect(150, 52, 230, 176, "none", T1_STROKE, stroke_w=1.4, dash="6 3", rx=10)}
  {text(265, 44, "Tier 1", size=13, weight=600, color=T1_STROKE)}

  <!-- non-renewables label -->
  {text(178, 102, "Non-", size=11, color=GRAY, anchor="start")}
  {text(178, 116, "renewables", size=11, color=GRAY, anchor="start")}

  <!-- renewables label -->
  {text(178, 178, "Renewables", size=11, color=GRAY, anchor="start")}

  <!-- ANN box 1 -->
  {rect(228, 88, 60, 36, T1_FILL, T1_STROKE, rx=5)}
  {text(258, 112, "ANN", size=13, weight=600, color=T1_STROKE)}

  <!-- ANN box 2 -->
  {rect(228, 160, 60, 36, T1_FILL, T1_STROKE, rx=5)}
  {text(258, 184, "ANN", size=13, weight=600, color=T1_STROKE)}

  <!-- arrows into ANN boxes -->
  {arrow(220, 106, 228, 106)}
  {arrow(220, 178, 228, 178)}

  <!-- arrows out of ANN boxes → right edge of tier1 -->
  {arrow(288, 106, 378, 130)}
  {arrow(288, 178, 378, 150)}

  <!-- ── MIDDLE LABEL ───────────────────────────────────────── -->
  {multi(435, 122, ["Individual", "source", "production", "forecasts"],
         size=11, color=GRAY, anchor="middle", leading=14)}

  <!-- arrow: tier1 right → tier2 left -->
  {arrow(490, 140, 528, 140)}

  <!-- ── TIER 2 DASHED BOX ─────────────────────────────────── -->
  {rect(530, 90, 160, 100, "none", T2_STROKE, stroke_w=1.4, dash="6 3", rx=10)}
  {text(610, 82, "Tier 2", size=13, weight=600, color=T2_STROKE)}

  <!-- CNN-LSTM box -->
  {rect(550, 108, 120, 64, T2_FILL, T2_STROKE, rx=5)}
  {text(610, 136, "CNN", size=13, weight=600, color=T2_STROKE)}
  {text(610, 153, "LSTM", size=13, weight=600, color=T2_STROKE)}

  <!-- arrow: tier2 → output -->
  {arrow(690, 140, 728, 140)}

  <!-- ── OUTPUT LABEL ───────────────────────────────────────── -->
  {multi(790, 122, ["96-hour", "carbon", "intensity", "forecast"],
         size=12, color=INK, anchor="middle", leading=15)}

  <!-- ── BOTTOM INPUT: weather ─────────────────────────────── -->
  {rect(150, 290, 160, 44, BOX_FILL, BOX_STR, rx=5)}
  {multi(230, 307, ["96-hour weather", "forecasts"], size=12, color=GRAY,
         anchor="middle", leading=16)}

  <!-- weather → tier1 (up) -->
  {arrow(210, 290, 240, 228)}
  <!-- weather → tier2 (up) -->
  {arrow(280, 290, 590, 190)}

  <!-- ── BOTTOM INPUT: historical CI ───────────────────────── -->
  {rect(530, 290, 160, 44, BOX_FILL, BOX_STR, rx=5)}
  {multi(610, 307, ["Historical carbon", "intensity"], size=12, color=GRAY,
         anchor="middle", leading=16)}

  <!-- historical CI → tier2 (up) -->
  {arrow(610, 290, 610, 190)}

</svg>"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write(svg)
print("wrote", OUT)
