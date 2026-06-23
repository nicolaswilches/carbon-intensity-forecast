# Poster content and layout (A0 portrait, Figma)

Drop-in copy and figure plan for the capstone poster. Format A0 portrait (841 x 1189 mm).
Reflects the final findings (single-tier wins, flows do not help, EM head-to-head), not
the May structure. Style: Arial, plotly_white figures, no em dashes, no emojis. Palette:
single-tier `#E05312` (orange), two-tier `#129FE0` (blue), actual gray `#7F7F7F`; regional
palette per `src/carbon_forecast/plotting/config.py`. Target 5 large figures, each readable
from 1.5 m.

---

## Layout map (top to bottom)

```
+----------------------------------------------------------+
|  HEADER BAND (full width)                                |
|  Title | author | affiliation | advisor                  |
+----------------------------------------------------------+
|  ONE-LINE TAKEAWAY (full width, large)                   |
+----------------------------------------------------------+
|  LEFT COLUMN              |  RIGHT COLUMN                 |
|  1 Problem & gap          |  4 Headline result (atlas)    |
|  2 Approach (architecture |  5 Finding 1: single beats    |
|    figure)                |    two-tier  [FIG]            |
|  3 Data & setup           |  6 Finding 2: flows don't help|
|                           |  7 Why: Tier 1 noise  [FIG]   |
|                           |  8 Head-to-head vs EM         |
+----------------------------------------------------------+
|  FOOTER BAND (full width)                                |
|  Conclusions | references (compressed) | QR to repo+report|
+----------------------------------------------------------+
```

---

## HEADER BAND

- **Title:** Open, Reproducible Consumption-Based Carbon Intensity Forecasting with
  Explicit Cross-Border Flow Modeling
- **Author:** Nicolás Higuera Wilches
- **Affiliation:** IE University, School of Science and Technology
- **Advisor:** Prof. Bissan Ghaddar
- **Capstone, MSc Data Science and Business Analytics, 2026**

## ONE-LINE TAKEAWAY (full width, largest non-title type)

> A simple model that predicts carbon intensity directly beats the standard two-tier
> pipeline in every grid we tested, and lands within a few points of the closed
> operational forecast.

---

## LEFT COLUMN

### 1. Problem and gap
- Computing can move in time and place, so running it on cleaner grids and cleaner hours
  cuts emissions. This needs an accurate, local measure of carbon intensity.
- What matters is **consumption-based** carbon intensity, which counts the carbon in
  imported electricity. Most open forecasters target only **production-based** intensity.
- **The gap:** no open, reproducible model targets consumption-based intensity with
  explicit cross-border flows. The one operational system that does (Electricity Maps) is
  closed. We open it and test what actually helps.

### 2. Approach  [ARCHITECTURE FIGURE, build in Figma]
- Five structurally diverse grids: US-MIDA-PJM, US-NY-NYIS, Finland, Belgium, Singapore.
- We compare **four frameworks** on two targets (production and consumption):
  - **Two-tier (CarbonCast-extended):** Tier 1 per-source ANNs forecast generation, plus
    Tier 1 flow ANNs forecast interconnector net flow; a Tier 2 CNN-LSTM turns those into
    a 96 hour carbon-intensity forecast. Our extension (flows + partner CI) marked as the
    contribution.
  - **Single-tier:** the same CNN-LSTM predicting carbon intensity directly, no Tier 1.
- Diagram: boxes for Tier 1 source ANNs and Tier 1 flow ANNs feeding the Tier 2 CNN-LSTM,
  with the flow path highlighted in orange as "our extension."

### 3. Data and setup
- Electricity Maps (academic license): hourly carbon intensity, power breakdown, flows.
  Open-Meteo weather (ERA5 history, GFS forecast). Jan 2021 to May 2026, five zones.
- Train 2021 to 2025, validate Jan to Apr 2026, **Test A** May 2026 (full 1 to 96 h).
- **Test B** live June 8 to 21 2026: captured the Electricity Maps forecast as it was
  issued (non-recoverable), enabling the head-to-head on the 1 to 24 h band.
- RMSE loss, Adam with gradient clipping, seed-averaged.

---

## RIGHT COLUMN

### 4. Headline result: per-region atlas (Test A, best model = single-tier consumption)
Big numbers, MAPE:

| Zone | MAPE |
|---|---|
| Singapore | **0.71%** |
| US-MIDA-PJM | 5.93% |
| US-NY-NYIS | 5.98% |
| Finland | 12.05% |
| Belgium | 41.57% |

Framing: accuracy spans two orders of magnitude. A tight gas island (Singapore) is
trivial; large US markets sit near 6%; small import-heavy, fast-decarbonizing European
grids are hardest, with Belgium the clear failure case (a May distribution shift the model
does not anticipate).

### 5. Finding 1: the two-tier structure does not earn its complexity  [FIG: framework comparison]
- A single-tier network matches or beats the two-tier model in **every zone**, decisively
  on the consumption target (e.g. Belgium 41.6 vs 64.4, Finland 12.1 vs 21.9).
- We still reproduce CarbonCast: production two-tier PJM **6.04** vs published 5.29.
- **Figure:** `results_framework_cons.pdf` (single orange vs two-tier blue bars per zone).
  Optionally pair with `results_framework_prod.pdf`.

### 6. Finding 2: cross-border flow inputs do not help
- Turning the interconnector and partner-CI inputs off leaves the model as good or better
  in **four of five zones**. Belgium is actively hurt (28.98 to 58.50 with flows on);
  Finland improves only inside the noise band.
- The feature designed to encode physical coupling helps only if it can be predicted well,
  and over this horizon it cannot.

### 7. Why: Tier 1 forecasts inject noise  [FIG: Tier 1 diagnostic]
- The only structural difference is feeding Tier 1 generation forecasts into Tier 2. Those
  forecasts are imperfect, so they add noise rather than signal and cap the two-tier model.
- Tier 1 total-generation error (normalized RMSE) grows with horizon and is worst exactly
  where the two-tier model fails most (Belgium 18% to 25%).
- **Figure:** `tier1_horizon_error.pdf` (per-region Tier 1 error by horizon).

### 8. Head-to-head vs Electricity Maps (Test B, 1 to 24 h)
- The open single-tier model lands within a few points of the closed operational forecast,
  beats it in Finland, and ties in Singapore.

| Zone | Electricity Maps | Ours (single cons) |
|---|---|---|
| Singapore | 0.63% | 0.65% |
| US-MIDA-PJM | 3.17% | 4.33% |
| US-NY-NYIS | 3.82% | 6.36% |
| Finland | 20.74% | **17.69%** |
| Belgium | 23.48% | 33.26% |

Caveat (small print): our models are replayed on actual weather, an optimistic upper
bound; even so they trail in the three larger grids.

---

## FOOTER BAND

### Conclusions
- An open, reproducible framework reaches operational-level accuracy on most grids.
- The two-tier decomposition and cross-border flow features do not improve a learned
  single-tier model; both add noise through imperfect first-tier forecasts.
- Per-region heterogeneity is large and structurally explained; Belgium is the open
  failure case (distribution shift, unmodeled nuclear outages).

### References (compressed, 3 to 4 max)
- Maji et al., CarbonCast, BuildSys 2022.
- Electricity Maps, operational forecast, 2025.
- Full reference list in the report.

### QR codes
- QR 1: GitHub repository.
- QR 2: full report (PDF).

---

## Figures to export as SVG for Figma (run with write_image to .svg)
Recommended 5 large:
1. ARCHITECTURE diagram, build natively in Figma (no script).
2. `results_framework_cons.pdf` (Finding 1), and optionally `results_framework_prod.pdf`.
3. `tier1_horizon_error.pdf` (Finding 3, the why).
4. `results_horizon_mape.pdf` (per-region behavior across horizon, optional).
5. A small **Test B bar** (EM vs ours) and a small **flow on/off bar**, not yet generated
   as standalone figures; can be added from `outputs/test_b_headtohead.csv` and the flow
   numbers if wanted.

Backups available: `results_fva_frameworks.pdf`, `eda_mix.pdf`, `eda_correlation.pdf`,
`eda_ci_evolution.pdf`.
