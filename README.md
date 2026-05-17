# carbon-intensity-forecast

Open, reproducible deep learning framework for short-horizon consumption-based carbon intensity forecasting, with explicit cross-border flow modeling. Five zones (US-MIDA-PJM, US-NY-NYIS, FI, BE, SG). Benchmarked head-to-head against Electricity Maps' operational forecast.

Phase 1 capstone, MSc Data Science and Business Analytics, IE University. Advisor: Prof. Bissan Ghaddar. Deadline 2026-07-03.

## Setup

```bash
uv sync --extra dev
cp .env.example .env  # then fill in EM_API_KEY
set -a && source .env && set +a
```

## Layout

```
src/carbon_forecast/   installable package (data, models, evaluation, plotting, utils)
scripts/               extraction and collection entry points
notebooks/             S01..S07 session-numbered exploration
data/raw/              immutable EM and weather pulls (gitignored)
data/processed/        modelling-ready joins (gitignored)
models/                trained artefacts (gitignored, published to HF Hub at end)
outputs/               figures and tables
reports/               proposal + final report (Overleaf-synced via drag-drop)
docs/                  literature notes and design docs
tests/                 lean pytest suite
```

See `CLAUDE.md` for conventions and `TASKS.md` for the current task board.
