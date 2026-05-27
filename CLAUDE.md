# CLAUDE.md

Carbon intensity forecasting capstone, Phase 1 of three. MSc Data Science and Business Analytics, IE University. Advisor: Prof. Bissan Ghaddar. Deadline 2026-07-03.

Inherits from `ds/projects/CLAUDE.md` (workspace conventions: uv, Jupyter session-numbered notebooks, flat memory).

## Status

Week 1 complete: data pipeline, full 2021-2025 extraction across 5 zones (1,200 Parquet files), S01 validation, S02 descriptive analysis, and the plotting design system in `src/carbon_forecast/plotting/config.py`. Week 2 next: E2 (CarbonCast-faithful) reproduction. Proposal on Overleaf and shared with Prof. Ghaddar.

## Locked thesis (frozen 2026-05-10 via /grill-me)

Open, reproducible deep learning framework for short-horizon consumption-based carbon intensity forecasting, with explicit cross-border flow modeling, across five zones (US-MIDA-PJM, US-NY-NYIS, FI, BE, SG), benchmarked against Electricity Maps' operational forecast.

Four contributions:
1. Open architecture (EM's operational model is closed-source).
2. Feature ablation on interconnector flows.
3. Per-region heterogeneity atlas.
4. Head-to-head benchmark with failure-mode analysis.

## Where decisions live

- Cross-session memory: `~/.claude/projects/-Users-nicolaswilches-ds-projects-carbon-intensity-forecast/memory/`. Index in `MEMORY.md`. Read `project_thesis.md` first; it carries the full locked decisions log including the weekly timeline.
- Project glossary: `UBIQUITOUS_LANGUAGE.md` at the project root. Canonical terms (Tier 1 / Tier 2, E2 / E3, Test A / Test B, etc.).
- Open tasks: `TASKS.md` at the project root.
- Recurring lessons: `LESSONS.md` at the project root.

## Stack

- Python via `uv`. Package: `carbon_forecast` under `src/`.
- Deep learning: Keras on TensorFlow.
- Data: Electricity Maps API (academic license, key in `.env`, gitignored). Open-Meteo for weather (ERA5 historical, GFS forecast).
- Storage: Parquet under `data/raw/{em,weather}/` and `data/processed/`.
- Notebooks: session-numbered `notebooks/S01_*.ipynb` through `S07_*.ipynb`.
- Report: Overleaf with `acmart sigconf`. Local source of truth at `reports/proposal/` and (later) `reports/report/`. Free Overleaf, so drag-drop sync only (Y1 workflow).
- Poster: A0 portrait in Figma.

## Conventions

- Timestamps stored in UTC; local only for plotting, marked explicitly.
- Zone codes follow Electricity Maps zone keys.
- Plotly defaults: `template='plotly_white'`, `font=dict(family='Arial')`.
- Three palettes (regional, energy source, model comparison) defined in `src/carbon_forecast/plotting/config.py`.
- Writing style: no em dashes, no emojis, varied sentence structure per Michelle Kassorla's article (see `memory/feedback_style.md`).
- Commits: imperative, lowercase. NO `Co-Authored-By` footer (`memory/feedback_commit_attribution.md`).
- PRs even when working solo, squash-merge.
- GitHub Projects kanban mirrors `TASKS.md`.

## Likely commands

```bash
# Source the API key
set -a && source .env && set +a

# Extract a month of EM history (script TBD in Week 1)
python scripts/extract_historical.py --zone BE --month 2024-01

# Run the lean test suite
pytest tests/
```

## Next steps

Week 1 deliverables in `TASKS.md`. Full week-by-week timeline in `memory/project_thesis.md` under the "Weekly timeline" section.
