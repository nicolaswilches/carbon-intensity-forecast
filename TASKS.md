# TASKS

Phase 1 deadline: 2026-07-03. Full weekly timeline in `memory/project_thesis.md`.

## Done (grilling phase, 2026-05-10 to 2026-05-12)

- [x] Lock thesis: open consumption-based CI forecasting with cross-border flows, 5 zones
- [x] Lock architecture: E3 CarbonCast-extended modular two-phase, E2 faithful baseline
- [x] Lock evaluation: RMSE loss, MAPE primary, M3a full failure-mode breakdown
- [x] Lock splits: chronological dual-test (Train, Val, Test A May 2026, Test B Jun 8-21)
- [x] Lock data plan: EM academic key, Open-Meteo hybrid (ERA5 + GFS)
- [x] Lock codebase: src/carbon_forecast/, Keras on TF, pyproject.toml, lean pytest
- [x] Lock design language and report template (acmart sigconf on Overleaf)
- [x] Verify EM academic key returns real data across all 5 zones (2021 onward)
- [x] Write reports/proposal/proposal.tex, upload to Overleaf, share with Prof. Ghaddar
- [x] Draft UBIQUITOUS_LANGUAGE.md project glossary

## Week 1 (May 11-17): foundation — COMPLETE

- [x] Init `pyproject.toml` with uv
- [x] Scaffold `src/carbon_forecast/{data,models,evaluation,plotting,utils}/`
- [x] `em_client.py` with retry, sandbox detection, tests
- [x] `weather_client.py` (archive + GFS), tests
- [x] `storage.py`: paths, flatten functions, atomic Parquet I/O, tests
- [x] `zones.py` with locked 5-zone metadata, bounding-box centroids from electricitymap-contrib
- [x] `extract.py` + `scripts/extract_historical.py` with idempotent skip, continue-on-error, dry-run
- [x] Path convention amended to monthly Parquet (memory: project_thesis.md)
- [x] Multi-coordinate weather (PJM) added as stretch experiment (memory: project_thesis.md)
- [x] Run historical EM extraction 2021-01 to 2025-12 (1,200 files, all 5 zones)
- [x] Run ERA5 weather pull (same window)
- [x] Author `notebooks/S01_extraction_validation.ipynb` (data validated clean: 1200/1200)
- [x] Author `notebooks/S02_descriptive_analysis.ipynb` (2 sections, 8 charts)
- [x] Build `src/carbon_forecast/plotting/config.py` design system (palettes, style_fig, percentile colorscale)
- [x] Re-pull PJM weather against corrected centroid
- [x] Set up nbstripout (keep notebooks lean in git)
- [ ] Install Zotero with Better BibTeX, configure export to `references.bib` (user-side, carried)
- [ ] Init GitHub Projects kanban mirroring this file (user-side, carried)
- [ ] Track Prof. Ghaddar's proposal comments on Overleaf (user-side, carried)
- [ ] Fill S01 + S02 findings cells from observed figures (carried)

## CONTINGENCY (catch-up from Jun 8, ~3 weeks behind; deadline Jul 3)

Collection-first because Test B window (Jun 8-21) is a live, non-recoverable clock.

- [x] Test B forecast collector: em_client forecast method, storage snapshot flatten/paths, scripts/collect_forecasts.py, launchd 6h schedule (installed + kickstart-verified)
- [!] EM CI forecast horizon is 24h not 72h on academic key; power-breakdown forecast 401. Contribution 4 narrows to 1-24h. Flag to Prof. Ghaddar; consider EM support email (memory: project_em_forecast_horizon.md)
- [x] Data-processing pipeline (prerequisite for all tiers): utils/calendar, data/processor (CarbonCast Table 1 EFs), data/normalize (train-only), windowing. All 5 zones materialized to data/processed/{zone}.parquet (43824h each)

## Week 2 (compressed into today): E2 reproduction

- [ ] Tier 1 source ANN (per-source feedforward, 168h input, 96h dense head)
- [ ] Tier 1 flow ANN (per-interconnector)
- [ ] Tier 2 CNN-LSTM
- [ ] E2 orchestrator (CarbonCast-faithful), train + validate on Jan-Apr 2026
- [ ] Notebook S04 (E2 results)

## Weeks 3 to 8

See `memory/project_thesis.md` "Weekly timeline" section. Stretch experiments and Plan-B rules are also there.
