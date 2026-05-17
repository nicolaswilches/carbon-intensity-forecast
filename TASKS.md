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

## Week 1 (May 11-17): foundation

- [ ] Init `pyproject.toml` with uv (pandas, numpy, tensorflow, keras, plotly, pyarrow, requests, holidays, python-dotenv, scipy)
- [ ] Scaffold `src/carbon_forecast/{data,models,evaluation,plotting,utils}/` with `__init__.py` stubs
- [ ] Build `src/carbon_forecast/data/em_client.py` with retry, idempotency, sandbox-mode detection
- [ ] Build `src/carbon_forecast/data/weather_client.py` for Open-Meteo (archive + `/v1/gfs`)
- [ ] Build `src/carbon_forecast/data/storage.py` Parquet helpers
- [ ] Build `scripts/extract_historical.py`: monthly `/past-range` chunks across all 5 zones, 3 endpoints
- [ ] Run historical EM extraction for 2021-01-01 to 2025-12-31
- [ ] Run ERA5 weather pull (Open-Meteo archive) for the same window, one coordinate per zone
- [ ] Author notebook `S01_extraction_validation.ipynb`: shape checks, sandbox-disclaimer check, sample plots
- [ ] Author notebook `S02_descriptive_analysis.ipynb`: regional patterns, distributions, seasonality
- [ ] Install Zotero with Better BibTeX, configure export path for `references.bib`
- [ ] Initialise GitHub Projects kanban (Backlog / In Progress / Review / Done) mirroring this file
- [ ] Track Prof. Ghaddar's proposal comments and respond on the Overleaf project

## Weeks 2 to 8

See `memory/project_thesis.md` "Weekly timeline" section. Stretch experiments and Plan-B rules are also there.
