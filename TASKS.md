# TASKS

Phase 1 deadline: 2026-07-03. Full weekly timeline in `memory/project_thesis.md`.

## NOW — status (as of 2026-06-27)

Report argument-complete and adversarial-reviewed on `main`. All review flags resolved.

- [x] Per-zone training window (advisor): BE → 2024, FI → 2024, SG/NYIS/PJM full. All 4 frameworks. `final_metrics.csv` is ground truth.
- [x] Two adversarial review passes complete — all critical/high/medium/low flags fixed (see archive below).
- [ ] USER-SIDE: sync `main` -> Overleaf, compile, verify layout (no local LaTeX).
- [ ] HF Hub: upload models (incl E2, now saved) + preds; make repo public; link in README.
- [ ] Poster: `reports/poster/poster.html` built; add real QR codes once repo public; print to A0 PDF from Chrome.
- [ ] FUTURE: faithful GFS-weather head-to-head (drop perfect-weather caveat); longer Test B; broader per-zone-window study.

## Report polish — adversarial review (complete as of 2026-06-27)

### Pass 1 flags (all fixed)
- [x] H1: Soften CarbonCast reproduction language; confounder footnote — §abstract, §8.1, conclusion
- [x] M1: Seed std subscripts in tab:structural; variance note in tab:flow caption
- [x] M2: Implementation details subsection (hyperparameters) in §5
- [x] M3: Mechanistic explanation for flat error-by-horizon in §8.3
- [x] M4: Two-tier causal chain explicit in §9.1
- [x] M5: EnsembleCI softened to "strong" in §2
- [x] M6: Atlas heterogeneity qualified as "descriptively consistent with" in conclusion
- [x] L1–L6: BE outage source; GitHub URL + MIT; §4.6 fragment; NYIS softened; CEF note; partner zone description

### Pass 2 flags (all fixed)
Critical:
- [x] C1: FI window in limitations was "2023" — corrected to "2024" (matches `final_metrics.csv`)
- [x] C2: "Finland result — our only outright win" — removed; EM leads in all 5 zones on Test B
- [x] C3: CI units (gCO₂eq/kWh) never stated — added in §3 at equation definition
High:
- [x] H2: §2 gap paragraph still said "strongest prior work" — changed to "strongest academic baseline"
- [x] H3: §4.6 opened with sentence fragment — rewritten as full sentence
- [x] H4: §4.3 Finland "halves its carbon intensity over," — completed to "over the five-year window"
- [x] H5: "competitive across all five" in abstract + conclusion — replaced with "with larger gaps in the two transitioning grids"
Medium:
- [x] M7: "most academic forecasters target" — backed with citations to CarbonCast + EnsembleCI
- [x] M8: Dangling scope-2 "variant" mention removed (never reported)
- [x] M9: Belgium nuclear framing updated — 2022 extension decision acknowledged
- [x] M10: "several seeds" → "three seeds" in §7
- [x] M11: OOF-on-short-window limitation added to §9.2
- [x] M12: tab:flow caption explains why its values differ from tab:structural
Low:
- [x] L7: §1 notes flow-modeling contribution is a null result ("finding they do not improve accuracy")
- [x] L8: ~40 scored origins per zone stated in §8 Test B section
- [x] L9: Noise-band claim in §8.1 linked to explicit delta and Table 2
- [x] L10: 168h lookback attributed to CarbonCast inheritance in §4.6

Standing rule: commit + push report changes to main immediately (memory: feedback_report_autocommit).

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
- [x] Partner-zone CI forecasts (E3 input, not backfillable) added to collect_forecasts.py: 20 external partners auto-derived from flow columns, CI-forecast-only. Same launchd job
- [ ] Partner-zone CI history (E3 input, backfillable): add 20 partner zones to historical past-range extraction. Not urgent
- [x] Data-processing pipeline (prerequisite for all tiers): utils/calendar, data/processor (CarbonCast Table 1 EFs), data/normalize (train-only), windowing. All 5 zones materialized to data/processed/{zone}.parquet (43824h each)

## Week 2 (compressed into today): E2 reproduction

- [x] Tier 1 source ANN (per-source feedforward, 168h input, 96h dense head). Clip >=0, MAE/RMSE eval. Smoke-trained BE wind
- [x] Tier 1 flow ANN (per-interconnector, signed net flow, no clip). Smoke-trained BE<->FR
- [x] Tier 2 CNN-LSTM (Conv1D 4->pool->Conv1D 16->LSTM 24->dropout->Dense 96). Unified 264-step seq (168 past + 96 future) with is_future flag. Smoke-trained BE, MAPE valid here
- [x] E2 orchestrator (carboncast_faithful.py): out-of-fold Tier 1 forecasts (leave-one-year-out) for Tier 2 training + final models for inference. Segmented windowing for gapped folds. Smoke-trained BE end-to-end
- [x] Notebook S04 (E2 results): imports orchestrator, loads processed parquets (Colab-portable), full-settings train, MAPE/MAE/RMSE, per-horizon degradation curve. Built; full run pending (BE-first then 5 zones)

## DEADLINE Friday 2026-06-19 (advisor, see memory/advisor_directives.md)

- [x] Single-tier CI baseline (Tier 2 standalone), both targets, 5 zones, val-selected
- [x] Production two-tier (E2) real-split 5-zone, val-selected; consumption two-tier done (FI on 2023 window)
- [x] Comparison: two-tier does NOT earn complexity (single-tier wins/ties all). In report Results. outputs/structural_comparison.csv
- [x] FI shorter window (2023): cons two-tier 24.04 -> 21.89
- [x] Report started: Related Work, full EDA (figures), Methodology, Setup, Iterative Findings, structural ablation, Tier 1 diagnostic. Live on Overleaf via GitHub sync
- [x] CarbonCast PJM benchmark: ours ~6.0 vs published 5.29 lifecycle (faithful repro)
- [x] Report: abstract headline numbers, Finland-window effect (24.04->21.89), conclusion synthesized
- [x] Report figures: per-horizon curve + forecast-vs-actual (scripts/make_results_figures.py from outputs/preds npz; wired into Results)
- [ ] Remaining report: feature ablation (flow on/off) — needs a dedicated seed-averaged run
- [ ] Post-Friday: separate BE model + diagnose what BE misses; optional 24x4 prototype on PJM

## Week 3 (in progress from 2026-06-09): E3 extension

- [x] Run E2 baseline S04 (BE, proxy split): MAPE 35.16%, MAE 70.93, RMSE 91.34 (14.4 min CPU). 5-zone run still a Colab job
- [x] Partner-CI history extraction: 20 partner zones, carbon-intensity/past-range 2021-2025 (1200/1200 months, 876k rows). extract_historical.py gained synthetic CI-only zone resolution + endpoint filter
- [x] Processor: per-partner net_flow_*_mw + partner_ci_* columns (gap-filled, coverage-aware). 5 frames re-materialized
- [x] Tier 2 generalized: config-driven target_col + dynamic_cols; future override is first-K channels (partner CI held as actuals = strategy A). E2 path regression-safe (106 tests pass)
- [x] Tier 1 flow: segmented windowing + val=None support, so it joins the OOF machinery like sources
- [x] E3 orchestrator carboncast_extended.py: source+flow Tier 1 (OOF+final) -> consumption-based Tier 2; partner CI via strategy A (train actuals) / persistence (inference). Smoke-tested BE end-to-end
- [x] Notebook S03 (feature engineering): cons-vs-prod gap, import dependence, partner-CI/gap corr 0.83 (BE), feature association, lookback ACF. Executed + findings filled
- [x] First E3 full BE run (proxy split): cons-based CI MAPE 42.59%, MAE 64.07, RMSE 79.05 (18.7 min). Lower MAE/RMSE than E2 on the harder cons-based target; MAPE not directly comparable (different targets). h96 MAPE spike to investigate
- [x] S05 E3 results notebook with full 1-96 per-horizon curve (MAPE + MAE, 2-panel). Diagnostic: error peaks at SHORT horizon (~h6), not h96; the "h96 spike" was a 5-point sampling artifact. Late-horizon rise is real (in MAE too) but secondary
- [x] Seed Tier 1 + Tier 2 (keras.utils.set_random_seed at orchestrator entry, default seed=0). Bit-exact reproducible on CPU
- [!] Variance large EVEN seeded: BE E3 3-seed MAPE 37.11 ± 4.95 (33.26-44.09), MAE 61.62 ± 3.51, RMSE 78.34 ± 3.35. Seed 0 (the default) is the unlucky high draw. Implication: tuning/comparison deltas < ~5 MAPE pts are noise; must seed-average. Variance reduction (more seeds / ensemble) now a WEEK 4 task
- [x] Last-value persistence feature: tested seed-averaged, NO improvement (within noise). Parked behind a toggle (off)
- [x] 5-zone seed-averaged on PROXY split: SG 1.36, PJM 7.15, NYIS 14.54, FI 25.66, BE 37.11 (MAPE). Strong per-region heterogeneity; variance is BE-specific
- [x] Extracted 2026 Jan-May (modeled + partners); switched S04/S05 to real split (train 21-25, val 26 Jan-Apr, Test A May)
- [!] BE unstable on real split (Test A): training diverges to NaN on some seeds; non-NaN seeds ~54-67 MAPE (vs 37 proxy). Data is clean (checked). FIX: add gradient clipping (clipnorm) to optimizer, then re-run BE
- [~] 4-zone real-split Test A run (FI/SG/PJM/NYIS) seed-averaged in progress; BE deferred to stability fix
- [ ] E3 5-zone run (Colab); compare vs E2 per zone (after seeding)
- [ ] Partner-CI forecast hook for Test B head-to-head (EM published partner forecast); persistence is the default elsewhere

## Weeks 4 to 8

See `memory/project_thesis.md` "Weekly timeline" section. Stretch experiments and Plan-B rules are also there.
