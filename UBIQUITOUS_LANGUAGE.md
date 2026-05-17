# Ubiquitous Language

Project glossary for the carbon intensity forecasting capstone (Phase 1). Extracted from the grilling conversation. Terms here are canonical: when alternatives exist, use the canonical term and avoid the listed aliases.

## Carbon Intensity Concepts

| Term | Definition | Aliases to avoid |
|---|---|---|
| **Carbon intensity (CI)** | Mass of CO2 equivalent emitted per unit of electrical energy, in gCO2eq/kWh. | Emission factor (a different concept), CO2 intensity |
| **Production-based CI** | Carbon intensity computed from electricity generated within a zone, ignoring imports and exports. | Generation-based CI, supply-side CI |
| **Consumption-based CI** | Carbon intensity computed for electricity consumed within a zone, accounting for interconnector flows. The target of E3 and the operationally correct variable for data-center scheduling. | Flow-traced CI (Electricity Maps' synonym), demand-side CI |
| **Lifecycle emission factor** | Carbon emitted per unit of a source's generation, including operational and infrastructural (supply-chain) emissions. | Scope 3 factor |
| **Direct emission factor** | Carbon emitted per unit of a source's generation, counting only operational combustion. | Scope 2 factor, operational factor |
| **Source** | A category of electricity generation: gas, coal, oil, nuclear, wind, solar, hydro, biomass, geothermal. | Fuel type, generation type |
| **Generation mix** | The decomposition of a zone's total production into per-source production at a given hour. | Source mix, fuel mix |
| **Interconnector** | A physical electrical connection between two zones that transmits import or export flow. | Tie line, cross-border line |
| **Flow** | Power exchange between two zones over an interconnector, in MW, signed by direction (import or export). | Exchange, transfer |
| **Partner zone** | A zone connected to a target zone via an interconnector. | Neighbor zone, source zone (overloaded) |

## Target Geography

| Term | Definition | Aliases to avoid |
|---|---|---|
| **Zone** | An Electricity Maps electrical region identified by a zone key (e.g. US-MIDA-PJM, BE, SG). | Region, area, grid |
| **Target zones** | The five zones in scope: US-MIDA-PJM, US-NY-NYIS, FI, BE, SG. | Scope zones |
| **Near-closed control** | A zone with minimal interconnection. Singapore in our set. Used to test whether production-based and consumption-based CI converge when flows are small. | Closed grid (not literally true), island |

## Forecasting Architecture

| Term | Definition | Aliases to avoid |
|---|---|---|
| **Forecast horizon** | The lookahead period of a forecast, in hours. Locked at 96 hours. | Prediction window |
| **Input window** | The historical context length fed to a model, in hours. Locked at 168 hours. | Lookback window, context length |
| **Tier 1** | The first stage of the two-phase architecture. Per-source ANNs and per-interconnector ANNs produce 96-hour forecasts of their respective target signals. | Stage 1, phase 1 (overloaded with project phases) |
| **Tier 2** | The second stage. A single CNN-LSTM consuming Tier 1 outputs plus historical consumption-based CI and weather, producing the 96-hour consumption-based CI forecast. | Stage 2, phase 2 (overloaded) |
| **Direct multi-output** | An output strategy where a single forward pass produces the full 96-hour forecast vector via a 96-unit dense head. The locked strategy for both tiers. | Single-shot prediction, one-shot prediction |
| **Recursive (autoregressive) output** | An output strategy where the model predicts t+1, feeds the prediction back, and iterates to t+H. Rejected for this project because of error compounding. | Roll-out, iterative forecasting |
| **Sequence-to-sequence (seq2seq)** | An encoder-decoder output strategy. Parked as a stretch alternative pending supervisor input. | Encoder-decoder |
| **E2** | The CarbonCast-faithful production-based variant. A reference baseline built on the same five zones. Not the headline contribution. | Faithful reproduction, reference model |
| **E3** | The CarbonCast-extended consumption-based variant. Adds per-interconnector Tier 1 ANNs and retargets Tier 2 to consumption-based CI. The headline model. | Extended model, our model |
| **Flow ablation** | An experiment that toggles the Tier 1 flow ANNs off and measures the change in Tier 2 performance. Tests Contribution 2's empirical claim. | Feature ablation (ambiguous; see Flagged Ambiguities) |

## Data Sources and Storage

| Term | Definition | Aliases to avoid |
|---|---|---|
| **Electricity Maps (EM)** | The primary data provider. Supplies historical and forecast CI, generation mix, and interconnector flows via HTTP API. | ElectricityMap (older brand spelling) |
| **EM operational forecast** | Electricity Maps' productionised 72-hour flow-traced CI forecast, in commercial use since January 2025. The benchmark target for Contribution 4. | EM forecast (ambiguous), production forecast (overloaded) |
| **Sandbox tier** | An Electricity Maps API tier returning synthetic, structurally-correct payloads marked `SANDBOX_MODE_DATA`. Useful for plumbing tests only. | Free tier (imprecise) |
| **Academic tier** | The Electricity Maps API tier with real historical and forecast data, granted under academic license. The tier in active use. | Paid tier (we are not paying), real tier |
| **`/past`** | EM endpoint returning a single historical CI observation for a zone at a given timestamp. | Historical endpoint |
| **`/past-range`** | EM endpoint returning a date-range of historical observations in one call. Used for monthly bulk pulls. | Bulk historical endpoint |
| **`/forecast`** | EM endpoint returning the operational forecast (default flow-traced). | Forward endpoint |
| **`/power-breakdown/past`** | EM endpoint returning historical generation mix plus interconnector totals. Exposes `powerConsumptionBreakdown` and `powerProductionBreakdown` as separate fields. | Mix endpoint |
| **`/electricity-flows/past`** | EM v4 endpoint returning per-partner-zone import and export flows in MW. | Flow endpoint |
| **Open-Meteo** | Weather data provider used in a hybrid configuration: ERA5 for historical, GFS for forecast. | Weather API (too generic) |
| **ERA5** | The ECMWF reanalysis dataset used for historical weather. The gold standard reanalysis product. | ECMWF reanalysis, reanalysis (too generic) |
| **GFS** | NOAA Global Forecast System. Used for weather forecasts during the operational test window. | NCEP GFS, GFS forecast (when in context) |
| **Parquet** | The columnar storage format used for all raw and processed data. | Columnar format (less specific) |
| **Forecast snapshot** | A single capture of EM's operational forecast at a given clock time during the Test B window. Stored under `data/raw/em/forecasts/{zone}/{snapshot_iso}.parquet`. | Forecast pull, forecast record |

## Evaluation Framework

| Term | Definition | Aliases to avoid |
|---|---|---|
| **Train fold** | 2021-01-01 to 2025-12-31. Used to fit all model parameters. | Training set (acceptable alias) |
| **Validation fold** | 2026-01-01 to 2026-04-30. Used for hyperparameter tuning and model selection. Can be inspected many times. | Dev set, val set |
| **Test A** | 2026-05-01 to 2026-05-31. The chronological hold-out fold for general accuracy reporting. Touched once. | May test, hold-out (ambiguous; there are two test sets) |
| **Test B** | 2026-06-08 to 2026-06-21. The two-week head-to-head window during which EM forecasts are captured live. Touched once. | June test, head-to-head set |
| **Head-to-head benchmark** | The comparison between E3 and EM's operational forecast over Test B. Forms Contribution 4. | EM benchmark, operational benchmark |
| **MAPE** | Mean absolute percentage error. The primary reporting metric, matching CarbonCast and EnsembleCI. | Percentage error |
| **MAE** | Mean absolute error in gCO2eq/kWh. Secondary metric for physical-unit interpretability. | Average error |
| **RMSE** | Root mean square error. Training loss (CarbonCast-faithful) and a secondary outlier-sensitive readout. | Quadratic error |
| **Per-horizon degradation curve** | A plot of forecast error against horizon (1, 6, 12, 24, 48, 72, 96 hours). The signature visual for Contribution 4. | Horizon plot, degradation plot |
| **Failure-mode analysis** | Slicing test data along axes (time-of-day, day-of-week, season, renewable regime) to characterise where each model wins, loses, or fails. The discipline that turns Contribution 4 from a single MAPE number into a paragraph-level story. | Error analysis (broader), residual analysis (specific to residuals) |
| **Breakdown axis** | A dimension along which a test fold is sliced to compute per-slice metrics. Six axes are in scope. | Slice dimension |
| **Diebold-Mariano test** | A pairwise hypothesis test for whether two forecasts have significantly different accuracy. Used to defend per-slice comparison claims. | DM test |

## Project Decision Identifiers

| Term | Definition | Aliases to avoid |
|---|---|---|
| **Contribution 1** | Open, reproducible architecture (EM's operational model is closed-source). | Open contribution |
| **Contribution 2** | Feature ablation on interconnector flows. | Ablation contribution |
| **Contribution 3** | Per-region heterogeneity atlas across the five target zones. | Heterogeneity contribution |
| **Contribution 4** | Head-to-head benchmark against EM's operational consumption-based forecast, with per-horizon and per-failure-mode characterisation. | Benchmark contribution |
| **M3a** | The locked full failure-mode breakdown across all six axes. | Full breakdown |
| **M3b** | A trimmed failure-mode breakdown limited to per-horizon, per-region, per-season. The fallback if Week 7 hits timeline pressure. | Trimmed breakdown |
| **P5b** | The locked weather data approach: Open-Meteo hybrid (ERA5 historical, GFS forecast). | Open-Meteo hybrid (informal alias OK) |
| **P5a** | Pure NCEP GFS weather pipeline (CarbonCast-faithful). Deferred as a stretch ablation. | GFS-only weather |
| **Y1** | The locked Overleaf sync workflow: local repo as source of truth, manual drag-drop to Overleaf at checkpoints. | One-way sync (ambiguous), local-first |
| **Locked decision** | A decision committed to project memory after explicit user confirmation. Treated as immovable unless new evidence invalidates a premise. | Confirmed decision, agreed decision |
| **Stretch experiment** | An optional experiment scheduled only if its parent week finishes ahead of plan. Examples: 24h-window ablation, pure NCEP GFS weather, full seq2seq variant. | Bonus experiment, stretch goal |
| **Grilling** | The structured interview that produced these decisions, walking the decision tree branch by branch via the `/grill-me` skill. | Q&A, design review |

## Codebase and Tooling

| Term | Definition | Aliases to avoid |
|---|---|---|
| **`carbon_forecast`** | The installable Python package; lives under `src/carbon_forecast/`. | The codebase (vague) |
| **Notebook session** | A session-numbered notebook (S01 through S07) representing one stage of the linear exploration arc. | Notebook (acceptable when unambiguous) |
| **Lit note** | A markdown file under `docs/lit-notes/`, one per paper, summarising it via a 6-bullet template (problem, method, dataset, result, gap, relevance). | Paper summary, reading note |
| **Proposal** | The 3-to-4-page acmart-sigconf document under `reports/proposal/`, intended for the supervisor. Not the main report. | Pitch (informal), brief |
| **Report** | The full 30-plus-page acmart-sigconf academic report. The primary deliverable. Lives under `reports/report/`. | Paper (we are not submitting), thesis |
| **Poster** | The A0 portrait Figma artifact exported to PDF, summarising the contribution and key results. | Slide, summary |

## Key Relationships

- A **Zone** has zero or more **Interconnectors** to **Partner zones**.
- The **Flow** on each **Interconnector** at each hour determines the gap between **Production-based CI** and **Consumption-based CI** for that zone.
- **E3** extends **E2** by adding per-interconnector **Tier 1** ANNs and retargeting **Tier 2** to **Consumption-based CI**.
- **Test B** is the only fold where the **EM operational forecast** is captured live, so **Contribution 4** can only be evaluated on **Test B**.
- **Failure-mode analysis** slices both **Test A** and **Test B** along each **Breakdown axis** to support **Contribution 4** with statistically defensible (Diebold-Mariano) per-slice claims.
- The four **Contributions** map roughly: Contribution 1 to the open **codebase** (`carbon_forecast`), Contribution 2 to the **flow ablation**, Contribution 3 to the **per-region heterogeneity atlas**, Contribution 4 to the **head-to-head benchmark**.

## Example dialogue

> **Dev:** When we evaluate **E3** on **Test B**, are we comparing against the **EM operational forecast** or against **E2**?

> **Domain expert:** Both. **Test B** is the head-to-head window where **EM** forecast snapshots are captured live. We compare **E3** against **EM** for **Contribution 4**, against **E2** as the same-zone **production-based** reference, and against classical baselines (seasonal naive, ARIMA, Prophet). All comparisons get **Diebold-Mariano** significance tests.

> **Dev:** And the per-horizon analysis just slices **Test B** at 1, 6, 12, 24, 48, 72, and 96 hours?

> **Domain expert:** Yes. The 1-to-72h slices include the **EM** comparison; the 96h slice has no **EM** baseline because **EM** publishes only up to 72h. So at 96h, **E3** is compared against classical baselines only.

> **Dev:** And the **flow ablation** runs on those same folds?

> **Domain expert:** Yes. The **flow ablation** is **E3** with the per-interconnector **Tier 1** ANNs removed from **Tier 2**'s inputs. It directly tests whether explicit cross-border **flow** modeling improves **consumption-based CI** forecast accuracy, which is the empirical claim behind **Contribution 2**.

## Flagged ambiguities

- **"Phase"** was used in two senses during the grilling. **Project phase** refers to the three-part program (Phase 1 forecasting, Phase 2 optimization, Phase 3 simulation). The two-stage architecture was sometimes called "phase" early on. **Canonical:** keep **"phase"** for the project's three-part program. Use **"tier"** for the two-stage architecture.
- **"Region"** versus **"Zone"**. Electricity Maps uses **"zone"** as the API identifier. The literature often uses **"region"**. **Canonical:** **"zone"** when identifying an EM electrical region; **"region"** only as informal geographic context.
- **"Forecast"** can mean our model's prediction or EM's operational forecast. **Canonical:** **"open forecast"** for our model output when disambiguation matters; **"EM operational forecast"** for the benchmark target.
- **"Production"** is overloaded. It can mean electricity generation (per-source production, generation mix) or operational deployment (production-tier API, productionised model). **Canonical:** prefer **"generation"** for electricity output; keep **"production-based CI"** as a fixed CI-specific term; use **"operational"** for deployment context.
- **"Test"** is overloaded. **Test A** is the May 2026 hold-out fold; **Test B** is the June 2026 head-to-head window; **pytest** tests are code-level unit tests. **Canonical:** **"test fold"** with **A** or **B** for evaluation data; **"unit test"** or **"pytest"** for code-level tests.
- **"Ablation"** has been used both for the **flow ablation** (toggling Tier 1 flow inputs) and for the **24h input-window ablation**. Both are valid ablations. Specify the target of the ablation (flow, window, weather source) when referring to it.
- **"Stretch experiment"** versus **"ablation"** are orthogonal. An **ablation** is a methodology: run with and without a component to measure its effect. A **stretch experiment** is a scope decision: run only if time allows. The 24h-window ablation is a stretch experiment that happens to be structured as an ablation. The flow ablation is core (not stretch), even though it is methodologically an ablation.
- **"Open"** in **"open architecture"** means open-source and reproducible, not "open" in the geographical sense. Use **"open-source"** when the meaning is the code is publicly available; reserve **"open architecture"** for the broader claim that includes modularity and reproducibility.
