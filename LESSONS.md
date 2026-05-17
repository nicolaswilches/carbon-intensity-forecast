# LESSONS

- `.env` must use `KEY=value` with no spaces. `KEY = value` breaks `bash source` (interpreted as command invocation). Fix with `sed -i.bak -E 's/^([A-Z_][A-Z0-9_]*)[[:space:]]*=[[:space:]]*/\1=/' .env`.
- Verify API tier before committing to a data plan. Electricity Maps sandbox keys return synthetic data with `"_disclaimer": "SANDBOX MODE..."` and `"estimationMethod": "SANDBOX_MODE_DATA"`; do not persist into `data/raw/`.
- Read the paper before claiming its architecture. CarbonCast Tier 2 is a learned CNN-LSTM consuming Tier 1 outputs plus historical CI plus weather, not deterministic emission-factor aggregation. The deterministic version is their non-hierarchical baseline (which they beat).
- For free Overleaf, git, GitHub, and Dropbox sync are Premium-only. Use Y1: drag-drop with local repo as source of truth.
- "Ablation" (a methodology: run with and without a component) and "stretch experiment" (a scope decision: run only if time allows) are orthogonal. Specify both axes when describing an experiment.
- The `/grill-me` skill produces a robust decision log when paired with memory writes after each lock-in. Do not skip the memory writes; the conversation context is lost across sessions, the memory is not.
