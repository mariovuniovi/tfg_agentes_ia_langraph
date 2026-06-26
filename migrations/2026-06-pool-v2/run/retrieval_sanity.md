# Retrieval sanity check (pool migration v2)

## query_alpha (~500 monthly, no seasonality, 0 exog)
  1. brent_oil                                  sim=0.370
  2. crypto_weekly                              sim=0.370
  3. sunspots                                   sim=0.370

## query_beta (~3000 daily, mild weekly seas, 2 unknown-future exog, high vol)
  1. bike_sharing_daily                         sim=0.333
  2. appliances_energy                          sim=0.296
  3. ppi_chemicals                              sim=0.259

## query_gamma (~250 monthly, mild annual seas, 1 unknown-future exog)
  1. bike_sharing_daily                         sim=0.370
  2. co2_mauna_loa                              sim=0.370
  3. air_passengers                             sim=0.370


## Interpretation (surfaced for human review — retrieval weights NOT tuned)

The new datasets are seeded and retrievable, but they do NOT rank above the
retained datasets for the chemical-industry-like queries. This is consistent
with the design (structural, not thematic, retrieval), not a defect:

- henry_hub_gas / ppi_chemicals profile as `very_small` MONTHLY SEASONAL — the
  same structural cluster as air_passengers / co2_mauna_loa. Retrieval groups
  them together rather than privileging the new ones.
- brent_oil profiles as `medium` / freq `B` / no-seasonality — clusters with the
  daily series, away from the monthly ones.
- Similarity scores are coarse (many ties ~0.37) because weighted-overlap runs
  over a handful of bucketed fields.

Per the migration plan's Step 7 rule, retrieval weights were left unchanged.

### Genuine observation (possible profiler gap, out of migration scope)
`exog_future_availability` is None in EVERY stored forecasting profile, including
datasets that declare exogenous columns (ppi_chemicals, oil_weekly, gold_macro,
fx_weekly, sp500_weekly). The profiler does not appear to populate this field,
which weakens exogenous-based retrieval matching. Flagged for human decision.
