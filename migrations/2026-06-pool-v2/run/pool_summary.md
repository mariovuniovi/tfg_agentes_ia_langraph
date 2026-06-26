# Forecasting pool - results & quality summary

| dataset | src | freq | n_obs | hist | h | exog | seas/trend | champion | RMSE | nRMSE | vs-naive |
|---|---|---|---|---|---|---|---|---|---|---|---|
| air_passengers | statsmode | MS | 144 | short | 12 | 0 | S/T | auto_arima | 23.56 | 0.196 | 71.2% |
| appliances_energy | uci_url | H | 3290 | long | 24 | 7 | S/- | seasonal_naive | 68.84 | 0.848 | 12.1% |
| bike_sharing_daily | uci_url | D | 731 | medium | 14 | 7 | S/T | ets | 1385.29 | 0.715 | 15.7% |
| brent_oil | fred | B | 3915 | long | 30 | 0 | -/T | lightgbm_forecaster | 4.29 | 0.173 | 4.0% |
| co2_mauna_loa | statsmode | MS | 526 | medium | 6 | 0 | S/T | ets | 0.33 | 0.019 | 90.2% |
| coal_feedstock_daily | yfinance_ | B | 2548 | long | 30 | 5 | -/T | naive | 9.49 | 0.130 | 0.0% |
| crypto_weekly | local | W | 335 | medium | 13 | 0 | -/T | random_forest_forecaster | 4652.87 | 0.255 | 18.6% |
| fx_weekly | local | W | 335 | medium | 13 | 3 | -/T | gbm_forecaster | 0.02 | 0.309 | 14.0% |
| gold_macro_weekly | yfinance_ | W | 1013 | medium | 13 | 7 | S/T | random_forest_forecaster | 65.79 | 0.150 | 3.2% |
| henry_hub_gas | fred | MS | 330 | medium | 12 | 0 | S/T | extra_trees_forecaster | 1.23 | 0.573 | 29.8% |
| nile | statsmode | YS | 100 | short | 3 | 0 | -/T | svr_forecaster | 114.53 | 0.677 | 30.6% |
| oil_weekly | yfinance_ | W | 1013 | medium | 13 | 7 | S/- | extra_trees_forecaster | 5.46 | 0.251 | 3.1% |
| ppi_chemicals | fred | MS | 474 | medium | 12 | 1 | S/T | extra_trees_forecaster | 11.16 | 0.147 | 1.2% |
| sp500_weekly | yfinance_ | W | 1011 | medium | 13 | 7 | S/T | xgboost_forecaster | 170.85 | 0.148 | 9.6% |
| sunspots | statsmode | YS | 309 | medium | 3 | 0 | -/T | svr_forecaster | 9.29 | 0.230 | 80.2% |
