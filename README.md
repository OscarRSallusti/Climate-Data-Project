# Climate-Data-Project

Forecasting hourly day-ahead electricity prices up to two weeks ahead from weather,
and measuring how the forecast improves as newer weather forecasts arrive.

Current market: **Germany-Luxembourg (DE-LU)**. Everything market-specific lives in
`config/de_lu.yaml`.

## Results so far

Test period: forecasts issued daily from March to October 2026 (242 origins), models
trained on April 2024 to October 2025. Mean absolute error in EUR/MWh; the mean
price over 2026 was about 120.

| Model | All 14 days | Days 1-6 | Days 7-14 | vs naive |
|---|---|---|---|---|
| GBM + observed weather (upper bound) | 21.9 | 21.9 | 22.0 | 0.50 |
| **LSTM + weather forecast** | 30.7 | 27.1 | 33.4 | 0.70 |
| **GBM + weather forecast** | 31.2 | 28.4 | 33.2 | 0.71 |
| GBM, calendar and price history only | 34.6 | 34.0 | 35.1 | 0.79 |
| 7-day hourly profile | 40.5 | 40.2 | 40.7 | 0.92 |
| Same hour last week | 43.9 | 45.5 | 42.7 | 1.00 |

![Error by lead time](reports/de_lu/mae_by_lead.png)

- Weather carries most of the predictable signal. With perfect weather knowledge
  the error is half the naive error at every lead.
- With real forecasts, the benefit is large for the next 1-3 days and fades by
  day 6. The LSTM beats the tree model at short leads.
- Days 7-14 are close to weather-blind in this backtest. That is a data
  limitation, not a model result: see "Known limitations".

How one forecast changed as delivery approached (the test day with the largest
revision, picked without looking at the outcome):

![Forecast revisions](reports/de_lu/revisions.png)

## How it works

**Data** (all free, no API keys):

| What | Source | Coverage |
|---|---|---|
| Day-ahead prices | [energy-charts.info](https://api.energy-charts.info) (Fraunhofer ISE, from SMARD) | hourly from Jan 2015 (joint zone with Austria until Sep 2018); 15-minute prices since Oct 2025 are averaged to hourly |
| Generation by source, load | energy-charts.info | hourly from Jan 2015 |
| Installed wind, solar, battery capacity | energy-charts.info | monthly from 2002 |
| Observed weather | [Open-Meteo](https://open-meteo.com) archive API (reanalysis) | hourly from Jan 2015 (available back to the 1940s) |
| Past and live weather forecasts (default) | [NOAA GEFS via dynamical.org](https://dynamical.org/catalog/noaa-gefs-forecast-35-day/) (Icechunk/Zarr on AWS Open Data, CC BY 4.0) | one 31-member ensemble run per day (00 UTC) from Oct 2020, 16 days ahead |
| Past weather forecasts (alternative) | Open-Meteo previous-runs API, GFS | what the model predicted 1-7 days before each hour, from Apr 2024 |
| Live weather forecast (alternative) | Open-Meteo forecast API, GFS | 16 days ahead; every download is kept as a snapshot |

**Forecast source.** `weather.forecast_source` in the config (or `--source`)
picks where weather forecasts come from:

- `gefs` (default): the NOAA GEFS ensemble. The store is read point by point, so
  only the chunks around the 14 locations are downloaded (about 40 MB and 20 s per
  daily run; the full archive from Oct 2020 takes around 13 hours on a 2 MB/s
  connection and under 1 GB on disk). Each run is reduced to the four national
  drivers per member, then to the ensemble mean (the model input) and spread
  (stored for later use). Runs reach 16 days, so every target day sees a real
  forecast, and training can start in October 2020.
- `open_meteo`: the previous-runs archive, from April 2024 and only up to 7 days
  ahead, so target days 7-14 fall back to climatology.

Generation, load and capacity are collected but not yet used by the models.

Weather is pulled at 14 points and reduced to four national drivers: a wind
capacity-factor proxy (100 m wind through a generic turbine power curve, six
northern sites), solar irradiance (five southern/eastern sites), temperature at six
load centres, and precipitation.

**Forecast setup.** A forecast is made on the morning of day D, before the noon
auction. It knows prices through day D, observed weather through D-1, and weather
forecasts issued by the end of D-1. It predicts the 336 hourly prices of days
D+1 to D+14. For target day k the model sees the forecast issued k+1 days before;
with GEFS that is the 00 UTC run of day D-1 for every target day. Nothing in the
backtest uses information that was not available at the time
(`tests/test_features.py` and `tests/test_gefs.py` check this).

**Models.**

- `gbm_weather`: gradient-boosted trees, one pooled model over all leads.
- `lstm`: encoder-decoder LSTM. The encoder reads the last week of prices and
  weather; the decoder steps through the 336 target hours with the expected
  weather and calendar. Average of three seeds.
- `gbm_calendar`: the tree model without weather, to isolate what weather adds.
- `gbm_oracle`: the tree model given observed weather for all 14 days, an upper
  bound on what better weather forecasts could deliver.

All models predict a correction to the last week's hour-of-day price profile, and
prices are normalised per forecast by the median and spread of the previous 28
days so that changes in the overall price level do not break them.

## Usage

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run climate-power fetch      # download or refresh all data (first run: ~10 min, plus ~13 h for the GEFS archive)
uv run climate-power backtest   # train, evaluate, write reports/de_lu/
uv run climate-power forecast   # live 14-day forecast from the latest data
uv run pytest
```

The GEFS download can be interrupted and resumed: every finished run is stored
under `data/de_lu/gefs/raw/`, and the next `fetch` only gets what is missing.

Comparison runs: `--source open_meteo` switches the forecast source,
`--train-start YYYY-MM-DD` shortens the training window, and `--tag NAME` writes
models and reports to `models/de_lu/NAME/` and `reports/de_lu/NAME/` so they do
not overwrite the main run. For example:

```bash
uv run climate-power --source open_meteo fetch
uv run climate-power --source open_meteo --tag open_meteo backtest
uv run climate-power --source open_meteo snapshot   # archive the current Open-Meteo 16-day forecast
```

The LSTM trains on a CUDA GPU when PyTorch can see one, otherwise on the CPU.
Data and trained models are written to `data/` and `models/` and are not committed.

## Known limitations

- **Forecast history starts in October 2020.** Prices and observed weather go
  back to 2015, but the GEFS store starts with the current model version (GEFSv12)
  in October 2020. The same model version was re-run for 2000-2019 (the GEFSv12
  reforecast, 5 members), but only as global GRIB files on AWS, about 300 MB per
  day for the variables used here. With the Open-Meteo source, forecasts start in
  April 2024 and stop at 7 days ahead.
- **Conservative lead.** Using the run of day D-1 is up to a day staler than what
  is really available on the morning of day D (the 00 UTC run of day D is
  published by about 06 UTC).
- **Grid points, not interpolation.** GEFS values are taken at the nearest grid
  point (0.25 degrees; 100 m wind comes from the 0.5 degree grid), while observed
  weather comes from Open-Meteo's reanalysis. The models learn the mapping, but
  forecast and observation are not exactly the same quantity.
- **Single train/test split.** No rolling re-training yet, and the live command
  uses the models from the backtest rather than ones refitted on all data.
- **Point forecasts only.** No prediction intervals, and price spikes are
  under-predicted.
- **Weather only.** Gas and carbon prices, plant outages and cross-border flows
  are not modelled; the 28-day normalisation absorbs their slow movements only.
- **ERCOT is not included.** ERCOT's website and API reject connections from
  outside the US. Adding it needs a US connection or a third-party source such as
  gridstatus.io (free key), plus a Texas location list in a new config file.

## Ideas for next steps

- Bring back the Fourier representative-day model as a baseline and as features.
- Predict wind, solar and load first, then price from residual load. This stage
  needs only observed weather, so it can train on the full history from 2015 with
  installed capacity as an input.
- Extend the forecast history to 2015 with the GEFSv12 reforecast (same model
  version as the operational archive), reading only the needed GRIB messages via
  their `.idx` byte ranges.
- Quantile outputs for prediction intervals.
- Feed the ensemble spread (already stored as `*_sd` columns in
  `data/de_lu/gefs/features.parquet`) to the models, so they see forecast
  uncertainty.
