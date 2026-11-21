# Model comparison: what each model and setting is good and bad at

Companion to the [README](README.md). Covers every forecaster and configuration tried so
far for DE-LU day-ahead prices, what it does well, where it falls short, and what we
learned from it.

All numbers come from one test setup unless noted. Forecasts were issued daily from
1 March to 7 November 2026 (252 origins), and each predicts the 336 hourly prices of the
next 14 days. The validation period runs from 15 November 2025 to 14 February 2026
(92 origins). Errors are mean absolute errors in EUR/MWh. The mean price in the test
period was about 125. Every lead day is scored on the same delivery days. The reports
behind each number are in `reports/de_lu/<run>/`.

## At a glance

| Model | Weather forecast | Trained from | All 14 days | Days 1-6 | Days 7-14 | vs naive |
|---|---|---|---|---|---|---|
| GBM + observed weather (upper bound) | observed | Apr 2024 | 22.5 | 22.3 | 22.7 | 0.49 |
| **LSTM** | Open-Meteo | Apr 2024 | **31.6** | **27.8** | 34.5 | 0.68 |
| **LSTM + GBM average** | GEFS + spread | Oct 2020 | **31.6** | 27.8 | **34.5** | 0.68 |
| LSTM + GBM average | GEFS + spread | Apr 2024 | 31.6 | 27.8 | 34.5 | 0.68 |
| LSTM | GEFS + spread | Apr 2024 | 31.8 | 28.0 | 34.6 | 0.69 |
| Combination chosen on validation | three runs | mixed | 32.1 | 28.4 | 34.8 | 0.69 |
| GBM + weather | GEFS + spread | Oct 2020 | 32.2 | 28.8 | 34.7 | 0.69 |
| LSTM | GEFS + spread | Oct 2020 | 32.8 | 28.7 | 35.8 | 0.71 |
| GBM + weather | Open-Meteo | Apr 2024 | 33.2 | 30.2 | 35.4 | 0.72 |
| GBM, calendar and price history only | none | Apr 2024 | 35.6 | 34.3 | 36.6 | 0.77 |
| 7-day hourly profile | none | none | 42.4 | 41.4 | 43.1 | 0.92 |
| Same hour last week | none | none | 46.3 | 47.6 | 45.3 | 1.00 |

Three setups tie at 31.6, within 0.05 of each other, and that difference is noise. No
model gets close to the upper bound of 22.5. The gap of about 9 EUR/MWh is the cost of not
knowing the weather.

## The models

### Same hour last week (`naive_week`)
Repeats last week's prices.
- **Good:** needs nothing, can't break, and is the yardstick every other model is
  measured against ("vs naive").
- **Bad:** knows nothing about weather or holidays, and copies last week's price spikes
  forward.

### 7-day hourly profile (`profile_7d`)
The average price at each hour of the day over the last week, repeated for 14 days.
- **Good:** smooths out single bad days, giving 8% less error than the naive forecast.
  Every trained model predicts a correction to this profile, which keeps them close to
  recent prices.
- **Bad:** flat across the two weeks, so it can't see a windy weekend or a cold snap
  coming.

### GBM, calendar and price history only (`gbm_calendar`)
Gradient-boosted trees using the hour, weekday, holidays, season and recent price level,
but no weather.
- **Good:** measures what weather adds, since it is the same model as the weather GBM
  without the weather. It is fast, with no GPU needed, and holds up in winter: on validation
  it beat two of the weather GBMs.
- **Bad:** the worst trained model in spring to autumn (35.6–37.3), when solar output
  drives prices. It is sensitive to the training window: trained from 2020 it reaches 37.3,
  because the 2021–22 energy crisis teaches it the wrong price patterns.

### GBM + weather forecast (`gbm_weather`)
The same trees, plus the forecast wind, solar, temperature and precipitation for each
hour, their change from last week, daily averages, and (with GEFS) the ensemble spread.
- **Good:**
  - Stable: retraining gives the same model, and it hardly cares about the training
    window (32.2 trained from 2020, 32.6 from 2024).
  - Copes best with the longer GEFS history: in the run trained from 2020 it beats that
    run's LSTM overall (32.2 against 32.8).
  - Quick to train, and easy to inspect through its feature importances.
- **Bad:**
  - Weaker than the LSTM at short leads: 25.1 against 22.6 on day 1 with Open-Meteo.
  - Ignores the ensemble spread (no gain from it).
  - Worst of the weather models with Open-Meteo, whose forecasts stop at day 6.

### LSTM + weather forecast (`lstm`)
An encoder-decoder network: it reads the last week of prices and weather, then steps
through the 336 target hours with the forecast weather and calendar. It is the average of
three seeds.
- **Good:**
  - The best single model at short leads: 22.6 on day 1 and 26.6 on day 3 with
    Open-Meteo.
  - The only model that gains from the ensemble spread (0.3–0.6 less error).
  - Runs are reproducible here: rerunning a backtest gave identical numbers.
- **Bad:**
  - Needs a GPU to train comfortably and is harder to interpret.
  - Gets worse with more history: 32.8 trained from 2020 against 31.8 from 2024 with GEFS,
    because older price regimes confuse it more than they confuse the trees.
  - Long leads are no better than the trees: about 34–36 on days 10–14 whatever the source.

### LSTM + GBM average (`ensemble`)
An equal-weight average of the two weather models from the same run, reported by every
backtest.
- **Good:** with GEFS it is the most reliable choice. It improves on both of its parts
  for either training window (31.6 against 32.2 and 32.8 for the 2020 run), because the
  two models make different mistakes. It needs no extra training and no weights to tune.
- **Bad:** with Open-Meteo it is slightly worse than the LSTM on its own (31.8 against
  31.6), because the GBM is clearly the weaker partner there. It still costs two models
  to maintain.

### Combination of runs chosen on validation (`combine`)
A weighted average of models from several runs (Open-Meteo and GEFS sources, both
training windows, LSTM and both GBMs). The weights come from greedy forward selection on
the validation period. Used live with `forecast --combined`.
- **Good:** the right way to choose weights, because the test period is never used.
  On validation it cut the error from 18.2 (best single model) to 17.4.
- **Bad:** it didn't carry over to the test period: 32.1 against 31.6 for the best single
  model, and not better than equal weights (32.2). The validation period is three winter
  months, and model rankings change completely between winter and summer.
  - The GEFS LSTM trained from 2020 was best of the nine members on validation but fifth
    on test.
  - The calendar-only GBM got a fifth of the weight but was the worst model on test.
- **Verdict:** keep the tool, but don't rely on its weights until it can choose them on
  a validation period that covers every season (see "Next" below).

### GBM + observed weather (`gbm_oracle`)
The weather GBM given the observed weather for all 14 days. It can't be used for real
forecasts.
- **Good:** sets the ceiling. Perfect weather would halve the naive error at every lead
  (22.5), which shows weather is worth forecasting better.
- **Bad:** not a forecast. It also uses observed reanalysis data rather than forecasts,
  so the models' mapping from weather to price isn't quite the same thing.

## Settings compared

### Weather forecast source: Open-Meteo or GEFS

| | Open-Meteo previous runs (GFS) | NOAA GEFS ensemble (dynamical.org) |
|---|---|---|
| History | from Apr 2024 | from Oct 2020 |
| Reach | 7 days; target days 7–14 use climatology | 16 days; every target day gets a real forecast |
| Members | 1 (deterministic, higher resolution) | 31, giving mean and spread |
| Size and speed | small API calls | about 640 MB, about 9 h for the first download |
| Best LSTM, days 1-3 | **22.6 / 25.1 / 26.6** | 23.3 / 25.4 / 27.5 |
| Best LSTM, days 5-8 | 30.7 / 32.8 / 34.2 / 35.1 | **30.5 / 31.8 / 33.1 / 34.1** |
| Days 10-14 | 34.2–34.8 | 34.6–35.4 |

The best LSTM rows use the GEFS run trained from 2024, with spread.

- Open-Meteo is better for the first three days: a single higher-resolution run beats the
  ensemble mean.
- GEFS is better for days 5–8, where Open-Meteo has run out or is about to.
- Beyond about day 9, neither beats climatology. Weather forecasts that far out carry
  almost no price signal in this setup.
- Overall they tie (31.6–31.8). For the GBM, GEFS is clearly better (32.2–32.6 against
  33.2).

### Ensemble spread (GEFS only, on by default)
The spread of the 31 GEFS members, given to the models next to the mean. In the archive
it is clearly informative: hours in the top fifth by spread have 2–4 times the
forecast error of those in the bottom fifth.

| | Without spread | With spread |
|---|---|---|
| LSTM, trained from 2020 | 33.4 | **32.8** |
| LSTM, trained from 2024 | 32.1 | **31.8** |
| GBM, trained from 2020 | 32.2 | 32.2 |
| GBM, trained from 2024 | **32.5** | 32.6 |
| LSTM + GBM average, 2020 / 2024 | 32.0 / 31.8 | **31.6 / 31.6** |

- **Good:** a small but consistent gain for the LSTM and the average, at every lead,
  and it costs nothing (the spread is already computed).
- **Bad:** the trees don't use it. The gain is close to the size of the
  differences between seeds, so it is suggestive rather than proven.

### Training window: from October 2020 or April 2024 (GEFS only)

- **From 2020:** five years of data, including the 2021–22 gas crisis. Better for the GBM
  (32.2 against 32.6) and for the LSTM + GBM average (tied).
- **From 2024:** only the current price regime. Better for the LSTM (31.8 against 32.8)
  and much better for the calendar-only GBM (35.6 against 37.3).
- **Lesson:** more history only helps models that can't latch onto patterns from
  another price regime. The 28-day price normalisation doesn't fully remove the
  difference between crisis prices and today's.

## Tried and dropped

- **Blended source (Open-Meteo for days 1–6, GEFS after, GEFS spread throughout).**
  It was worse than either source alone (LSTM 32.9, GBM 32.5), even on days 1–6, where
  its inputs were almost the same as Open-Meteo's. The switch at day 7 and the extra inputs
  seem too much for 576 training forecasts. The code was removed.
- **Switching models at day 7** (Open-Meteo LSTM for days 1–6, GEFS LSTM after): 31.7,
  no better than the Open-Meteo LSTM on its own.
- **Pairs chosen by looking at the test period.** The Open-Meteo LSTM averaged with the
  GEFS GBM trained from 2020 scored 30.6, which looked like a clear win. It was one of 14
  combinations tried on the test period itself. Chosen properly on validation, the
  combination scores 32.1, so the 30.6 was a selection effect. It is listed here so it
  isn't rediscovered as a result.

## Recommendation for now

- **Live forecasts:** use the Open-Meteo LSTM
  (`climate-power --source open_meteo --tag open_meteo forecast`) or the GEFS LSTM + GBM
  average (the default run, `climate-power forecast`, column `ensemble`). They tie in
  the backtest. Open-Meteo is better for the first three days, GEFS for days 5–8.
- Treat forecasts beyond about day 9 as calendar-and-season estimates. The weather adds
  little there.

## Next

- **Rolling evaluation.** Retrain at several origins through 2024–2026, so models and
  combination weights are judged on every season rather than one winter. This is the
  main gap: without it, differences under about 0.5 EUR/MWh can't be told apart, and the
  combination can't be trusted.
- Predict wind, solar and load first, then price from residual load, so the long
  observed history from 2015 can be used without the GEFS limit.
- Quantile outputs, where the ensemble spread should matter more than it does for
  point forecasts.
