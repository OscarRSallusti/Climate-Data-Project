"""Forecast the next 14 days from the latest prices and the current weather forecast."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import plots
from .backtest import UNIT
from .config import Config
from .features import build_master, build_samples, usable
from .prices import update_prices
from .weather import fetch_live_forecast, update_actual


def run_forecast(cfg: Config) -> pd.DataFrame:
    path = cfg.model_dir / "models.joblib"
    if not path.exists():
        raise SystemExit("No trained models found. Run `climate-power backtest` first.")
    models = joblib.load(path)

    price = update_prices(cfg)
    actual = update_actual(cfg)
    live = fetch_live_forecast(cfg)  # also stored as a snapshot

    # Origin = first hour after the last local day with a complete set of prices.
    local = price.index.tz_convert(cfg.timezone)
    last_day = local[-1].normalize() if local[-1].hour == 23 else local[-1].normalize() - pd.Timedelta(days=1)
    origin = (last_day + pd.Timedelta(days=1)).tz_convert("UTC")
    end = origin + pd.Timedelta(hours=cfg.horizon - 1)

    master = build_master(cfg, price, actual, None, live_wide=live, end=end)
    pos = np.array([master.index.get_loc(origin)])
    s = build_samples(master, cfg, pos, "asavail")
    if not usable(s)[0]:
        raise SystemExit("Inputs for the latest origin are incomplete (missing prices or weather).")

    times = pd.date_range(origin, periods=cfg.horizon, freq="h")
    out = pd.DataFrame({name: s.denorm(m.predict(s))[0] for name, m in models.items()}, index=times)
    out.index.name = "time_utc"
    out.insert(0, "days_ahead", np.arange(cfg.horizon) // 24 + 1)

    folder = cfg.data_dir / "forecasts"
    folder.mkdir(parents=True, exist_ok=True)
    out.round(2).to_csv(folder / f"{origin.tz_convert(cfg.timezone):%Y%m%d}.csv")

    local_times = times.tz_convert(cfg.timezone)
    hist = price.tz_convert(cfg.timezone)
    hist = hist[local_times[0] - pd.Timedelta(days=7) :]
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    plots.forecast_plot(
        local_times,
        {name: out[name].to_numpy() for name in models},
        hist,
        UNIT,
        f"Live 14-day forecast from {local_times[0]:%d %b %Y}",
        cfg.report_dir / "latest_forecast.png",
        split=local_times[cfg.max_lead * 24],
    )
    daily = out.drop(columns="days_ahead").groupby(out["days_ahead"]).mean().round(1)
    daily.insert(0, "date", [d.date() for d in local_times[::24]])
    print(f"\nDaily mean forecast ({UNIT}):\n", daily.to_string())
    return out
