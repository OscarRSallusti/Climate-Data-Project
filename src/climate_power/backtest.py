"""Train every model on the past and score it on a held-out test period."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import plots
from .config import Config
from .features import Samples, build_master, build_samples, complete, origin_positions
from .models.gbm import GBM
from .prices import load_prices
from .weather import load_actual, load_forecast_archive

UNIT = "EUR/MWh"


def split_masks(cfg: Config, s: Samples) -> dict[str, np.ndarray]:
    horizon = pd.Timedelta(hours=cfg.horizon)
    val_start = pd.Timestamp(cfg.raw["split"]["val_start"], tz="UTC")
    test_start = pd.Timestamp(cfg.raw["split"]["test_start"], tz="UTC")
    ok = complete(s)
    o = s.origins
    return {
        # A sample's targets reach 14 days past its origin, so each set must end
        # that long before the next one starts.
        "train": ok & (o + horizon <= val_start),
        "val": ok & (o >= val_start) & (o + horizon <= test_start),
        "test": ok & (o >= test_start),
    }


def run_backtest(cfg: Config, with_lstm: bool = True) -> pd.DataFrame:
    master = build_master(cfg, load_prices(cfg), load_actual(cfg), load_forecast_archive(cfg))
    pos = origin_positions(master, cfg)
    avail = build_samples(master, cfg, pos, "asavail")
    oracle = build_samples(master, cfg, pos, "oracle")
    masks = split_masks(cfg, avail)
    for name, mask in masks.items():
        o = avail.origins[mask]
        print(f"{name}: {mask.sum()} forecast origins, {o[0].date()} -> {o[-1].date()}", flush=True)

    tr, va, te = (avail.subset(masks[k]) for k in ("train", "val", "test"))
    o_tr, o_va, o_te = (oracle.subset(masks[k]) for k in ("train", "val", "test"))

    preds = {"naive_week": te.denorm(te.naive), "profile_7d": te.denorm(te.profile)}
    models = {}
    print("fitting gbm_calendar", flush=True)
    models["gbm_calendar"] = GBM(use_weather=False).fit(tr, va)
    preds["gbm_calendar"] = te.denorm(models["gbm_calendar"].predict(te))
    print("fitting gbm_weather", flush=True)
    models["gbm_weather"] = GBM(use_weather=True).fit(tr, va)
    preds["gbm_weather"] = te.denorm(models["gbm_weather"].predict(te))
    print("fitting gbm_oracle", flush=True)
    gbm_oracle = GBM(use_weather=True).fit(o_tr, o_va)
    preds["gbm_oracle"] = o_te.denorm(gbm_oracle.predict(o_te))
    if with_lstm:
        from .models.lstm import LSTMForecaster

        print("fitting lstm", flush=True)
        models["lstm"] = LSTMForecaster().fit(tr, va)
        preds["lstm"] = te.denorm(models["lstm"].predict(te))

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, cfg.model_dir / "models.joblib")

    # Scores
    # Score every lead on the same delivery days (those forecast at all 14 leads),
    # otherwise an unusual week at either end of the test period skews the comparison.
    days = cfg.horizon // 24
    n = len(te.origins)
    gaps = pd.Series(te.origins).diff().dropna()
    daily_origins = gaps.between(pd.Timedelta(hours=23), pd.Timedelta(hours=25)).all()
    target = np.arange(n)[:, None] + np.arange(days)[None]  # delivery-day number of [origin, lead]
    common = (target >= days - 1) & (target <= n - 1) if daily_origins else np.ones((n, days), bool)
    w = np.broadcast_to(common[..., None], (n, days, 24))

    def mean(x, lead=slice(None)):
        return x[:, lead][w[:, lead]].mean()

    err = {name: np.abs(p - te.price).reshape(n, days, 24) for name, p in preds.items()}
    by_lead = pd.DataFrame(
        {name: [mean(e, slice(k, k + 1)) for k in range(days)] for name, e in err.items()}, index=range(1, days + 1)
    )
    by_lead.index.name = "days_ahead"
    overall = pd.DataFrame(
        {
            "MAE": {name: mean(e) for name, e in err.items()},
            f"MAE days 1-{cfg.max_lead}": {name: mean(e, slice(0, cfg.max_lead)) for name, e in err.items()},
            f"MAE days {cfg.max_lead + 1}-{days}": {name: mean(e, slice(cfg.max_lead, days)) for name, e in err.items()},
            "RMSE": {name: np.sqrt(mean(e**2)) for name, e in err.items()},
        }
    )
    overall["MAE vs naive"] = overall["MAE"] / overall.loc["naive_week", "MAE"]
    overall = overall.sort_values("MAE")

    out = cfg.report_dir
    out.mkdir(parents=True, exist_ok=True)
    by_lead.round(2).to_csv(out / "mae_by_lead.csv")
    overall.round(3).to_csv(out / "summary.csv")
    order = [c for c in plots.COLORS if c in by_lead.columns]
    plots.mae_by_lead(by_lead[order], UNIT, out / "mae_by_lead.png", cfg.max_lead)

    # Forecast revisions for one delivery day: the one whose forecast was revised
    # most between two weeks out and the day before (chosen without looking at the outcome).
    if n > days and daily_origins:
        p = preds["gbm_weather"].reshape(n, days, 24)
        t = max(range(days - 1, n), key=lambda i: abs(p[i, 0].mean() - p[i - days + 1, days - 1].mean()))
        tz_day = te.origins[t].tz_convert(cfg.timezone)
        plots.revisions(
            te.price[t, :24], {k: p[t - k + 1, k - 1] for k in range(1, days + 1)}, tz_day, UNIT, out / "revisions.png"
        )

    # One full 14-day forecast from the middle of the test period.
    i = n // 2
    times = pd.date_range(te.origins[i], periods=cfg.horizon, freq="h").tz_convert(cfg.timezone)
    hist = pd.Series(master.price, index=master.index.tz_convert(cfg.timezone))
    hist = hist[times[0] - pd.Timedelta(days=7) : times[-1]]
    shown = {k: preds[k][i] for k in ("gbm_weather", "lstm", "gbm_calendar") if k in preds}
    plots.forecast_plot(
        times, shown, hist, UNIT, f"14-day forecast issued for {times[0]:%d %b %Y} onward", out / "example_forecast.png",
        split=times[cfg.max_lead * 24],
    )

    pd.set_option("display.width", 200)
    print("\nTest-period error by lead (MAE, EUR/MWh):\n", by_lead.round(1).to_string())
    print("\nSummary:\n", overall.round(2).to_string())
    return overall
