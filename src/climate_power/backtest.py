"""Train every model on the past and score it on a held-out test period."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import plots
from .config import Config
from .features import SPREAD, WX, Samples, build_master, build_samples, complete, origin_positions
from .models.gbm import GBM
from .prices import load_prices
from .weather import load_actual

UNIT = "EUR/MWh"
SHORT = 6  # summary columns split the horizon here, the same for every source so runs stay comparable


def forecast_inputs(cfg: Config) -> dict[int, pd.DataFrame]:
    """{N: national drivers of the forecast issued N days before each hour}, from the active source."""
    if cfg.source == "gefs":
        from . import gefs

        columns = WX + SPREAD if cfg.use_spread else WX
        return gefs.lead_frames(gefs.load_features(cfg), cfg.max_lead + 1, columns)
    from .weather import forecast_frames, load_forecast_archive

    return forecast_frames(cfg, load_forecast_archive(cfg))


ENSEMBLE_OF = ("lstm", "gbm_weather")


def ensemble(preds: dict[str, np.ndarray]) -> np.ndarray:
    """Equal-weight average of the two weather models: their errors are only partly shared."""
    return np.mean([preds[k] for k in ENSEMBLE_OF], axis=0)


def save_predictions(cfg: Config, splits: dict[str, tuple[Samples, dict[str, np.ndarray]]]) -> None:
    """Validation and test forecasts in long form, so runs with different sources can be combined later
    (see `combine`): combinations are chosen on the validation period and scored on the test period."""
    parts = []
    for split, (s, preds) in splits.items():
        n, h = s.price.shape
        parts.append(
            pd.DataFrame(
                {
                    "split": split,
                    "origin": np.repeat(s.origins, h),
                    "lead_hour": np.tile(np.arange(h, dtype="int16"), n),
                    "price": s.price.reshape(-1),
                    **{name: p.reshape(-1).astype("float32") for name, p in preds.items()},
                }
            )
        )
    pd.concat(parts, ignore_index=True).to_parquet(cfg.model_dir / "predictions.parquet")
    old = cfg.model_dir / "test_predictions.parquet"  # earlier, test-only layout
    old.unlink(missing_ok=True)


def score(origins: pd.DatetimeIndex, price: np.ndarray, preds: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MAE by lead day and summary table.

    Every lead is scored on the same delivery days (those forecast at all lead days),
    otherwise an unusual week at either end of the period skews the comparison.
    """
    n, horizon = price.shape
    days = horizon // 24
    gaps = pd.Series(origins).diff().dropna()
    daily_origins = gaps.between(pd.Timedelta(hours=23), pd.Timedelta(hours=25)).all()
    target = np.arange(n)[:, None] + np.arange(days)[None]  # delivery-day number of [origin, lead]
    common = (target >= days - 1) & (target <= n - 1) if daily_origins else np.ones((n, days), bool)
    w = np.broadcast_to(common[..., None], (n, days, 24))

    def mean(x, lead=slice(None)):
        return x[:, lead][w[:, lead]].mean()

    err = {name: np.abs(p - price).reshape(n, days, 24) for name, p in preds.items()}
    by_lead = pd.DataFrame(
        {name: [mean(e, slice(k, k + 1)) for k in range(days)] for name, e in err.items()}, index=range(1, days + 1)
    )
    by_lead.index.name = "days_ahead"
    overall = pd.DataFrame(
        {
            "MAE": {name: mean(e) for name, e in err.items()},
            f"MAE days 1-{SHORT}": {name: mean(e, slice(0, SHORT)) for name, e in err.items()},
            f"MAE days {SHORT + 1}-{days}": {name: mean(e, slice(SHORT, days)) for name, e in err.items()},
            "RMSE": {name: np.sqrt(mean(e**2)) for name, e in err.items()},
        }
    )
    if "naive_week" in overall.index:
        overall["MAE vs naive"] = overall["MAE"] / overall.loc["naive_week", "MAE"]
    return by_lead, overall.sort_values("MAE")


def split_masks(cfg: Config, s: Samples) -> dict[str, np.ndarray]:
    horizon = pd.Timedelta(hours=cfg.horizon)
    val_start = pd.Timestamp(cfg.raw["split"]["val_start"], tz="UTC")
    test_start = pd.Timestamp(cfg.raw["split"]["test_start"], tz="UTC")
    ok = complete(s)
    o = s.origins
    return {
        # A sample's targets reach 14 days past its origin, so each set must end
        # that long before the next one starts.
        "train": ok & (o >= cfg.train_start) & (o + horizon <= val_start),
        "val": ok & (o >= val_start) & (o + horizon <= test_start),
        "test": ok & (o >= test_start),
    }


def run_backtest(cfg: Config, with_lstm: bool = True) -> pd.DataFrame:
    spread = ", with ensemble spread" if cfg.use_spread else ""
    print(f"forecast source: {cfg.source} (real forecasts for target days 1-{cfg.max_lead}{spread})", flush=True)
    master = build_master(cfg, load_prices(cfg), load_actual(cfg), forecast_inputs(cfg))
    pos = origin_positions(master, cfg)
    avail = build_samples(master, cfg, pos, "asavail")
    oracle = build_samples(master, cfg, pos, "oracle")
    masks = split_masks(cfg, avail)
    for name, mask in masks.items():
        o = avail.origins[mask]
        print(f"{name}: {mask.sum()} forecast origins, {o[0].date()} -> {o[-1].date()}", flush=True)

    tr, va, te = (avail.subset(masks[k]) for k in ("train", "val", "test"))
    o_tr, o_va, o_te = (oracle.subset(masks[k]) for k in ("train", "val", "test"))

    models = {}
    print("fitting gbm_calendar", flush=True)
    models["gbm_calendar"] = GBM(use_weather=False).fit(tr, va)
    print("fitting gbm_weather", flush=True)
    models["gbm_weather"] = GBM(use_weather=True).fit(tr, va)
    print("fitting gbm_oracle", flush=True)
    gbm_oracle = GBM(use_weather=True).fit(o_tr, o_va)
    if with_lstm:
        from .models.lstm import LSTMForecaster

        print("fitting lstm", flush=True)
        models["lstm"] = LSTMForecaster().fit(tr, va)

    def predict(s: Samples, o: Samples) -> dict[str, np.ndarray]:
        out = {"naive_week": s.denorm(s.naive), "profile_7d": s.denorm(s.profile)}
        out.update({name: s.denorm(m.predict(s)) for name, m in models.items()})
        out["gbm_oracle"] = o.denorm(gbm_oracle.predict(o))
        if with_lstm:
            out["ensemble"] = ensemble(out)
        return out

    preds = predict(te, o_te)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "source": cfg.source,
            "max_lead": cfg.max_lead,
            "use_spread": cfg.use_spread,
            "train_start": str(cfg.train_start.date()),
            "models": models,
        },
        cfg.model_dir / "models.joblib",
    )
    save_predictions(cfg, {"val": (va, predict(va, o_va)), "test": (te, preds)})

    days = cfg.horizon // 24
    n = len(te.origins)
    daily_origins = pd.Series(te.origins).diff().dropna().between(pd.Timedelta(hours=23), pd.Timedelta(hours=25)).all()
    by_lead, overall = score(te.origins, te.price, preds)

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
    shown = {k: preds[k][i] for k in ("ensemble", "gbm_weather", "lstm", "gbm_calendar") if k in preds}
    plots.forecast_plot(
        times, shown, hist, UNIT, f"14-day forecast issued for {times[0]:%d %b %Y} onward", out / "example_forecast.png",
        split=times[cfg.max_lead * 24] if cfg.max_lead < days else None,
    )

    pd.set_option("display.width", 200)
    print("\nTest-period error by lead (MAE, EUR/MWh):\n", by_lead.round(1).to_string())
    print("\nSummary:\n", overall.round(2).to_string())
    return overall
