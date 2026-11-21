"""Forecast the next 14 days from the latest prices and the current weather forecast."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from . import plots
from .backtest import ENSEMBLE_OF, UNIT, forecast_inputs
from .config import Config
from .features import build_master, build_samples, usable, weather_features
from .prices import update_prices
from .weather import fetch_live_forecast, update_actual


@dataclass
class LiveData:
    """Latest inputs, downloaded once and shared by every run of a combined forecast."""

    cfg: Config
    price: pd.Series = field(init=False)
    actual: pd.DataFrame = field(init=False)
    _sources: dict = field(default_factory=dict)

    def __post_init__(self):
        self.price = update_prices(self.cfg)
        self.actual = update_actual(self.cfg)

    def weather(self, cfg: Config):
        """(forecast lead frames or None, live WX frame) for the run's source."""
        key = (cfg.source, cfg.use_spread)
        if key not in self._sources:
            if cfg.source == "gefs":
                from . import gefs

                if not any(s == "gefs" for s, _ in self._sources):
                    gefs.update_archive(cfg)
                feats = gefs.update_features(cfg)
                self._sources[key] = (forecast_inputs(cfg), gefs.latest_frame(feats))
            else:
                live = next((v[1] for (s, _), v in self._sources.items() if s == "open_meteo"), None)
                if live is None:
                    live = weather_features(cfg, fetch_live_forecast(cfg))  # also stored as a snapshot
                self._sources[key] = (None, live)
        return self._sources[key]

    def origin(self) -> pd.Timestamp:
        """First hour after the last local day with a complete set of prices."""
        local = self.price.index.tz_convert(self.cfg.timezone)
        last_day = local[-1].normalize() if local[-1].hour == 23 else local[-1].normalize() - pd.Timedelta(days=1)
        return (last_day + pd.Timedelta(days=1)).tz_convert("UTC")


def load_models(cfg: Config) -> dict:
    path = cfg.model_dir / "models.joblib"
    if not path.exists():
        raise SystemExit(f"No trained models in {cfg.model_dir}. Run `climate-power backtest` first.")
    saved = joblib.load(path)
    if "models" not in saved:  # saved before forecast sources existed
        saved = {"source": "open_meteo", "max_lead": 6, "models": saved}
    if saved["source"] != cfg.source or saved["max_lead"] != cfg.max_lead:
        raise SystemExit(
            f"The saved models were trained with forecast source {saved['source']!r}, "
            f"but the active source is {cfg.source!r}. Run `climate-power backtest` first."
        )
    if saved.get("use_spread", False) != cfg.use_spread:
        raise SystemExit(
            f"The saved models were trained {'with' if saved.get('use_spread') else 'without'} the ensemble spread, "
            f"but the active settings say {'with' if cfg.use_spread else 'without'}. "
            "Run `climate-power backtest` first, or match --no-spread."
        )
    return saved["models"]


def predict_latest(cfg: Config, data: LiveData, models: dict) -> pd.DataFrame:
    """Every model's 14-day forecast from the latest origin, one column per model."""
    forecast, live = data.weather(cfg)
    origin = data.origin()
    end = origin + pd.Timedelta(hours=cfg.horizon - 1)
    master = build_master(cfg, data.price, data.actual, forecast, live=live, end=end)
    pos = np.array([master.index.get_loc(origin)])
    s = build_samples(master, cfg, pos, "asavail")
    if not usable(s)[0]:
        raise SystemExit(f"Inputs for the latest origin are incomplete for source {cfg.source!r}.")
    times = pd.date_range(origin, periods=cfg.horizon, freq="h")
    return pd.DataFrame({name: s.denorm(m.predict(s))[0] for name, m in models.items()}, index=times)


def run_forecast(cfg: Config) -> pd.DataFrame:
    models = load_models(cfg)
    data = LiveData(cfg)
    out = predict_latest(cfg, data, models)
    if all(k in out for k in ENSEMBLE_OF):
        out["ensemble"] = out[list(ENSEMBLE_OF)].mean(axis=1)
    split = cfg.max_lead * 24 if cfg.max_lead * 24 < cfg.horizon else None
    return _write(cfg, out, data.price, "", split)


def run_combined_forecast(cfg: Config) -> pd.DataFrame:
    """Weighted combination of models from several runs, as chosen by `climate-power combine`."""
    from .combine import COMBINED_TAG

    combo = cfg.with_overrides(tag=COMBINED_TAG)
    path = combo.model_dir / "combination.json"
    if not path.exists():
        raise SystemExit("No combination found. Run `climate-power combine` first.")
    spec = json.loads(path.read_text())
    data = LiveData(cfg)
    columns, combined = {}, 0.0
    for member in spec["members"]:
        run_cfg = cfg.with_overrides(source=member["source"], tag=member["run"], spread=bool(member["use_spread"]))
        models = load_models(run_cfg)
        pred = predict_latest(run_cfg, data, {member["model"]: models[member["model"]]})[member["model"]]
        columns[f"{member['run']}:{member['model']}"] = pred
        combined = combined + member["weight"] * pred
    out = pd.DataFrame({"combined": combined, **columns})
    return _write(combo, out, data.price, "_combined", None)


def _write(cfg: Config, out: pd.DataFrame, price: pd.Series, suffix: str, split_hour: int | None) -> pd.DataFrame:
    out.index.name = "time_utc"
    out.insert(0, "days_ahead", np.arange(len(out)) // 24 + 1)
    times = out.index
    local_times = times.tz_convert(cfg.timezone)

    folder = cfg.data_dir / "forecasts"
    folder.mkdir(parents=True, exist_ok=True)
    out.round(2).to_csv(folder / f"{local_times[0]:%Y%m%d}{suffix}.csv")

    hist = price.tz_convert(cfg.timezone)
    hist = hist[local_times[0] - pd.Timedelta(days=7) :]
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    plots.forecast_plot(
        local_times,
        {name: out[name].to_numpy() for name in out.columns if name != "days_ahead"},
        hist,
        UNIT,
        f"Live 14-day forecast from {local_times[0]:%d %b %Y}",
        cfg.report_dir / "latest_forecast.png",
        split=local_times[split_hour] if split_hour is not None else None,
    )
    daily = out.drop(columns="days_ahead").groupby(out["days_ahead"]).mean().round(1)
    daily.insert(0, "date", [d.date() for d in local_times[::24]])
    print(f"\nDaily mean forecast ({UNIT}):\n", daily.to_string())
    return out
