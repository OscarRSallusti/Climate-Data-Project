"""Hourly weather from Open-Meteo (free for non-commercial use, no key).

Three views of the same variables at the same locations:

* actual    - reanalysis / analysis history (archive API)
* forecast  - what the forecast model said N days before each hour, N = 1..7
              (previous-runs API). This is what makes an honest backtest possible.
* live      - the current 16-day forecast (forecast API), also saved as a
              timestamped snapshot so a true long-lead forecast archive builds up.

Wide frames are indexed by UTC hour. Columns are "location|variable" for
actual/live and "location|variable|N" for the forecast archive.
"""

from __future__ import annotations

import time

import pandas as pd

from .config import Config
from .http import get_json

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
LEADS = range(1, 8)  # previous_day1 .. previous_day7
PAUSE = 8  # seconds between bulk requests, keeps us under the per-minute quota


def _request(cfg: Config, url: str, hourly: list[str], **extra) -> list[dict]:
    params = {
        "latitude": ",".join(str(loc.lat) for loc in cfg.locations),
        "longitude": ",".join(str(loc.lon) for loc in cfg.locations),
        "hourly": ",".join(hourly),
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        **extra,
    }
    out = get_json(url, params)
    return out if isinstance(out, list) else [out]


def _to_frame(cfg: Config, payload: list[dict], rename=None) -> pd.DataFrame:
    cols = {}
    index = None
    for loc, d in zip(cfg.locations, payload, strict=True):
        h = d["hourly"]
        index = pd.to_datetime(h["time"], utc=True)
        for key, values in h.items():
            if key == "time":
                continue
            name = rename(key) if rename else key
            cols[f"{loc.name}|{name}"] = pd.Series(values, index=index, dtype="float64")
    df = pd.DataFrame(cols, index=index)
    df.index.name = "time"
    return df


def _chunks(start: pd.Timestamp, end: pd.Timestamp, freq: str):
    edges = pd.date_range(start, end, freq=freq)
    edges = edges.insert(0, start) if len(edges) == 0 or edges[0] != start else edges
    for i, a in enumerate(edges):
        b = edges[i + 1] - pd.Timedelta(days=1) if i + 1 < len(edges) else end
        yield a, b


def _update(cfg: Config, kind: str, start: str, freq: str, fetch) -> pd.DataFrame:
    """Fetch `kind` in calendar chunks, caching finished chunks on disk."""
    cache = cfg.data_dir / "weather_chunks" / kind
    cache.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    frames = []
    for a, b in _chunks(pd.Timestamp(start), today, freq):
        path = cache / f"{a:%Y%m%d}_{b:%Y%m%d}.parquet"
        settled = b < today - pd.Timedelta(days=10)  # recent values still get revised
        if path.exists() and settled:
            frames.append(pd.read_parquet(path))
            continue
        print(f"weather {kind} {a.date()} -> {b.date()}", flush=True)
        df = fetch(a, b)
        df.to_parquet(path)
        frames.append(df)
        time.sleep(PAUSE)
    # Chunks that ended "today" on an earlier run overlap newer ones; keep the latest.
    for path in cache.glob("*.parquet"):
        a, b = (pd.Timestamp(x) for x in path.stem.split("_"))
        if not any(a == ca and b == cb for ca, cb in _chunks(pd.Timestamp(start), today, freq)):
            path.unlink()
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.to_parquet(cfg.data_dir / f"weather_{kind}.parquet")
    return df


def update_actual(cfg: Config) -> pd.DataFrame:
    def fetch(a, b):
        payload = _request(
            cfg, ARCHIVE_URL, cfg.variables, start_date=f"{a:%Y-%m-%d}", end_date=f"{b:%Y-%m-%d}"
        )
        return _to_frame(cfg, payload)

    return _update(cfg, "actual", cfg.raw["weather"]["actual_start"], "YS", fetch)


def update_forecast_archive(cfg: Config) -> pd.DataFrame:
    hourly = [f"{v}_previous_day{n}" for v in cfg.variables for n in LEADS]

    def rename(key: str) -> str:
        var, n = key.rsplit("_previous_day", 1)
        return f"{var}|{n}"

    def fetch(a, b):
        payload = _request(
            cfg,
            PREVIOUS_RUNS_URL,
            hourly,
            start_date=f"{a:%Y-%m-%d}",
            end_date=f"{b:%Y-%m-%d}",
            models=cfg.raw["weather"]["forecast_model"],
        )
        return _to_frame(cfg, payload, rename)

    return _update(cfg, "forecast", cfg.raw["weather"]["forecast_archive_start"], "MS", fetch)


def fetch_live_forecast(cfg: Config, past_days: int = 3, save_snapshot: bool = True) -> pd.DataFrame:
    """Current forecast out to 16 days. Each call is archived under data/.../snapshots."""
    payload = _request(
        cfg,
        FORECAST_URL,
        cfg.variables,
        forecast_days=16,
        past_days=past_days,
        models=cfg.raw["weather"]["forecast_model"],
    )
    df = _to_frame(cfg, payload)
    if save_snapshot:
        snap = cfg.data_dir / "snapshots"
        snap.mkdir(parents=True, exist_ok=True)
        issued = pd.Timestamp.now("UTC")
        df.to_parquet(snap / f"{issued:%Y%m%dT%H%M}Z.parquet")
    return df


def load_actual(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.data_dir / "weather_actual.parquet")


def load_forecast_archive(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.data_dir / "weather_forecast.parquet")
