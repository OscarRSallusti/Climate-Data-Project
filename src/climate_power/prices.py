"""Hourly day-ahead prices.

Source: api.energy-charts.info (Fraunhofer ISE, data from SMARD / Bundesnetzagentur,
CC BY 4.0). The series is stitched from the bidding zones listed in the config:
Germany shared a zone with Austria until 2018-09-30. The German auction moved to
15-minute products on 2025-10-01; those are averaged to hourly.
"""

from __future__ import annotations

import time

import pandas as pd

from .config import Config
from .http import get_json

URL = "https://api.energy-charts.info/price"


def _fetch(zone: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Hourly prices for local delivery days start..end inclusive."""
    parts = []
    chunk = start
    while chunk <= end:
        chunk_end = min(chunk + pd.Timedelta(days=365), end)
        print(f"prices {zone} {chunk.date()} -> {chunk_end.date()}", flush=True)
        d = get_json(URL, {"bzn": zone, "start": f"{chunk:%Y-%m-%d}", "end": f"{chunk_end:%Y-%m-%d}"})
        if d.get("unix_seconds"):
            s = pd.Series(d["price"], index=pd.to_datetime(d["unix_seconds"], unit="s", utc=True), dtype="float64")
            parts.append(s.groupby(s.index.floor("h")).mean())
        chunk = chunk_end + pd.Timedelta(days=1)
        time.sleep(2)
    return pd.concat(parts) if parts else pd.Series(dtype="float64")


def price_path(cfg: Config):
    return cfg.data_dir / "prices.parquet"


def update_prices(cfg: Config) -> pd.Series:
    """Download whatever is missing, refresh the last week, and return the full series."""
    path = price_path(cfg)
    old = load_prices(cfg) if path.exists() else pd.Series(dtype="float64")
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    have_from = old.index[0].tz_convert(cfg.timezone).tz_localize(None).normalize() if len(old) else None
    have_to = old.index[-1].tz_localize(None).normalize() if len(old) else None

    parts = [old]
    for seg in cfg.raw["prices"]["zones"]:
        start = pd.Timestamp(seg["start"])
        end = pd.Timestamp(seg["end"]) if seg.get("end") else today + pd.Timedelta(days=2)
        if have_from is None:
            parts.append(_fetch(seg["zone"], start, end))
            continue
        if start < have_from:  # history older than what is stored
            parts.append(_fetch(seg["zone"], start, min(end, have_from)))
        refresh = max(start, have_to - pd.Timedelta(days=7))
        if refresh <= end:  # the recent tail
            parts.append(_fetch(seg["zone"], refresh, end))

    s = pd.concat([p for p in parts if len(p)])
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    s = s.rename("price")
    s.index.name = "time"
    path.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(path)
    return s


def load_prices(cfg: Config) -> pd.Series:
    return pd.read_parquet(price_path(cfg))["price"]
