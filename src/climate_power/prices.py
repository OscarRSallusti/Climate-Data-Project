"""Hourly day-ahead prices.

Source: api.energy-charts.info (Fraunhofer ISE, data from SMARD / Bundesnetzagentur,
CC BY 4.0). The German auction moved to 15-minute products on 2025-10-01; those
are averaged to hourly so the series is consistent over time.
"""

from __future__ import annotations

import time

import pandas as pd

from .config import Config
from .http import get_json

URL = "https://api.energy-charts.info/price"


def _fetch(zone: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    d = get_json(URL, {"bzn": zone, "start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")})
    if not d.get("unix_seconds"):
        return pd.Series(dtype="float64")
    s = pd.Series(d["price"], index=pd.to_datetime(d["unix_seconds"], unit="s", utc=True), dtype="float64")
    return s.groupby(s.index.floor("h")).mean()


def price_path(cfg: Config):
    return cfg.data_dir / "prices.parquet"


def update_prices(cfg: Config) -> pd.Series:
    """Download missing history, refresh the last week, and return the full series."""
    path = price_path(cfg)
    zone = cfg.raw["prices"]["bidding_zone"]
    old = load_prices(cfg) if path.exists() else pd.Series(dtype="float64")
    start = pd.Timestamp(cfg.raw["prices"]["start"])
    if len(old):
        start = old.index[-1].tz_localize(None).normalize() - pd.Timedelta(days=7)
    end = pd.Timestamp.now("UTC").tz_localize(None).normalize() + pd.Timedelta(days=2)

    parts = [old]
    chunk = start
    while chunk < end:
        chunk_end = min(chunk + pd.Timedelta(days=365), end)
        print(f"prices {zone} {chunk.date()} -> {chunk_end.date()}", flush=True)
        parts.append(_fetch(zone, chunk, chunk_end))
        chunk = chunk_end + pd.Timedelta(days=1)
        time.sleep(2)

    s = pd.concat([p for p in parts if len(p)])
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    s = s.rename("price")
    s.index.name = "time"
    path.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(path)
    return s


def load_prices(cfg: Config) -> pd.Series:
    return pd.read_parquet(price_path(cfg))["price"]
