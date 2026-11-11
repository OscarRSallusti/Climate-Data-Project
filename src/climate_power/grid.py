"""Grid fundamentals: hourly generation by source, load, and installed capacity.

Source: api.energy-charts.info (Fraunhofer ISE; ENTSO-E, SMARD and the
Marktstammdatenregister underneath). Generation and load start in 2015,
installed capacity is monthly from 2002.

These are what turn weather into prices: weather sets the capacity factor,
capacity scales it to generation, and load minus wind and solar (residual load)
is what the price-setting plants have to cover.
"""

from __future__ import annotations

import time

import pandas as pd

from .config import Config
from .http import get_json

BASE = "https://api.energy-charts.info/"


def update_generation(cfg: Config) -> pd.DataFrame:
    """Hourly mean power in MW per source, plus load. One cached file per year."""
    country = cfg.raw["grid"]["country"]
    cache = cfg.data_dir / "grid_chunks"
    cache.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    frames = []
    for year in range(pd.Timestamp(cfg.raw["grid"]["start"]).year, today.year + 1):
        path = cache / f"{year}.parquet"
        settled = today > pd.Timestamp(year + 1, 2, 1)  # recent values still get revised
        if path.exists() and settled:
            frames.append(pd.read_parquet(path))
            continue
        print(f"generation {country} {year}", flush=True)
        end = min(pd.Timestamp(year, 12, 31), today)
        d = get_json(BASE + "public_power", {"country": country, "start": f"{year}-01-01", "end": f"{end:%Y-%m-%d}"})
        index = pd.to_datetime(d["unix_seconds"], unit="s", utc=True)
        df = pd.DataFrame({p["name"]: p["data"] for p in d["production_types"]}, index=index).astype("float64")
        df = df.groupby(df.index.floor("h")).mean()
        df.index.name = "time"
        df.to_parquet(path)
        frames.append(df)
        time.sleep(5)
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.to_parquet(cfg.data_dir / "generation.parquet")
    return df


def update_capacity(cfg: Config) -> pd.DataFrame:
    """Installed capacity in GW at each month start."""
    d = get_json(
        BASE + "installed_power",
        {"country": cfg.raw["grid"]["country"], "time_step": "monthly", "installation_decommission": "false"},
    )
    index = pd.to_datetime(d["time"], format="%m.%Y")
    df = pd.DataFrame({p["name"]: p["data"] for p in d["production_types"]}, index=index).astype("float64")
    df.index.name = "month"
    df.to_parquet(cfg.data_dir / "capacity.parquet")
    return df


def load_generation(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.data_dir / "generation.parquet")


def load_capacity(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.data_dir / "capacity.parquet")
