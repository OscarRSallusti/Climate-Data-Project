from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("gefs", "open_meteo")


@dataclass(frozen=True)
class Location:
    name: str
    lat: float
    lon: float
    groups: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    raw: dict
    market: str
    timezone: str
    locations: tuple[Location, ...]
    source: str  # where weather forecasts come from, one of SOURCES
    tag: str | None = None  # keeps the outputs of comparison runs apart
    train_start_override: str | None = None

    def with_overrides(self, source: str | None = None, tag: str | None = None, train_start: str | None = None):
        if source is not None and source not in SOURCES:
            raise ValueError(f"unknown forecast source {source!r}, expected one of {SOURCES}")
        return replace(
            self,
            source=source or self.source,
            tag=tag or self.tag,
            train_start_override=train_start or self.train_start_override,
        )

    @property
    def data_dir(self) -> Path:
        return ROOT / "data" / self.market

    @property
    def model_dir(self) -> Path:
        base = ROOT / "models" / self.market
        return base / self.tag if self.tag else base

    @property
    def report_dir(self) -> Path:
        base = ROOT / "reports" / self.market
        return base / self.tag if self.tag else base

    @property
    def variables(self) -> list[str]:
        return list(self.raw["weather"]["variables"])

    @property
    def history(self) -> int:
        return int(self.raw["model"]["history_hours"])

    @property
    def horizon(self) -> int:
        return int(self.raw["model"]["horizon_hours"])

    @property
    def source_cfg(self) -> dict:
        return self.raw["weather"][self.source]

    @property
    def max_lead(self) -> int:
        """Last target day that sees a real weather forecast; later days use climatology."""
        return min(int(self.source_cfg["max_lead_days"]), self.horizon // 24)

    @property
    def archive_start(self) -> pd.Timestamp:
        """First day of the forecast archive of the active source (UTC)."""
        return pd.Timestamp(self.source_cfg["archive_start"], tz="UTC")

    @property
    def train_start(self) -> pd.Timestamp:
        """First forecast origin used for training.

        A forecast made on day D sees runs issued up to D-1 (one or two UTC days back
        for the first local hours), so origins from three days after the archive
        start have real forecasts in every input.
        """
        value = self.train_start_override or self.raw["split"].get("train_start")
        if value:
            return pd.Timestamp(value, tz="UTC")
        return self.archive_start + pd.Timedelta(days=3)

    def group(self, name: str) -> list[str]:
        return [loc.name for loc in self.locations if name in loc.groups]


def load_config(path: str | Path = "config/de_lu.yaml") -> Config:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    raw = yaml.safe_load(path.read_text())
    locations = tuple(
        Location(name, float(v["lat"]), float(v["lon"]), tuple(v["groups"]))
        for name, v in raw["weather"]["locations"].items()
    )
    source = raw["weather"].get("forecast_source", "open_meteo")
    if source not in SOURCES:
        raise ValueError(f"unknown forecast_source {source!r} in {path}, expected one of {SOURCES}")
    return Config(raw=raw, market=raw["market"], timezone=raw["timezone"], locations=locations, source=source)
