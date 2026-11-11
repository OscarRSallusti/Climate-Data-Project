from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


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

    @property
    def data_dir(self) -> Path:
        return ROOT / "data" / self.market

    @property
    def model_dir(self) -> Path:
        return ROOT / "models" / self.market

    @property
    def report_dir(self) -> Path:
        return ROOT / "reports" / self.market

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
    def max_lead(self) -> int:
        return int(self.raw["model"]["max_forecast_lead_days"])

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
    return Config(raw=raw, market=raw["market"], timezone=raw["timezone"], locations=locations)
