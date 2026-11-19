"""Archived and live NOAA GEFS ensemble forecasts, from dynamical.org.

Source: NOAA GEFS 35-day forecast as processed by dynamical.org (CC BY 4.0,
"NOAA GEFS from dynamical.org"), stored as Icechunk/Zarr on AWS Open Data. One
00 UTC run per day since 2020-10-01, 31 members, 3-hourly to 10 days, then
6-hourly. The store is read point by point, so only the chunks around the
configured locations are downloaded (roughly 40 MB per run).

Three steps, each cached under data/<market>/gefs/:

1. raw/YYYY/YYYYMMDD.npz - the five raw variables for every member and lead up
   to 16 days at each location, exactly as served (one file per run).
2. features.parquet      - each run turned into hourly national drivers (the
   same four as the observed weather), ensemble mean and spread per lead hour.
3. lead_frames()         - for every lead day N, the value each hour had in the
   run issued N days before it. This is what the models see, so a forecast made
   on day D only ever uses runs that existed on day D.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .features import WX, national_drivers

RAW_VARS = [
    "wind_u_100m",
    "wind_v_100m",
    "temperature_2m",  # deg C, instantaneous
    "precipitation_surface",  # mm/s, mean since the previous step
    "downward_short_wave_radiation_flux_surface",  # W/m2, mean over the last 3 or 6 h (see _deaccumulate_sw)
]
MAX_LEAD_H = 384  # 16 days, the part of the 35-day run that is kept
HOURS = np.arange(MAX_LEAD_H + 1)  # hourly lead grid of processed runs
SPREAD = [f"{name}_sd" for name in WX]


# ---------------------------------------------------------------- download


def _dir(cfg: Config) -> Path:
    return cfg.data_dir / "gefs"


def raw_path(cfg: Config, init: pd.Timestamp) -> Path:
    return _dir(cfg) / "raw" / f"{init:%Y}" / f"{init:%Y%m%d}.npz"


def open_store(cfg: Config):
    """Lazy xarray view of the whole GEFS store (metadata only, nothing downloaded)."""
    import icechunk
    import pystac
    import xarray as xr

    g = cfg.raw["weather"]["gefs"]
    collection = pystac.Catalog.from_file(g["catalog"]).get_child(g["dataset"])
    if collection is None:
        raise SystemExit(f"dataset {g['dataset']!r} not found in {g['catalog']}")
    repo = icechunk.Repository.open(icechunk.http_storage(collection.assets["icechunk-https"].href))
    return xr.open_zarr(repo.readonly_session("main").store, chunks=None)


def _location_meta(cfg: Config) -> dict:
    return {loc.name: [loc.lat, loc.lon] for loc in cfg.locations}


def _check_locations(cfg: Config) -> None:
    """Raw files hold one column per configured location; refuse to mix layouts."""
    meta_path = _dir(cfg) / "raw" / "locations.json"
    want = _location_meta(cfg)
    if meta_path.exists():
        have = json.loads(meta_path.read_text())
        if have != want:
            raise SystemExit(
                f"The locations in the config differ from those of the stored GEFS runs ({meta_path}).\n"
                f"Move or delete {meta_path.parent} to download them again for the new locations."
            )
    else:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(want, indent=1))


def _complete_inits(cfg: Config, ds, start: pd.Timestamp) -> pd.DatetimeIndex:
    """00 UTC runs from `start` whose members all reach 16 days (the newest may still be arriving).

    The store's `ingested_forecast_length` is not filled in, so the newest runs
    are checked directly: every member must have a value at 16 days.
    """
    inits = pd.DatetimeIndex(ds.init_time.values).tz_localize("UTC")
    keep = (inits >= start) & (inits.hour == 0)
    loc = cfg.locations[0]
    for i in np.flatnonzero(keep)[-3:]:
        last = (
            ds["temperature_2m"]
            .isel(init_time=i)
            .sel(lead_time=np.timedelta64(MAX_LEAD_H, "h"), latitude=loc.lat, longitude=loc.lon, method="nearest")
        )
        if np.isnan(last.values).any():
            keep[i] = False
    return inits[keep]


def _fetch_one(cfg: Config, ds, init: pd.Timestamp, n_leads: int) -> None:
    import xarray as xr

    lat = xr.DataArray([loc.lat for loc in cfg.locations], dims="loc")
    lon = xr.DataArray([loc.lon for loc in cfg.locations], dims="loc")
    sub = (
        ds[RAW_VARS]
        .sel(init_time=init.tz_localize(None))
        .isel(lead_time=slice(0, n_leads))
        .sel(latitude=lat, longitude=lon, method="nearest")
        .load()
    )
    data = np.stack([sub[v].transpose("ensemble_member", "lead_time", "loc").values for v in RAW_VARS])
    lead_h = (sub.lead_time.values / np.timedelta64(1, "h")).astype("int16")
    path = raw_path(cfg, init)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        data=data.astype("float32"),  # [variable, member, lead, location]
        lead_hours=lead_h,
        variables=np.array(RAW_VARS),
        locations=np.array([loc.name for loc in cfg.locations]),
        grid_lat=sub.latitude.values,
        grid_lon=sub.longitude.values,
    )
    os.replace(tmp, path)  # never leave a half-written run behind


def update_archive(cfg: Config, tries: int = 4) -> pd.DatetimeIndex:
    """Download every run not stored yet. Safe to interrupt and re-run."""
    _check_locations(cfg)
    ds = open_store(cfg)
    inits = _complete_inits(cfg, ds, cfg.archive_start)
    missing = [t for t in inits if not raw_path(cfg, t).exists()]
    n_leads = int((ds.lead_time.values <= np.timedelta64(MAX_LEAD_H, "h")).sum())
    have = len(inits) - len(missing)
    print(
        f"gefs: {len(inits)} runs {inits[0].date()} -> {inits[-1].date()}, {have} stored, {len(missing)} to fetch",
        flush=True,
    )
    if not missing:
        return inits

    def job(init):
        for attempt in range(tries):
            try:
                _fetch_one(cfg, ds, init, n_leads)
                return init, None
            except Exception as e:  # noqa: BLE001 - network errors come in many types; back off and retry
                err = e
                time.sleep(10 * (attempt + 1))
        return init, err

    t0, done, failed = time.time(), 0, []
    workers = int(cfg.raw["weather"]["gefs"].get("workers", 4))
    with ThreadPoolExecutor(workers) as pool:
        for fut in as_completed([pool.submit(job, t) for t in missing]):
            init, err = fut.result()
            done += 1
            if err is not None:
                failed.append(init)
                print(f"  gefs {init.date()} failed: {type(err).__name__}: {err}", flush=True)
            if done % 25 == 0 or done == len(missing):
                rate = (time.time() - t0) / done
                left = (len(missing) - done) * rate / 3600
                print(f"  gefs {done}/{len(missing)} runs, {rate:.1f} s/run, about {left:.1f} h left", flush=True)
    if failed:
        print(
            f"gefs: {len(failed)} runs failed and will be retried on the next run: "
            f"{', '.join(str(t.date()) for t in sorted(failed)[:10])}{' ...' if len(failed) > 10 else ''}",
            flush=True,
        )
    return inits


# ---------------------------------------------------------------- processing


def _deaccumulate_sw(sw: np.ndarray, lead_h: np.ndarray) -> np.ndarray:
    """Turn GFS-style radiation averages into plain per-step averages.

    Up to 240 h, values at 6-hourly boundaries (00/06/12/18 UTC for a 00 UTC run)
    are means over the last 6 h, those in between over the last 3 h. The mean over
    the second half of a 6 h window is therefore 2 * (6 h mean) - (first 3 h mean).
    Past 240 h the step is 6 h and the values are already per-step.
    """
    out = sw.copy()
    for i in range(1, len(lead_h)):
        if lead_h[i] % 6 == 0 and lead_h[i] - lead_h[i - 1] == 3:
            out[..., i] = np.maximum(2.0 * sw[..., i] - sw[..., i - 1], 0.0)
    return out


def _to_hourly(values: np.ndarray, lead_h: np.ndarray, kind: str) -> np.ndarray:
    """Values on the native lead axis (last) to the hourly grid HOURS.

    instant: linear interpolation.
    mean:    step averages; the hour ending at h takes the mean of the step that
             contains it (Open-Meteo's "preceding hour" convention).
    """
    beyond = HOURS > lead_h[-1]  # past the end of the run: unknown
    if kind == "mean":
        out = values[..., np.minimum(np.searchsorted(lead_h, HOURS, side="left"), len(lead_h) - 1)]
    else:
        j = np.clip(np.searchsorted(lead_h, HOURS, side="right"), 1, len(lead_h) - 1)
        lo, hi = lead_h[j - 1], lead_h[j]
        w = np.clip((HOURS - lo) / (hi - lo), 0.0, 1.0)
        out = values[..., j - 1] * (1 - w) + values[..., j] * w
    return np.where(beyond, np.nan, out)


def process_run(cfg: Config, raw: dict) -> tuple[np.ndarray, np.ndarray]:
    """One raw run -> hourly national drivers: ensemble mean and spread, each [lead hour, len(WX)]."""
    names = list(raw["locations"])
    want = [loc.name for loc in cfg.locations]
    if names != want:
        raise ValueError(f"raw run has locations {names}, config has {want}")
    lead_h = raw["lead_hours"].astype("int64")
    data = raw["data"].astype("float64")  # [variable, member, lead, location]
    v = dict(zip(raw["variables"], data, strict=True))
    to_lead_last = (0, 2, 1)  # [member, lead, loc] -> [member, loc, lead] for interpolation along leads
    ws = np.hypot(v["wind_u_100m"], v["wind_v_100m"]).transpose(to_lead_last)
    temp = v["temperature_2m"].transpose(to_lead_last)
    precip = (v["precipitation_surface"] * 3600.0).transpose(to_lead_last)  # mm/s -> mm/h
    sw = _deaccumulate_sw(v["downward_short_wave_radiation_flux_surface"].transpose(to_lead_last), lead_h)

    def hourly(x, kind):
        return np.moveaxis(_to_hourly(x, lead_h, kind), -1, 1)  # [member, hour, loc]

    drivers = national_drivers(
        cfg, hourly(ws, "instant"), hourly(sw, "mean"), hourly(temp, "instant"), hourly(precip, "mean")
    )  # [member, hour, len(WX)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN hours (lead 0 of mean variables)
        return np.nanmean(drivers, axis=0), np.nanstd(drivers, axis=0)


def features_path(cfg: Config) -> Path:
    return _dir(cfg) / "features.parquet"


def update_features(cfg: Config) -> pd.DataFrame:
    """Process every stored raw run that is new or changed since the last call."""
    path = features_path(cfg)
    old = pd.read_parquet(path) if path.exists() else None
    built = path.stat().st_mtime if path.exists() else 0.0
    raw_files = sorted((_dir(cfg) / "raw").glob("*/*.npz"))
    raw_files = [p for p in raw_files if not p.name.endswith(".tmp.npz")]
    # New runs, plus any rewritten since the last build. Checking membership as well
    # as mtime catches runs that landed while a previous build was in progress.
    have = set() if old is None else {f"{t:%Y%m%d}" for t in old["init_time"].unique()}
    todo = [p for p in raw_files if p.stem not in have or p.stat().st_mtime > built]
    if old is not None and not todo:
        return old

    rows = []
    for i, p in enumerate(todo):
        init = pd.Timestamp(p.stem, tz="UTC")
        with np.load(p) as raw:
            mean, sd = process_run(cfg, dict(raw))
        df = pd.DataFrame(np.concatenate([mean, sd], axis=1).astype("float32"), columns=WX + SPREAD)
        df.insert(0, "lead_hour", HOURS.astype("int16"))
        df.insert(0, "init_time", init)
        rows.append(df)
        if (i + 1) % 250 == 0:
            print(f"  gefs features {i + 1}/{len(todo)}", flush=True)
    new = pd.concat(rows, ignore_index=True)
    if old is not None:
        old = old[~old["init_time"].isin(new["init_time"].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["init_time", "lead_hour"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path)
    print(f"gefs features: {new['init_time'].nunique()} runs, {len(todo)} newly processed", flush=True)
    return new


def load_features(cfg: Config) -> pd.DataFrame:
    path = features_path(cfg)
    if not path.exists():
        raise SystemExit("No GEFS forecasts found. Run `climate-power fetch` first.")
    return pd.read_parquet(path)


def _as_cube(feats: pd.DataFrame, columns: list[str]) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Long table -> (run times, array [run, lead hour, column])."""
    inits = pd.DatetimeIndex(feats["init_time"].unique())
    n_lead = len(HOURS)
    if len(feats) != len(inits) * n_lead:
        raise ValueError("GEFS feature table is not complete for every run; delete features.parquet and re-run")
    return inits, feats[columns].to_numpy("float32").reshape(len(inits), n_lead, len(columns))


def lead_frames(feats: pd.DataFrame, max_lead_day: int, columns: list[str] = WX) -> dict[int, pd.DataFrame]:
    """For N = 1..max_lead_day: the value at hour t in the run issued at 00 UTC, N days before t's UTC day.

    That run's lead for hour t is 24N to 24N+23 hours, the same meaning as
    Open-Meteo's `previous_dayN`. Hours whose run is missing are NaN.
    """
    if (max_lead_day + 1) * 24 - 1 > MAX_LEAD_H:
        raise ValueError(f"lead day {max_lead_day} needs runs longer than {MAX_LEAD_H} h")
    inits, cube = _as_cube(feats, columns)
    out = {}
    for n in range(1, max_lead_day + 1):
        hours = np.arange(24 * n, 24 * n + 24)
        times = (inits.values[:, None] + hours[None] * np.timedelta64(1, "h")).reshape(-1)
        values = cube[:, hours].reshape(-1, len(columns))
        index = pd.DatetimeIndex(times, name="time").tz_localize("UTC")
        out[n] = pd.DataFrame(values, index=index, columns=columns)
    return out


def latest_frame(feats: pd.DataFrame, columns: list[str] = WX, runs: int = 20) -> pd.DataFrame:
    """Each hour from the newest of the last `runs` runs that covers it: the best forecast known now."""
    inits, cube = _as_cube(feats, columns)
    first = max(0, len(inits) - runs)
    index = pd.date_range(inits[first], inits[-1] + pd.Timedelta(hours=MAX_LEAD_H), freq="h", name="time")
    out = pd.DataFrame(np.nan, index=index, columns=columns, dtype="float32")
    for i in range(first, len(inits)):  # oldest to newest, newer overwrites
        times = pd.DatetimeIndex(inits[i] + pd.to_timedelta(HOURS, unit="h"))
        frame = pd.DataFrame(cube[i], index=times, columns=columns).reindex(index)
        out = frame.combine_first(out)
    return out
