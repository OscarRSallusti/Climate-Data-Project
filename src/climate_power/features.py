"""Turn raw prices and weather into model-ready samples without leaking the future.

Forecast setup
--------------
A forecast is made on the morning of local day D, before the noon day-ahead
auction. Known at that point:

* prices through the end of day D (cleared the day before),
* observed weather through day D-1,
* weather forecasts issued up to the end of day D-1.

The targets are the 336 hourly prices of days D+1 .. D+14. "Origin" is the first
target hour (local midnight starting D+1).

Weather the model may see for target day k (1..14) is the forecast issued k+1
days before the hour, which Open-Meteo archives for lead <= 7 days. So days
1..6 use real archived forecasts and days 7..14 fall back to climatology. The
"oracle" mode swaps in observed weather everywhere, as an upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import holidays
import numpy as np
import pandas as pd

from .config import Config

WX = ["wind_cf", "solar", "temp", "precip"]
NORM_DAYS = 28


def power_curve(ws):
    """Generic turbine: cut-in 3 m/s, cubic ramp to rated at 12 m/s, cut-out 25 m/s."""
    ws = np.asarray(ws, dtype="float64")
    cf = np.clip((ws**3 - 3.0**3) / (12.0**3 - 3.0**3), 0.0, 1.0)
    return np.where(ws >= 25.0, 0.0, cf)


def weather_features(cfg: Config, wide: pd.DataFrame, lead: int | None = None) -> pd.DataFrame:
    """Collapse per-location variables into four national drivers."""
    suffix = "" if lead is None else f"|{lead}"

    def cols(group: str | None, var: str) -> pd.DataFrame:
        names = cfg.group(group) if group else [loc.name for loc in cfg.locations]
        return wide[[f"{n}|{var}{suffix}" for n in names]]

    wind = cols("wind", "wind_speed_100m")
    out = pd.DataFrame(index=wide.index)
    out["wind_cf"] = pd.DataFrame(power_curve(wind), index=wide.index).mean(axis=1, skipna=False)
    out["solar"] = cols("solar", "shortwave_radiation").mean(axis=1, skipna=False)
    out["temp"] = cols("load", "temperature_2m").mean(axis=1, skipna=False)
    out["precip"] = cols(None, "precipitation").mean(axis=1, skipna=False)
    return out[WX]


def climatology(feats: pd.DataFrame, end: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Smooth normal for each (day of year, UTC hour), from data up to `end` only."""
    hist = feats.loc[: pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)].dropna()
    doy = np.minimum(hist.index.day_of_year, 365)
    table = hist.groupby([doy, hist.index.hour]).mean()
    table = table.reindex(pd.MultiIndex.from_product([range(1, 366), range(24)]))
    out = {}
    for col in feats.columns:
        grid = table[col].unstack()  # 365 x 24
        padded = pd.concat([grid.iloc[-15:], grid, grid.iloc[:15]])  # wrap the year
        smooth = padded.rolling(31, center=True, min_periods=1).mean().iloc[15:-15]
        out[col] = smooth.to_numpy()[np.minimum(index.day_of_year, 365) - 1, index.hour]
    return pd.DataFrame(out, index=index)


@dataclass
class Master:
    """Aligned hourly arrays on one contiguous UTC index."""

    index: pd.DatetimeIndex
    price: np.ndarray  # [T]
    actual: np.ndarray  # [T, F] observed weather features, standardised
    forecast: np.ndarray  # [8, T, F] forecast[N] = issued N days before; index 0 unused
    clim: np.ndarray  # [T, F]
    hour: np.ndarray  # local hour of day
    dow: np.ndarray  # local day of week, Monday = 0
    doy: np.ndarray  # local day of year
    dayoff: np.ndarray  # weekend or public holiday
    wx_mean: np.ndarray
    wx_std: np.ndarray


def build_master(
    cfg: Config,
    price: pd.Series,
    actual_wide: pd.DataFrame,
    forecast_wide: pd.DataFrame | None,
    live_wide: pd.DataFrame | None = None,
    end: pd.Timestamp | None = None,
) -> Master:
    """Assemble the aligned arrays.

    Backtest: pass `forecast_wide` (the previous-runs archive).
    Live: pass `live_wide` (the current forecast); it stands in for every lead and
    fills the hours observations do not cover yet.
    """
    actual = weather_features(cfg, actual_wide)
    start = pd.Timestamp(cfg.raw["weather"]["forecast_archive_start"], tz="UTC") - pd.Timedelta(days=45)
    if live_wide is not None:
        start = price.index[-1] - pd.Timedelta(days=60)
    end = end or min(price.index[-1], actual.index[-1])
    index = pd.date_range(start, end, freq="h")

    clim_end = cfg.raw["model"]["climatology_end"]
    clim = climatology(actual, clim_end, index)
    stats = actual.loc[:clim_end]
    mean, std = stats.mean().to_numpy(), stats.std().to_numpy()

    def z(df: pd.DataFrame) -> np.ndarray:
        return ((df.reindex(index) - mean) / std).to_numpy(dtype="float32")

    clim_z = z(clim)
    fc = np.full((8, len(index), len(WX)), np.nan, dtype="float32")
    if live_wide is not None:
        live = weather_features(cfg, live_wide)
        # Observations are final a couple of days back; after that trust the live run.
        settled = pd.Timestamp.now("UTC").floor("h") - pd.Timedelta(days=2)
        observed = actual.reindex(index)
        observed[index > settled] = np.nan
        combined = observed.combine_first(live.reindex(index))
        act_z = z(combined)
        fc[:] = np.where(np.isnan(act_z), clim_z, act_z)[None]
    else:
        act_z = z(actual)
        for n in range(1, 8):
            fc[n] = z(weather_features(cfg, forecast_wide, lead=n))
        # Patch archive gaps with the next-older forecast, then climatology.
        for n in range(6, 0, -1):
            fc[n] = np.where(np.isnan(fc[n]), fc[n + 1], fc[n])
        fc[1:] = np.where(np.isnan(fc[1:]), clim_z[None], fc[1:])

    local = index.tz_convert(cfg.timezone)
    years = range(local[0].year, local[-1].year + 1)
    hol = holidays.country_holidays(cfg.raw["holidays_country"], years=years)
    is_holiday = np.array([d in hol for d in local.date])
    return Master(
        index=index,
        price=price.reindex(index).to_numpy(dtype="float64"),
        actual=act_z,
        forecast=fc,
        clim=clim_z,
        hour=local.hour.to_numpy(),
        dow=local.day_of_week.to_numpy(),
        doy=local.day_of_year.to_numpy(),
        dayoff=((local.day_of_week >= 5) | is_holiday).astype("float32"),
        wx_mean=mean,
        wx_std=std,
    )


@dataclass
class Samples:
    origins: pd.DatetimeIndex  # first target hour of each sample
    enc: np.ndarray  # [n, L, Fe]  encoder sequence for the LSTM
    dec: np.ndarray  # [n, H, Fd]  decoder sequence for the LSTM
    tab: pd.DataFrame  # n*H rows of tabular features for the tree model
    y: np.ndarray  # [n, H] normalised target (NaN when unknown)
    price: np.ndarray  # [n, H] target in currency units
    center: np.ndarray  # [n]
    scale: np.ndarray  # [n]
    profile: np.ndarray  # [n, H] normalised 7-day hour-of-day mean, a baseline
    naive: np.ndarray  # [n, H] normalised same-hour-last-week, a baseline

    def denorm(self, z: np.ndarray) -> np.ndarray:
        return z * self.scale[:, None] + self.center[:, None]

    def subset(self, mask: np.ndarray) -> Samples:
        h = self.y.shape[1]
        return Samples(
            origins=self.origins[mask],
            enc=self.enc[mask],
            dec=self.dec[mask],
            tab=self.tab[np.repeat(mask, h)].reset_index(drop=True),
            y=self.y[mask],
            price=self.price[mask],
            center=self.center[mask],
            scale=self.scale[mask],
            profile=self.profile[mask],
            naive=self.naive[mask],
        )


DEC_PROFILE = 0  # column of `profile` in the decoder input; the LSTM predicts a residual on it
WEATHER_TAB_PREFIXES = ("wx_", "dwx_", "day_", "is_fc")


def origin_positions(m: Master, cfg: Config) -> np.ndarray:
    pos = np.flatnonzero(m.hour == 0)
    lo = max(NORM_DAYS * 24, cfg.history)
    return pos[(pos >= lo) & (pos + cfg.horizon <= len(m.index))]


def build_samples(m: Master, cfg: Config, pos: np.ndarray, weather: str = "asavail") -> Samples:
    """weather: "asavail" (forecasts then climatology) or "oracle" (observed)."""
    L, H, F = cfg.history, cfg.horizon, len(WX)
    n = len(pos)
    days = H // 24
    enc_idx = pos[:, None] + np.arange(-L, 0)[None]
    dec_idx = pos[:, None] + np.arange(H)[None]

    # Per-sample robust normalisation from the last 28 days: keeps the model
    # usable when the overall price level or volatility drifts.
    hist = m.price[pos[:, None] + np.arange(-NORM_DAYS * 24, 0)[None]]
    center = np.nanmedian(hist, axis=1)
    q75, q25 = np.nanpercentile(hist, [75, 25], axis=1)
    scale = np.maximum((q75 - q25) / 1.349, 5.0)

    p_enc = (m.price[enc_idx] - center[:, None]) / scale[:, None]
    price = m.price[dec_idx]  # NaN where not yet known (live forecasts)
    y = (price - center[:, None]) / scale[:, None]

    # Encoder weather: observations, except the final day which is still a forecast.
    wx_enc = m.actual[enc_idx].copy()
    wx_enc[:, -24:] = m.forecast[1][enc_idx[:, -24:]]

    k = np.arange(H) // 24 + 1  # target day 1..14
    wx_dec = np.empty((n, H, F), dtype="float32")
    for day in range(1, days + 1):
        sl = slice((day - 1) * 24, day * 24)
        if weather == "oracle":
            src = m.actual
        elif day <= cfg.max_lead:
            src = m.forecast[day + 1]
        else:
            src = m.clim
        wx_dec[:, sl] = src[dec_idx[:, sl]]
    is_fc = np.broadcast_to(((k <= cfg.max_lead) | (weather == "oracle")).astype("float32"), (n, H))

    profile = np.tile(p_enc.reshape(n, L // 24, 24).mean(axis=1), (1, days))
    naive = np.tile(p_enc[:, -168:], (1, H // 168))

    hour, dow, doy, dayoff = m.hour[dec_idx], m.dow[dec_idx], m.doy[dec_idx], m.dayoff[dec_idx]

    def cyc(x, period):
        a = 2 * np.pi * x / period
        return np.sin(a), np.cos(a)

    h_sin, h_cos = cyc(hour, 24)
    d_sin, d_cos = cyc(dow, 7)
    y_sin, y_cos = cyc(doy, 365.25)
    eh_sin, eh_cos = cyc(m.hour[enc_idx], 24)

    enc = np.concatenate(
        [p_enc[..., None], wx_enc, eh_sin[..., None], eh_cos[..., None], m.dayoff[enc_idx][..., None]],
        axis=-1,
    ).astype("float32")
    dec_parts = [profile, naive, *np.moveaxis(wx_dec, -1, 0), is_fc, np.broadcast_to(k / days, (n, H))]
    dec_parts += [h_sin, h_cos, d_sin, d_cos, y_sin, y_cos, dayoff]
    dec = np.stack(dec_parts, axis=-1).astype("float32")

    # Tabular view. Anchors are last week's weather at the same hour of day, so
    # the trees can relate a change in weather to a change in price.
    anchor = np.tile(wx_enc.reshape(n, L // 24, 24, F).mean(axis=1), (1, days, 1))
    day_mean = np.repeat(wx_dec.reshape(n, days, 24, F).mean(axis=2), 24, axis=1)
    tab = {
        "k": np.broadcast_to(k, (n, H)),
        "hour": hour,
        "dow": dow,
        "dayoff": dayoff,
        "y_sin": y_sin,
        "y_cos": y_cos,
        "profile": profile,
        "naive": naive,
        "log_scale": np.broadcast_to(np.log(scale)[:, None], (n, H)),
        "level": np.broadcast_to((center / scale)[:, None], (n, H)),
        "is_fc": is_fc,
    }
    for i, name in enumerate(WX):
        tab[f"wx_{name}"] = wx_dec[..., i]
        tab[f"dwx_{name}"] = wx_dec[..., i] - anchor[..., i]
    for name in ("wind_cf", "solar", "temp"):
        tab[f"day_{name}"] = day_mean[..., WX.index(name)]
    tab = pd.DataFrame({key: np.asarray(v, dtype="float32").reshape(-1) for key, v in tab.items()})

    return Samples(
        origins=m.index[pos],
        enc=enc,
        dec=dec,
        tab=tab,
        y=y,
        price=price,
        center=center,
        scale=scale,
        profile=profile,
        naive=naive,
    )


def complete(s: Samples) -> np.ndarray:
    """Samples usable for training: nothing missing in inputs or targets."""
    return usable(s) & ~np.isnan(s.y).any(axis=1)


def usable(s: Samples) -> np.ndarray:
    """Samples with complete inputs; targets may still be unknown."""
    return ~(np.isnan(s.enc).any(axis=(1, 2)) | np.isnan(s.dec).any(axis=(1, 2)))
