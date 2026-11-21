import numpy as np
import pandas as pd
import pytest

from climate_power.config import load_config
from climate_power.features import (
    SPREAD,
    WX,
    Master,
    build_master,
    build_samples,
    origin_positions,
    power_curve,
)


def test_power_curve():
    cf = power_curve([0, 3, 12, 20, 25, 30])
    assert cf.tolist() == [0, 0, 1, 1, 0, 0]
    assert 0 < power_curve([8])[0] < 1


@pytest.fixture(params=["gefs", "open_meteo"])
def setup(request):
    cfg = load_config().with_overrides(source=request.param)
    rng = np.random.default_rng(0)
    index = pd.date_range("2025-01-01", periods=24 * 80, freq="h", tz="UTC")
    T, F = len(index), len(WX)
    local = index.tz_convert(cfg.timezone)
    # forecast[n] is filled with the constant n so its use is traceable
    forecast = np.stack([np.full((T, F), n, dtype="float32") for n in range(cfg.max_lead + 2)])
    m = Master(
        index=index,
        price=rng.normal(80, 30, T),
        actual=rng.normal(size=(T, F)).astype("float32"),
        forecast=forecast,
        clim=np.full((T, F), -1, dtype="float32"),
        hour=local.hour.to_numpy(),
        dow=local.day_of_week.to_numpy(),
        doy=local.day_of_year.to_numpy(),
        dayoff=(local.day_of_week >= 5).astype("float32"),
        wx_mean=np.zeros(F),
        wx_std=np.ones(F),
    )
    return cfg, m, origin_positions(m, cfg)


def with_spread(m: Master) -> Master:
    """spread[n] is n + 0.5 and the climatological spread 9, so their use is traceable too."""
    m.spread = m.forecast + 0.5
    m.spread[0] = np.nan
    m.spread_clim = np.full_like(m.clim, 9.0)
    return m


def test_target_day_k_uses_forecast_issued_k_plus_one_days_before(setup):
    cfg, m, pos = setup
    s = build_samples(m, cfg, pos[:3])
    wind = s.tab["wx_wind_cf"].to_numpy().reshape(3, cfg.horizon)
    for day in range(1, 15):
        expected = day + 1 if day <= cfg.max_lead else -1  # -1 marks climatology
        assert (wind[:, (day - 1) * 24 : day * 24] == expected).all()


def test_inputs_do_not_depend_on_the_future(setup):
    cfg, m, pos = setup
    p = pos[5:6]
    before = build_samples(m, cfg, p)
    m.price[p[0] :] += 1000  # prices from the first target hour on
    m.actual[p[0] - 24 :] += 1000  # observations from the forecast day on
    after = build_samples(m, cfg, p)
    assert np.array_equal(before.enc, after.enc)
    assert np.array_equal(before.dec, after.dec)
    assert before.tab.equals(after.tab)
    assert not np.array_equal(before.y, after.y)


def test_origins_are_local_midnight(setup):
    cfg, m, pos = setup
    assert (m.index[pos].tz_convert(cfg.timezone).hour == 0).all()


def test_max_lead_follows_source():
    cfg = load_config()
    assert cfg.with_overrides(source="gefs").max_lead == 14
    assert cfg.with_overrides(source="open_meteo").max_lead == 6
    gefs = cfg.with_overrides(source="gefs")
    assert gefs.train_start == pd.Timestamp("2020-10-04", tz="UTC")
    assert gefs.with_overrides(train_start="2024-04-03").train_start == pd.Timestamp("2024-04-03", tz="UTC")


def test_spread_comes_from_the_same_forecast_as_the_mean(setup):
    cfg, m, pos = setup
    s = build_samples(with_spread(m), cfg, pos[:3])
    wind, sd = (s.tab[c].to_numpy().reshape(3, cfg.horizon) for c in ("wx_wind_cf", "sd_wind_cf"))
    for day in range(1, 15):
        sl = slice((day - 1) * 24, day * 24)
        if day <= cfg.max_lead:
            assert (sd[:, sl] == wind[:, sl] + 0.5).all()
        else:
            assert (sd[:, sl] == 9.0).all()  # climatology carries the climatological spread
    plain = build_samples(Master(**{**m.__dict__, "spread": None, "spread_clim": None}), cfg, pos[:3])
    assert s.dec.shape[-1] == plain.dec.shape[-1] + len(WX)
    assert not any(c.startswith("sd_") for c in plain.tab.columns)


def test_oracle_has_zero_spread(setup):
    cfg, m, pos = setup
    s = build_samples(with_spread(m), cfg, pos[:3], "oracle")
    assert (s.tab[[f"sd_{name}" for name in WX]].to_numpy() == 0).all()


def test_spread_does_not_depend_on_the_future(setup):
    cfg, m, pos = setup
    m = with_spread(m)
    p = pos[5:6]
    before = build_samples(m, cfg, p)
    m.price[p[0] :] += 1000
    m.actual[p[0] - 24 :] += 1000
    after = build_samples(m, cfg, p)
    assert np.array_equal(before.dec, after.dec)
    assert before.tab.equals(after.tab)


def test_build_master_patches_spread_with_its_mean():
    """A missing run is filled from the next-older forecast, mean and spread together."""
    cfg = load_config().with_overrides(source="gefs")
    index = pd.date_range(
        cfg.archive_start - pd.Timedelta(days=60), pd.Timestamp("2021-01-31 23:00", tz="UTC"), freq="h"
    )
    rng = np.random.default_rng(1)
    actual_wide = pd.DataFrame(
        {
            f"{loc.name}|{var}": rng.uniform(1, 10, len(index))
            for loc in cfg.locations
            for var in ("wind_speed_100m", "shortwave_radiation", "temperature_2m", "precipitation")
        },
        index=index,
    )
    price = pd.Series(rng.normal(80, 20, len(index)), index=index)
    frames = {n: pd.DataFrame({c: float(n) for c in WX + SPREAD}, index=index) for n in range(1, cfg.max_lead + 2)}
    gap = pd.Timestamp("2021-01-10 12:00", tz="UTC")
    frames[3].loc[gap] = np.nan  # run behind lead 3 missing for this hour
    m = build_master(cfg, price, actual_wide, frames)
    t = m.index.get_loc(gap)
    std = m.wx_std.astype("float32")
    np.testing.assert_allclose(m.spread[3, t], 4.0 / std, rtol=1e-6)  # taken from lead 4, like the mean
    np.testing.assert_allclose(m.spread[2, t], 2.0 / std, rtol=1e-6)  # neighbours untouched
    assert m.spread_clim.shape == m.clim.shape and (m.spread_clim > 0).all()
    assert not np.isnan(m.spread[1:]).any()

    # Live mode builds a short index far from the climatology period; the spread must still be defined.
    import copy
    from dataclasses import replace

    raw = copy.deepcopy(cfg.raw)
    raw["model"]["climatology_end"] = "2020-11-15"  # before the live window of the last 60 days
    live = pd.DataFrame(0.5, index=index, columns=WX)
    m_live = build_master(replace(cfg, raw=raw), price, actual_wide, frames, live=live, end=index[-1])
    assert np.isfinite(m_live.spread_clim).all() and (m_live.spread_clim > 0).all()


def test_spread_setting_by_source():
    cfg = load_config()
    assert cfg.with_overrides(source="gefs").use_spread
    assert not cfg.with_overrides(source="gefs", spread=False).use_spread
    assert not cfg.with_overrides(source="open_meteo").use_spread


def test_select_weights_prefers_members_with_independent_errors():
    from climate_power.combine import select_weights

    rng = np.random.default_rng(0)
    truth = np.zeros((200, 24))
    a = truth + rng.normal(0, 1, truth.shape)  # two good members, independent errors
    b = truth + rng.normal(0, 1, truth.shape)
    bad = truth + rng.normal(0, 3, truth.shape)
    w = select_weights({"a": a, "b": b, "bad": bad}, lambda p: np.abs(p - truth).mean())
    # Optimal weights for independent errors are 9:9:1 here, so "bad" earns only a little.
    assert w.get("bad", 0) < 0.1
    assert abs(w["a"] - w["b"]) < 0.15
    assert abs(sum(w.values()) - 1) < 1e-9
