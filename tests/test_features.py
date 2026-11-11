import numpy as np
import pandas as pd
import pytest

from climate_power.config import load_config
from climate_power.features import WX, Master, build_samples, origin_positions, power_curve


def test_power_curve():
    cf = power_curve([0, 3, 12, 20, 25, 30])
    assert cf.tolist() == [0, 0, 1, 1, 0, 0]
    assert 0 < power_curve([8])[0] < 1


@pytest.fixture
def setup():
    cfg = load_config()
    rng = np.random.default_rng(0)
    index = pd.date_range("2025-01-01", periods=24 * 80, freq="h", tz="UTC")
    T, F = len(index), len(WX)
    local = index.tz_convert(cfg.timezone)
    # forecast[n] is filled with the constant n so its use is traceable
    forecast = np.stack([np.full((T, F), n, dtype="float32") for n in range(8)])
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
