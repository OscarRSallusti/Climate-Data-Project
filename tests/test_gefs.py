import numpy as np
import pandas as pd
import pytest

from climate_power import gefs
from climate_power.config import load_config
from climate_power.features import WX


def test_deaccumulate_sw():
    # True 3-hourly means a, b, c, d; the archive stores 6 h means at 6-hourly boundaries.
    lead = np.array([0, 3, 6, 9, 12])
    a, b, c, d = 100.0, 300.0, 500.0, 200.0
    stored = np.array([np.nan, a, (a + b) / 2, c, (c + d) / 2])
    out = gefs._deaccumulate_sw(stored, lead)
    np.testing.assert_allclose(out[1:], [a, b, c, d])


def test_deaccumulate_leaves_6_hourly_steps():
    lead = np.array([234, 237, 240, 246, 252])
    stored = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    np.testing.assert_allclose(gefs._deaccumulate_sw(stored, lead), [10.0, 20.0, 40.0, 40.0, 50.0])


def test_to_hourly():
    lead = np.array([0, 3, 6, 12])
    x = np.array([0.0, 3.0, 6.0, 12.0])
    inst = gefs._to_hourly(x, lead, "instant")
    np.testing.assert_allclose(inst[:13], np.arange(13))  # linear in lead
    mean = gefs._to_hourly(np.array([np.nan, 1.0, 2.0, 3.0]), lead, "mean")
    # hour h holds the mean of the step containing (h-1, h]
    assert np.isnan(mean[0])
    np.testing.assert_allclose(mean[1:13], [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3])


def _fake_features(inits: pd.DatetimeIndex) -> pd.DataFrame:
    """Every value encodes its run (days since 2000) and lead hour: run * 1000 + lead."""
    rows = []
    for init in inits:
        code = (init - pd.Timestamp("2000-01-01", tz="UTC")).days * 1000 + gefs.HOURS
        df = pd.DataFrame({c: code.astype("float32") for c in WX + gefs.SPREAD})
        df.insert(0, "lead_hour", gefs.HOURS)
        df.insert(0, "init_time", init)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def test_lead_frames_use_the_run_issued_n_days_before():
    inits = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    frames = gefs.lead_frames(_fake_features(inits), 15)
    t = pd.date_range("2024-01-20", periods=72, freq="h", tz="UTC")
    for n, frame in frames.items():
        values = frame["wind_cf"].reindex(t).to_numpy()
        run = values // 1000
        lead = values % 1000
        expected_run = ((t.normalize() - pd.Timedelta(days=n)) - pd.Timestamp("2000-01-01", tz="UTC")).days
        np.testing.assert_array_equal(run, expected_run)
        assert (lead >= 24 * n).all() and (lead < 24 * (n + 1)).all()
        # never a run issued later than n days before the hour
        issued = pd.Timestamp("2000-01-01", tz="UTC") + pd.to_timedelta(run, unit="D")
        assert ((t - issued) >= pd.Timedelta(days=n)).all()


def test_lead_frames_refuse_leads_past_the_run():
    with pytest.raises(ValueError):
        gefs.lead_frames(_fake_features(pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")), 16)


def test_latest_frame_prefers_the_newest_run():
    inits = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    frame = gefs.latest_frame(_fake_features(inits))
    values = frame["temp"]
    newest = (inits[-1] - pd.Timestamp("2000-01-01", tz="UTC")).days
    assert (values[inits[-1] :] // 1000 == newest).all()
    # an hour before the newest run comes from the run before it
    assert values[inits[-1] - pd.Timedelta(hours=1)] // 1000 == newest - 1


def test_process_run_constant_fields():
    cfg = load_config()
    lead = np.r_[np.arange(0, 241, 3), np.arange(246, 385, 6)].astype("int16")
    n_member, n_loc = 3, len(cfg.locations)
    shape = (n_member, len(lead), n_loc)
    sw = np.full(shape, 400.0)
    sw[:, 0] = np.nan
    data = np.stack(
        [
            np.full(shape, 12.0),  # u: 12 m/s -> rated output
            np.zeros(shape),  # v
            np.full(shape, 15.0),  # temperature
            np.full(shape, 1.0 / 3600),  # 1 mm/h
            sw,
        ]
    )
    raw = {
        "data": data,
        "lead_hours": lead,
        "variables": np.array(gefs.RAW_VARS),
        "locations": np.array([loc.name for loc in cfg.locations]),
    }
    mean, sd = gefs.process_run(cfg, raw)
    assert mean.shape == (len(gefs.HOURS), len(WX))
    np.testing.assert_allclose(mean[1:], np.tile([1.0, 400.0, 15.0, 1.0], (len(gefs.HOURS) - 1, 1)), rtol=1e-6)
    np.testing.assert_allclose(sd[1:], 0.0, atol=1e-6)
    assert np.isnan(mean[0, WX.index("solar")]) and mean[0, WX.index("temp")] == pytest.approx(15.0)
