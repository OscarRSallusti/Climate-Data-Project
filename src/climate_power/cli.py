"""Weather-driven day-ahead electricity price forecasting."""

from __future__ import annotations

import argparse

from .config import SOURCES, load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="climate-power", description=__doc__)
    parser.add_argument("--config", default="config/de_lu.yaml")
    parser.add_argument(
        "--source", choices=SOURCES, help="weather forecast source (default: forecast_source in the config)"
    )
    parser.add_argument("--tag", help="write models and reports to a subfolder of this name, for comparison runs")
    parser.add_argument("--train-start", help="first training forecast origin, YYYY-MM-DD (default: archive start)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download / refresh prices, grid data, weather history and archived forecasts")
    sub.add_parser("snapshot", help="save the current 16-day Open-Meteo weather forecast")
    p = sub.add_parser("backtest", help="train all models and evaluate them on the test period")
    p.add_argument("--skip-lstm", action="store_true")
    sub.add_parser("forecast", help="14-day price forecast from the latest data (needs a backtest run)")
    args = parser.parse_args()
    cfg = load_config(args.config).with_overrides(source=args.source, tag=args.tag, train_start=args.train_start)

    if args.command == "fetch":
        from . import grid, prices, weather

        s = prices.update_prices(cfg)
        print(f"prices: {len(s):,} hours, {s.index[0]} -> {s.index[-1]}")
        g = grid.update_generation(cfg)
        print(f"generation and load: {g.shape}, {g.index[0]} -> {g.index[-1]}")
        c = grid.update_capacity(cfg)
        print(f"installed capacity: {c.shape}, {c.index[0].date()} -> {c.index[-1].date()}")
        a = weather.update_actual(cfg)
        print(f"weather actual: {a.shape}, -> {a.index[-1]}")
        if cfg.source == "gefs":
            from . import gefs

            gefs.update_archive(cfg)
            f = gefs.update_features(cfg)
            print(
                f"gefs forecast archive: {f['init_time'].nunique()} runs, "
                f"{f['init_time'].iloc[0].date()} -> {f['init_time'].iloc[-1].date()}"
            )
        else:
            f = weather.update_forecast_archive(cfg)
            print(f"weather forecast archive: {f.shape}, {f.index[0]} -> {f.index[-1]}")
    elif args.command == "snapshot":
        from . import weather

        df = weather.fetch_live_forecast(cfg)
        print(f"saved forecast snapshot: {df.index[0]} -> {df.index[-1]}")
    elif args.command == "backtest":
        from .backtest import run_backtest

        run_backtest(cfg, with_lstm=not args.skip_lstm)
    elif args.command == "forecast":
        from .live import run_forecast

        run_forecast(cfg)
