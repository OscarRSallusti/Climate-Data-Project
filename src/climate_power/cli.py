from __future__ import annotations

import argparse

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="climate-power", description=__doc__)
    parser.add_argument("--config", default="config/de_lu.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download / refresh prices, weather history and archived forecasts")
    sub.add_parser("snapshot", help="save the current 16-day weather forecast")
    p = sub.add_parser("backtest", help="train all models and evaluate them on the test period")
    p.add_argument("--skip-lstm", action="store_true")
    sub.add_parser("forecast", help="14-day price forecast from the latest data (needs a backtest run)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "fetch":
        from . import prices, weather

        s = prices.update_prices(cfg)
        print(f"prices: {len(s):,} hours, {s.index[0]} -> {s.index[-1]}")
        a = weather.update_actual(cfg)
        print(f"weather actual: {a.shape}, -> {a.index[-1]}")
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
