"""Combine forecasts from several backtest runs (different weather sources, training windows, models).

Models trained on different weather data make partly independent errors, so a weighted
average beats each of them. To keep the test score honest, the weights are chosen on the
validation period only, with greedy forward selection with replacement (Caruana et al.,
2004): start empty, repeatedly add the member that most lowers the validation error of the
running average, and keep the best prefix. Members picked k times get weight k / steps.
The test period is scored once, at the end.

Note that the models used the validation period for early stopping, so validation errors
are slightly optimistic for every member alike.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import plots
from .backtest import UNIT, score
from .config import Config

CANDIDATES = ("lstm", "gbm_weather", "gbm_calendar")
COMBINED_TAG = "combined"


def load_predictions(cfg: Config, tags: list[str]) -> dict[str, dict]:
    """{split: {"origins", "price", "preds": {"tag:model": [n, H]}}} for the runs in `tags`."""
    out: dict[str, dict] = {}
    for tag in tags:
        path = cfg.with_overrides(tag=tag).model_dir / "predictions.parquet"
        if not path.exists():
            raise SystemExit(f"No saved predictions for run {tag!r} ({path}). Re-run its backtest first.")
        df = pd.read_parquet(path)
        for split, part in df.groupby("split"):
            origins = pd.DatetimeIndex(part["origin"].unique())
            n, h = len(origins), int(part["lead_hour"].max()) + 1
            price = part["price"].to_numpy().reshape(n, h)
            if split not in out:
                out[split] = {"origins": origins, "price": price, "preds": {}}
                if "naive_week" in part:
                    out[split]["preds"]["naive_week"] = part["naive_week"].to_numpy().reshape(n, h)
            elif not (
                out[split]["origins"].equals(origins) and np.allclose(out[split]["price"], price, equal_nan=True)
            ):
                raise SystemExit(f"Run {tag!r} has a different {split} period from the others; re-run all backtests.")
            for model in CANDIDATES:
                if model in part:
                    out[split]["preds"][f"{tag}:{model}"] = part[model].to_numpy().reshape(n, h)
    return out


def select_weights(preds: dict[str, np.ndarray], mae, steps: int = 30) -> dict[str, float]:
    """Greedy forward selection with replacement. mae(prediction) -> error on the selection period."""
    names = list(preds)
    total = np.zeros_like(preds[names[0]])
    counts = dict.fromkeys(names, 0)
    best_err, best_counts = np.inf, None
    for step in range(1, steps + 1):
        errs = {name: mae((total + preds[name]) / step) for name in names}
        pick = min(errs, key=errs.get)
        total += preds[pick]
        counts[pick] += 1
        if errs[pick] < best_err - 1e-9:
            best_err, best_counts = errs[pick], dict(counts)
    used = sum(best_counts.values())
    return {name: c / used for name, c in best_counts.items() if c}


def weighted(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(w * preds[name] for name, w in weights.items())


def run_combine(cfg: Config, tags: list[str]) -> pd.DataFrame:
    data = load_predictions(cfg, tags)
    val, test = data["val"], data["test"]
    members = [k for k in val["preds"] if k != "naive_week"]

    def val_mae(p: np.ndarray) -> float:
        return float(score(val["origins"], val["price"], {"x": p})[1].loc["x", "MAE"])

    weights = select_weights({k: val["preds"][k] for k in members}, val_mae)
    best_single = min(members, key=lambda k: val_mae(val["preds"][k]))
    equal = {k: 1 / len(members) for k in members}
    print("weights chosen on the validation period:")
    for name, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"  {w:5.2f}  {name}")
    print(
        f"validation MAE: combined {val_mae(weighted(val['preds'], weights)):.2f}, "
        f"best single ({best_single}) {val_mae(val['preds'][best_single]):.2f}"
    )

    def rows(split: dict) -> dict[str, np.ndarray]:
        p = split["preds"]
        return {"combined": weighted(p, weights), "equal_weights": weighted(p, equal), **p}

    by_lead, overall = score(test["origins"], test["price"], rows(test))
    val_scores = score(val["origins"], val["price"], rows(val))[1]["MAE"]
    overall.insert(0, "val MAE", val_scores.reindex(overall.index))
    overall.insert(1, "weight", [weights.get(k, np.nan) for k in overall.index])

    # Live forecasts rebuild each member from its own run, so record how each run was made.
    import joblib

    runs = {}
    for tag in tags:
        saved = joblib.load(cfg.with_overrides(tag=tag).model_dir / "models.joblib")
        runs[tag] = {k: saved.get(k) for k in ("source", "use_spread", "train_start")}
    combo = cfg.with_overrides(tag=COMBINED_TAG)
    combo.model_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "chosen_on": "validation period",
        "val_period": [str(val["origins"][0].date()), str(val["origins"][-1].date())],
        "members": [
            {"run": name.split(":")[0], "model": name.split(":")[1], "weight": round(w, 6), **runs[name.split(":")[0]]}
            for name, w in sorted(weights.items(), key=lambda kv: -kv[1])
        ],
    }
    (combo.model_dir / "combination.json").write_text(json.dumps(spec, indent=2))

    out = combo.report_dir
    out.mkdir(parents=True, exist_ok=True)
    by_lead.round(2).to_csv(out / "mae_by_lead.csv")
    overall.round(3).to_csv(out / "summary.csv")
    shown = ["combined", "equal_weights", best_single, "naive_week"]
    plots.mae_by_lead(
        by_lead[shown].rename(columns={best_single: f"best single: {best_single}"}),
        UNIT,
        out / "mae_by_lead.png",
        cfg.horizon // 24,
    )

    pd.set_option("display.width", 200)
    print("\nTest period (scored once, after choosing the weights):\n", overall.round(2).to_string())
    return overall
