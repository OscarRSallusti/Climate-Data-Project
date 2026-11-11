from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Colour follows the model, wherever it appears.
COLORS = {
    "gbm_weather": "#2a78d6",
    "lstm": "#eb6834",
    "gbm_calendar": "#1baf7a",
    "gbm_oracle": "#eda100",
    "naive_week": "#e87ba4",
    "profile_7d": "#6250d6",
}
LABELS = {
    "gbm_weather": "GBM + weather forecast",
    "lstm": "LSTM + weather forecast",
    "gbm_calendar": "GBM, no weather",
    "gbm_oracle": "GBM + observed weather (upper bound)",
    "naive_week": "Naive: same hour last week",
    "profile_7d": "Baseline: 7-day hourly profile",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "font.size": 10,
        "legend.frameon": False,
    }
)


def mae_by_lead(table: pd.DataFrame, unit: str, path, max_lead: int) -> None:
    """table: rows = target day 1..14, columns = model names."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in table.columns:
        ax.plot(table.index, table[name], color=COLORS[name], lw=2, marker="o", ms=4, label=LABELS[name])
    ax.axvline(max_lead + 0.5, color=MUTED, lw=1, ls=":")
    ax.set_ylim(bottom=ax.get_ylim()[0] - 2.5)
    ax.text(max_lead + 0.6, ax.get_ylim()[0] + 0.3, "climatology replaces weather forecasts", va="bottom", color=MUTED, fontsize=9)
    ax.set_xticks(table.index)
    ax.set_xlabel("Days ahead")
    ax.set_ylabel(f"Mean absolute error ({unit})")
    ax.set_title("Forecast error by lead time, test period")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def revisions(actual: np.ndarray, by_lead: dict[int, np.ndarray], day: pd.Timestamp, unit: str, path) -> None:
    """How the forecast for one target day changed as it came closer."""
    leads = sorted(by_lead, reverse=True)
    ramp = ["#b7d3f6", "#86b6ef", "#5598e7", "#256abf", "#0d366b"]  # light = issued early
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1.5]})
    a.plot(leads, [by_lead[k].mean() for k in leads], color=COLORS["gbm_weather"], lw=2, marker="o", ms=5, label="Forecast")
    a.axhline(actual.mean(), color=INK, lw=1.5, ls="--", label="Outcome")
    a.invert_xaxis()
    a.set_xticks(leads)
    a.set_xlabel("Days before delivery")
    a.set_ylabel(f"Daily mean price ({unit})")
    a.set_title("Daily mean")
    a.legend()
    shown = [k for k in (14, 7, 4, 2, 1) if k in by_lead]
    for color, k in zip(ramp[-len(shown) :], shown):
        b.plot(range(24), by_lead[k], color=color, lw=2, label=f"{k} day{'s' if k > 1 else ''} before")
    b.plot(range(24), actual, color=INK, lw=2, ls="--", label="Outcome")
    b.set_xticks(range(0, 24, 3))
    b.set_xlabel("Hour of day (local)")
    b.set_ylabel(f"Price ({unit})")
    b.set_title("Hourly shape")
    b.legend(ncol=3, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.suptitle(f"GBM + weather: forecast revisions for delivery on {day:%a %d %b %Y}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def forecast_plot(times, series: dict[str, np.ndarray], history: pd.Series, unit: str, title: str, path, split=None) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(history.index, history.to_numpy(), color=INK, lw=1.5, label="Day-ahead price")
    for name, values in series.items():
        ax.plot(times, values, color=COLORS[name], lw=1.8, label=LABELS[name])
    if split is not None:
        ax.axvline(split, color=MUTED, lw=1, ls=":")
        ax.text(split, ax.get_ylim()[0], " climatology from here", va="bottom", color=MUTED, fontsize=9)
    ax.set_ylabel(f"Price ({unit})")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b", tz=times.tz))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
