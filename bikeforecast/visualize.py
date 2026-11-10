"""Create report charts from the latest replay run stored in the database."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import text

from .config import REPORT_DIR
from .db import get_engine


def main():
    engine = get_engine()
    REPORT_DIR.mkdir(exist_ok=True)
    run_id = pd.read_sql(text("SELECT run_id FROM daily_metrics ORDER BY id DESC LIMIT 1"), engine).iloc[0, 0]
    p = pd.read_sql(text("SELECT * FROM predictions WHERE run_id=:r ORDER BY ts"), engine,
                    params={"r": run_id}, parse_dates=["ts"])
    d = pd.read_sql(text("SELECT * FROM daily_metrics WHERE run_id=:r ORDER BY day"), engine,
                    params={"r": run_id}, parse_dates=["day"])

    # 1. predicted vs actual, last 14 days
    last = p[p["ts"] >= p["ts"].max() - pd.Timedelta(days=14)]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(last["ts"], last["actual"], label="Actual", color="#222", lw=1.4)
    ax.plot(last["ts"], last["predicted"], label="Model", color="#d1495b", lw=1.2)
    ax.plot(last["ts"], last["baseline"], label="Baseline (7-day same-hour avg)",
            color="#8da0cb", lw=0.9, alpha=0.8)
    ax.set(title="Hourly rentals: actual vs predicted (last 14 days)", ylabel="Rentals / hour")
    ax.legend(); fig.tight_layout(); fig.savefig(REPORT_DIR / "1_actual_vs_predicted.png", dpi=150)

    # 2. peak hours
    p["hr"] = p["ts"].dt.hour
    p["type"] = p["ts"].dt.dayofweek.map(lambda x: "Weekend" if x >= 5 else "Weekday")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (name, g) in zip(axes, p.groupby("type")):
        m = g.groupby("hr")[["actual", "predicted"]].mean()
        ax.plot(m.index, m["actual"], label="Actual", color="#222", marker="o", ms=3)
        ax.plot(m.index, m["predicted"], label="Model", color="#d1495b", marker="o", ms=3)
        ax.set(title=f"Average demand by hour: {name}", xlabel="Hour of day")
        ax.set_xticks(range(0, 24, 3))
    axes[0].set_ylabel("Avg rentals / hour"); axes[0].legend()
    fig.tight_layout(); fig.savefig(REPORT_DIR / "2_peak_hours.png", dpi=150)

    # 3. drift monitoring
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(d["day"], d["mae"], color="#bbb", lw=1, label="Daily MAE")
    ax.plot(d["day"], d["rolling_mae"], color="#d1495b", lw=2, label="Rolling 7-day MAE")
    ax.axhline(d["error_limit"].iloc[0], color="k", ls="--", label="Error limit")
    fl = d[d["flagged"].astype(bool)]
    ax.scatter(fl["day"], fl["rolling_mae"], color="k", zorder=3, s=14, label="Flagged")
    ax.set(title="Model accuracy over the replay", ylabel="MAE (rentals / hour)")
    ax.legend(); fig.tight_layout(); fig.savefig(REPORT_DIR / "3_error_monitoring.png", dpi=150)
    print(f"Saved charts for {run_id} to {REPORT_DIR}")


if __name__ == "__main__":
    main()
