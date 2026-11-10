"""Replay the test period one day at a time, like production:
predict the day -> reveal actuals -> store -> update monitoring -> maybe flag."""
import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sqlalchemy.orm import Session

from .baseline import same_hour_baseline
from .config import MODEL_DIR, ROLLING_DAYS, TREND_DAYS
from .data import load, split
from .db import DailyMetric, Prediction, get_engine, init_db
from .features import FEATURES, TARGET


def run():
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    model = joblib.load(MODEL_DIR / "model.joblib")
    limit = metrics["error_limit"]

    df = load()
    df["baseline"] = same_hour_baseline(df)
    _, test = split(df)

    engine = get_engine()
    init_db(engine)
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")

    daily, first_flag = [], None
    with Session(engine) as s:
        for day, g in test.groupby("dteday"):
            # 1) predict using only information available before the day happens
            pred = np.clip(model.predict(g[FEATURES]), 0, None)
            # 2) the day happens: actuals are revealed
            actual = g[TARGET].to_numpy()
            err = np.abs(pred - actual)
            base_err = np.abs(g["baseline"].to_numpy() - actual)

            s.add_all([
                Prediction(run_id=run_id, model_version=metrics["model_version"],
                           ts=t.to_pydatetime(), predicted=float(p), baseline=float(b),
                           actual=float(a), abs_error=float(e))
                for t, p, b, a, e in zip(g["ts"], pred, g["baseline"], actual, err)
            ])

            # 3) monitoring
            daily.append(err.mean())
            rolling = float(np.mean(daily[-ROLLING_DAYS:]))
            recent = daily[-TREND_DAYS:]
            trend = float(np.polyfit(range(len(recent)), recent, 1)[0]) if len(recent) >= TREND_DAYS else 0.0
            flagged = len(daily) >= ROLLING_DAYS and rolling > limit
            if flagged and first_flag is None:
                first_flag = day.date()
                print(f"FLAG: rolling {ROLLING_DAYS}-day MAE {rolling:.1f} > limit {limit:.1f} on {first_flag}")

            s.add(DailyMetric(run_id=run_id, day=day.date(), mae=float(err.mean()),
                              baseline_mae=float(base_err.mean()), rolling_mae=rolling,
                              error_trend=trend, error_limit=limit, flagged=bool(flagged)))
        s.commit()

    print(f"{run_id}: replayed {len(daily)} days | overall MAE {np.mean(daily):.1f} | "
          f"first flag: {first_flag or 'none'}")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
