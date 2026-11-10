"""Train the model, compare against the baseline, save model + metrics."""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from .baseline import same_hour_baseline
from .config import (ERROR_LIMIT_MULTIPLIER, MODEL_DIR, TRAIN_END, VAL_START)
from .data import load, split
from .features import FEATURES, TARGET


def make_model():
    return HistGradientBoostingRegressor(
        loss="poisson",            # counts are non-negative
        learning_rate=0.06,
        max_iter=400,
        max_leaf_nodes=31,
        random_state=0,
    )


def main():
    df = load()
    df["baseline"] = same_hour_baseline(df)
    train, test = split(df)

    # 1) Validation inside the training period -> sets the "normal" error level
    fit_part = train[train["dteday"] < VAL_START]
    val_part = train[train["dteday"] >= VAL_START]
    m = make_model().fit(fit_part[FEATURES], fit_part[TARGET])
    val_mae = mean_absolute_error(val_part[TARGET], np.clip(m.predict(val_part[FEATURES]), 0, None))

    # 2) Final model on all training data, evaluated on the future
    model = make_model().fit(train[FEATURES], train[TARGET])
    pred = np.clip(model.predict(test[FEATURES]), 0, None)
    metrics = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "train_end": TRAIN_END,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "val_mae": round(float(val_mae), 2),
        "test_mae_model": round(float(mean_absolute_error(test[TARGET], pred)), 2),
        "test_mae_baseline": round(float(mean_absolute_error(test[TARGET], test["baseline"])), 2),
        "error_limit": round(float(val_mae * ERROR_LIMIT_MULTIPLIER), 2),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
