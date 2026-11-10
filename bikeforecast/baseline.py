"""Baseline: average rentals at the same hour over the previous N days."""
import pandas as pd

from .config import BASELINE_WINDOW_DAYS


def same_hour_baseline(df: pd.DataFrame, window: int = BASELINE_WINDOW_DAYS) -> pd.Series:
    """Returns a Series aligned with df.index.

    For each (day, hour) the prediction uses only *earlier* days, so it is a
    legitimate day-ahead forecast even inside the test period.
    """
    pivot = df.pivot_table(index="dteday", columns="hr", values="cnt")
    pivot = pivot.reindex(pd.date_range(pivot.index.min(), pivot.index.max()))
    prev_mean = pivot.shift(1).rolling(window, min_periods=1).mean()
    stacked = prev_mean.stack()
    idx = pd.MultiIndex.from_arrays([df["dteday"], df["hr"]])
    out = pd.Series(stacked.reindex(idx).to_numpy(), index=df.index)
    # first days of the dataset have no history: fall back to overall hour mean
    hour_mean = df.groupby("hr")["cnt"].transform("mean")
    return out.fillna(hour_mean)
