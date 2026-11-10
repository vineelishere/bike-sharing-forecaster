"""Writes a fake data/hour.csv with the UCI schema, for offline testing/CI only.
Use --drift to make demand shift late in the test period (to see the flag fire)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
ts = pd.date_range("2011-01-01", "2012-12-31 23:00", freq="h")
df = pd.DataFrame({"dteday": ts.normalize(), "hr": ts.hour})
df["weekday"] = (ts.dayofweek + 1) % 7
df["mnth"] = ts.month
df["yr"] = (ts.year - 2011)
df["season"] = ((ts.month % 12) // 3 + 1)
df["holiday"] = 0
df["workingday"] = (~df["weekday"].isin([0, 6])).astype(int)
doy = ts.dayofyear.to_numpy()
t = 15 + 12 * np.sin((doy - 110) / 365 * 2 * np.pi) + rng.normal(0, 3, len(ts))
df["temp"] = ((t + 8) / 47).clip(0, 1); df["atemp"] = ((t + 16) / 66).clip(0, 1)
df["hum"] = rng.uniform(0.3, 0.95, len(ts)); df["windspeed"] = rng.uniform(0, 0.5, len(ts))
df["weathersit"] = rng.choice([1, 2, 3], len(ts), p=[.65, .27, .08])
h = df["hr"].to_numpy()
commute = np.exp(-((h - 8) ** 2) / 4) + np.exp(-((h - 17.5) ** 2) / 5)
leisure = np.exp(-((h - 14) ** 2) / 20)
shape = np.where(df["workingday"] == 1, 250 * commute + 40 * leisure, 200 * leisure)
lam = shape * (0.4 + 0.9 * df["temp"]) * (1 + 0.5 * df["yr"]) * (1 - 0.18 * (df["weathersit"] - 1))
lam = lam + 5
if "--drift" in sys.argv:
    lam = np.where(ts >= "2012-11-10", lam * 0.55, lam)
df["cnt"] = rng.poisson(lam)
df.insert(0, "instant", np.arange(1, len(df) + 1))
Path("data").mkdir(exist_ok=True)
df.to_csv("data/hour.csv", index=False)
print("wrote data/hour.csv", len(df), "rows")
