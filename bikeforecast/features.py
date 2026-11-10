"""Turn raw inputs (timestamp + weather) into the model's feature row(s)."""
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd

FEATURES = [
    "hr", "weekday", "season", "mnth", "yr", "holiday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
]
TARGET = "cnt"


def season_of(ts: datetime) -> int:
    """Dataset convention: 1=winter, 2=spring, 3=summer, 4=fall."""
    md = (ts.month, ts.day)
    if md < (3, 21) or md >= (12, 21):
        return 1
    if md < (6, 21):
        return 2
    if md < (9, 23):
        return 3
    return 4


def build_frame(
    timestamps: Iterable[datetime],
    *,
    temp_c: float,
    humidity_pct: float,
    windspeed_kmh: float,
    weather: int,
    feels_like_c: Optional[float] = None,
    holiday: bool = False,
) -> pd.DataFrame:
    """Convert human units to the dataset's normalised units (see UCI readme)."""
    if feels_like_c is None:
        feels_like_c = temp_c
    rows = []
    for ts in timestamps:
        weekday = (ts.weekday() + 1) % 7          # dataset: 0 = Sunday
        rows.append({
            "hr": ts.hour,
            "weekday": weekday,
            "season": season_of(ts),
            "mnth": ts.month,
            "yr": 0 if ts.year <= 2011 else 1,
            "holiday": int(holiday),
            "workingday": int(weekday not in (0, 6) and not holiday),
            "weathersit": weather,
            "temp": (temp_c + 8) / 47,            # min -8C, max 39C
            "atemp": (feels_like_c + 16) / 66,    # min -16C, max 50C
            "hum": humidity_pct / 100,
            "windspeed": windspeed_kmh / 67,
        })
    return pd.DataFrame(rows, columns=FEATURES)
