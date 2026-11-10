import io
import urllib.request
import zipfile

import pandas as pd

from .config import DATA_DIR, TRAIN_END

URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


def download() -> "Path":
    path = DATA_DIR / "hour.csv"
    if path.exists():
        return path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {URL} ...")
    with urllib.request.urlopen(URL) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extract("hour.csv", DATA_DIR)
    return path


def load() -> pd.DataFrame:
    df = pd.read_csv(download(), parse_dates=["dteday"])
    df["ts"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    return df.sort_values("ts").reset_index(drop=True)


def split(df: pd.DataFrame):
    """Chronological split. Never shuffle time series."""
    train = df[df["dteday"] <= TRAIN_END].copy()
    test = df[df["dteday"] > TRAIN_END].copy()
    return train, test
