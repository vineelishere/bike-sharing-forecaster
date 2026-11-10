import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg2://bike:bike@localhost:5432/bike"
)

# Chronological split: everything up to TRAIN_END is training data,
# everything after is the "future" we replay.
TRAIN_END = "2012-08-31"
VAL_START = "2012-07-01"      # last 2 months of training used to set the error limit

ERROR_LIMIT_MULTIPLIER = 1.5  # flag if rolling MAE > 1.5 x validation MAE
ROLLING_DAYS = 7
TREND_DAYS = 14
BASELINE_WINDOW_DAYS = 7
