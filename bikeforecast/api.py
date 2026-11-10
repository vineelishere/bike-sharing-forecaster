from contextlib import asynccontextmanager
from datetime import date as Date, datetime, timedelta
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .config import MODEL_DIR
from .db import DailyMetric, get_engine
from .features import build_frame

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = MODEL_DIR / "model.joblib"
    if not path.exists():
        raise RuntimeError("No model found. Run: python -m bikeforecast.train")
    _state["model"] = joblib.load(path)
    yield


app = FastAPI(title="Bike Demand Forecaster", version="1.0", lifespan=lifespan)


class Weather(BaseModel):
    temp_c: float = Field(..., ge=-8, le=39, description="Air temperature, °C")
    feels_like_c: Optional[float] = Field(None, ge=-16, le=50)
    humidity_pct: float = Field(..., ge=0, le=100)
    windspeed_kmh: float = Field(..., ge=0, le=67)
    weather: int = Field(1, ge=1, le=4,
                         description="1 clear/few clouds, 2 mist/cloudy, 3 light rain/snow, 4 heavy rain/snow")
    holiday: bool = False


class HourRequest(Weather):
    timestamp: datetime = Field(..., examples=["2012-12-20T08:00:00"])


class DayRequest(Weather):
    date: Date = Field(..., examples=["2012-12-20"])


class HourResponse(BaseModel):
    timestamp: datetime
    predicted_rentals: float


class DayResponse(BaseModel):
    date: Date
    total_predicted_rentals: float
    peak_hour: int
    hourly: List[HourResponse]


def _predict(timestamps, w: Weather):
    X = build_frame(timestamps, temp_c=w.temp_c, humidity_pct=w.humidity_pct,
                    windspeed_kmh=w.windspeed_kmh, weather=w.weather,
                    feels_like_c=w.feels_like_c, holiday=w.holiday)
    return np.clip(_state["model"].predict(X), 0, None)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/hour", response_model=HourResponse)
def predict_hour(req: HourRequest):
    p = _predict([req.timestamp], req)[0]
    return HourResponse(timestamp=req.timestamp, predicted_rentals=round(float(p), 1))


@app.post("/predict/day", response_model=DayResponse)
def predict_day(req: DayRequest):
    """Predicts all 24 hours, assuming the given weather holds for the whole day."""
    start = datetime.combine(req.date, datetime.min.time())
    stamps = [start + timedelta(hours=h) for h in range(24)]
    preds = _predict(stamps, req)
    return DayResponse(
        date=req.date,
        total_predicted_rentals=round(float(preds.sum()), 1),
        peak_hour=int(preds.argmax()),
        hourly=[HourResponse(timestamp=t, predicted_rentals=round(float(p), 1))
                for t, p in zip(stamps, preds)],
    )


@app.get("/monitoring/status")
def monitoring_status():
    """Latest model-health reading from the replay job stored in Postgres."""
    try:
        with Session(get_engine()) as s:
            last = s.query(DailyMetric).order_by(DailyMetric.id.desc()).first()
    except Exception:
        raise HTTPException(503, "Database unavailable")
    if last is None:
        raise HTTPException(404, "No replay run yet. Run: python -m bikeforecast.replay")
    return {
        "run_id": last.run_id,
        "latest_day": last.day,
        "rolling_mae": round(last.rolling_mae, 2),
        "error_limit": last.error_limit,
        "error_trend_per_day": round(last.error_trend, 3),
        "model_flagged": last.flagged,
    }
