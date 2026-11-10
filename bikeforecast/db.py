from sqlalchemy import (Boolean, Column, Date, DateTime, Float, Integer, String,
                        create_engine)
from sqlalchemy.orm import declarative_base

from .config import DATABASE_URL

Base = declarative_base()


class Prediction(Base):
    """One row per predicted hour, with the actual value once it is revealed."""
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    run_id = Column(String, index=True, nullable=False)
    model_version = Column(String)
    ts = Column(DateTime, index=True, nullable=False)
    predicted = Column(Float, nullable=False)
    baseline = Column(Float)
    actual = Column(Float)
    abs_error = Column(Float)


class DailyMetric(Base):
    """One row per replayed day: accuracy tracking + drift flag."""
    __tablename__ = "daily_metrics"
    id = Column(Integer, primary_key=True)
    run_id = Column(String, index=True, nullable=False)
    day = Column(Date, nullable=False)
    mae = Column(Float)
    baseline_mae = Column(Float)
    rolling_mae = Column(Float)
    error_trend = Column(Float)      # slope of daily MAE over the last N days
    error_limit = Column(Float)
    flagged = Column(Boolean, default=False)


def get_engine(url: str = DATABASE_URL):
    return create_engine(url, future=True, pool_pre_ping=True)


def init_db(engine):
    Base.metadata.create_all(engine)
