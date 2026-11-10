from datetime import datetime

import pandas as pd

from bikeforecast.baseline import same_hour_baseline
from bikeforecast.features import FEATURES, build_frame, season_of


def test_features_shape_and_weekday_convention():
    # 2011-01-01 was a Saturday -> dataset weekday 6, not a working day
    X = build_frame([datetime(2011, 1, 1, 9)], temp_c=10, humidity_pct=50,
                    windspeed_kmh=10, weather=1)
    assert list(X.columns) == FEATURES
    assert X.loc[0, "weekday"] == 6 and X.loc[0, "workingday"] == 0


def test_season():
    assert season_of(datetime(2012, 1, 5)) == 1
    assert season_of(datetime(2012, 7, 5)) == 3


def test_baseline_uses_only_past_days():
    days = pd.date_range("2012-01-01", periods=3)
    df = pd.DataFrame({"dteday": days, "hr": 8, "cnt": [10, 20, 1000]})
    b = same_hour_baseline(df, window=7)
    assert b.iloc[2] == 15          # mean of days 1-2, ignores day 3's own value
