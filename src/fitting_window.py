from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

@dataclass
class TimeHorizon:
    start: pd.Timestamp
    end: pd.Timestamp

class RollingFitter:
    def __init__(self, df: pd.DataFrame, time_horizon=None):
        self.time_horizon: Optional[TimeHorizon] = time_horizon
        self.df = df

def create_dataset(values, time_features, context_length, prediction_length, stride=1, drop_vals=['time']):
    """
    Create a dataset by sliding a window over the time series.
    Returns arrays:
      - past: (num_examples, context_length, input_size)
      - past_time: (num_examples, context_length, num_time_features)
      - future: (num_examples, prediction_length, input_size)
      - future_time: (num_examples, prediction_length, num_time_features)
    """
    values = values.drop(drop_vals, axis=1)
    X_past = []
    X_past_time = []
    X_future = []
    X_future_time = []
    total_length = context_length + prediction_length
    for i in range(0, len(values) - total_length + 1, stride):
        past = values[i: i + context_length]
        future = values[i + context_length: i + total_length]
        past_time = time_features[i: i + context_length]
        future_time = time_features[i + context_length: i + total_length]
        X_past.append(past)
        X_future.append(future)
        X_past_time.append(past_time)
        X_future_time.append(future_time)
    return (np.array(X_past), np.array(X_past_time),
            np.array(X_future), np.array(X_future_time))

def add_time_features(df, datetime_col):
    """
    Given a dataframe containing a datetime column, extract cyclic time features.
    Returns a new dataframe with only the generated time features.
    """
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Extract basic time features
    df.loc[:, "hour"] = df[datetime_col].dt.hour
    df.loc[:, "dayofweek"] = df[datetime_col].dt.dayofweek
    df.loc[:, "month"] = df[datetime_col].dt.month

    # Cyclical encoding for the features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Return only the time features (do not drop the original time column here)
    return df[["hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "month_sin", "month_cos"]]