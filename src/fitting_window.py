from dataclasses import dataclass
from typing import Optional

import pandas as pd

@dataclass
class TimeHorizon:
    start: pd.Timestamp
    end: pd.Timestamp

class RollingFitter:
    def __init__(self, df: pd.DataFrame, time_horizon=None):
        self.time_horizon: Optional[TimeHorizon] = time_horizon
        self.df = df