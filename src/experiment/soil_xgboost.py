from dataclasses import dataclass
import pandas as pd

from src.fitting_window import create_dataset, add_time_features


@dataclass
class SoilXGBoostConfig:
    stride: int
    context_length: int
    prediction_length: int
    y_cols: list[str]


class SoilXGBoostExperiment:
    def __init__(self, df: pd.DataFrame, config: SoilXGBoostConfig):
        self.df = df
        self.config = config
        self._create_dataset()

    def _create_dataset(self):
        self.df = self.df.reset_index(drop=True)
        time_features = add_time_features(self.df[['time']], datetime_col='time')

        # needs to be after time features
        object_cols = self.df.select_dtypes(include='object').columns
        self.df.drop(object_cols, axis=1, inplace=True)

        self.df = pd.concat([self.df, time_features], axis=1)

        lagged_y = []
        for shift in range(self.config.context_length):
            lagged_y.append(self.df[self.config.y_cols].shift(shift + 1).rename(lambda x : f'{x}_{shift + 1}').reset_index(drop=True))

        to_concat = [self.df, *lagged_y]
        X = pd.concat(to_concat, axis=1)
        self.X = X.iloc[self.config.context_length:-self.config.prediction_length]
        self.X = self.X.drop('time', axis=1)

        y = self.df[self.config.y_cols].shift(-self.config.prediction_length)
        self.y = y.iloc[self.config.context_length:-self.config.prediction_length]

        assert len(self.X) == len(self.y)

    def __call__(self):
        return self.X, self.y


if __name__ == "__main__":
    df = pd.read_parquet("../../data/processed/merged_df.parquet")
    config = SoilXGBoostConfig(
        stride=1,
        context_length=24 * 7,
        prediction_length=24,
        y_cols=['SWC_5', 'SWC_10', 'SWC_20', 'SWC_50']
    )

    X, y = SoilXGBoostExperiment(df=df, config=config)()
