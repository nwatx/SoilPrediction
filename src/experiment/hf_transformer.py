from dataclasses import dataclass
from typing import Optional, List

import pandas as pd
import torch
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction

from src.fitting_window import create_dataset, add_time_features


@dataclass
class HFExperimentConfig:
    stride: int
    static_categorical_features: Optional[List[str]]
    transformer_config: TimeSeriesTransformerConfig


class HFExperiment:
    def __init__(self, df: pd.DataFrame, config: HFExperimentConfig):
        self.df = df
        self.config = config

    def run(self):
        time_features = add_time_features(self.df[['time']], datetime_col='time').values
        dataset = create_dataset(values=self.df,
                                 time_features=time_features,
                                 context_length=self.config.transformer_config.context_length,
                                 prediction_length=self.config.transformer_config.prediction_length,
                                 stride=self.config.stride)

        X_past, X_past_time, X_future, X_future_time = dataset
        X_past, X_past_time = torch.tensor(X_past), torch.tensor(X_past_time)
        X_future, X_future_time = torch.tensor(X_future), torch.tensor(X_future_time)

        past_observed_mask = torch.ones_like(X_past)
        model = TimeSeriesTransformerForPrediction(config=self.config.transformer_config)
        outputs = model(
            past_values=X_past,
            past_time_features=X_past_time,
            past_observed_mask=past_observed_mask,
            static_categorical_features=self.config.static_categorical_features,
            future_values=X_future,
            future_time_features=X_future_time
        )

        loss = outputs.loss
        loss.backward()
        print(loss)

if __name__ == "__main__":
    df = pd.read_parquet("../../data/processed/merged_df.parquet")
    object_cols = df.select_dtypes(include='object').columns
    df.drop(object_cols, axis=1, inplace=True)

    config = HFExperimentConfig(
        stride=24,
        static_categorical_features=None,
        transformer_config=TimeSeriesTransformerConfig(
            prediction_length=24,
            context_length=24 * 7 * 2,
            num_time_features=6,
            input_size=24,
        )
    )

    experiment = HFExperiment(df=df, config=config)
    experiment.run()
