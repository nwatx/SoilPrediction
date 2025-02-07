from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel, TimeSeriesTransformerForPrediction


class HFTimeSeriesTransformer:
    def __init__(self):
        self.config = TimeSeriesTransformerConfig(prediction_length=7 * 24,
                                                  context_length=7 * 24 * 2)
        self.model = TimeSeriesTransformerForPrediction(self.config)

    def __call__(self):
        return self.model