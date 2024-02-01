from darts.models import RNNModel, LinearRegressionModel, DLinearModel, TransformerModel, AutoARIMA, ExponentialSmoothing
from darts import TimeSeries
import numpy as np
# data : np.array
def normalize_np(data):
    std = np.std(data)
    mean = np.mean(data)
    normalized_data = (data-mean) / std
    return normalized_data, mean, std

# data : pd.DataFrame
def normalize_pd(data):
    std = data.std()
    mean = data.mean()
    normalized_data = (data-mean) / std
    return normalized_data, mean, std

class Base_Model(object):
    def __init__(self, args):
        self.args = args
        self.model = None
        self.data_config = dict()

    def fit(self, timeseries):
        timeseries, mean, std = normalize_pd(timeseries)
        self.data_config['mean'] = mean
        self.data_config['std'] = std
        ts = TimeSeries.from_dataframe(timeseries)
        self.model.fit(ts)

    def predict(self, timeseries):
        timeseries, _, _ = normalize_np(timeseries)
        pred = self.model.predict(n=self.args.predict_horizon, series=TimeSeries.from_dataframe(timeseries))
        pd_pred = pred.pd_dataframe()
        transform_pred = pd_pred * self.data_config['std'] + self.data_config['mean']
        return transform_pred, pd_pred


class LSTM_Model(Base_Model):
    def __init__(self, args):
        super(LSTM_Model, self).__init__(args)
        self.model = RNNModel(input_chunk_length=args.lstm_input_chunk_length,
                              model='LSTM',
                              hidden_dim=args.hidden_dim,
                              n_rnn_layers=args.n_rnn_layer,
                              training_length=args.training_length,
                              n_epochs=args.epochs)
        pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0]}

class Linear_Model(Base_Model):
    def __init__(self, args):
        super(Linear_Model, self).__init__(args)
        self.args = args
        self.model = LinearRegressionModel(lags=args.lags)

class DLinear_Model(Base_Model):
    def __init__(self, args):
        super(DLinear_Model, self).__init__(args)
        self.args = args
        self.model = DLinearModel(input_chunk_length=args.DLinear_input_chunk_length,
                                  output_chunk_length=args.DLinear_output_chunk_length,
                                  n_epochs=args.epochs)

class Transformer_Model(Base_Model):
    def __init__(self, args):
        super(Transformer_Model, self).__init__(args)
        self.args = args
        self.model = TransformerModel(input_chunk_length=args.Transformer_input_chunk_length,
                                      output_chunk_length=args.Transformer_output_chunk_length,
                                      n_epochs=args.epochs)

class ARIMA_Model(Base_Model):
    def __init__(self, args):
        super(ARIMA_Model, self).__init__(args)
        self.model = AutoARIMA()


class Holt_Winters_Model(Base_Model):
    def __init__(self, args):
        super(Holt_Winters_Model, self).__init__(args)
        self.model = ExponentialSmoothing(damped=args.damped, seasonal_periods=args.seasonal_periods)

