from darts.models import RNNModel, LinearRegressionModel
import matplotlib.pyplot as plt
from arrivalPattern import ArrivalPattern
from darts import TimeSeries
import time
import numpy as np
import pandas as pd

def normalize_pd(data):
    std = data.std()
    mean = data.mean()
    normalized_data = (data-mean) / std
    return normalized_data, mean, std
pattern = ArrivalPattern('cycle')
param = ('normal', {200: [(40, 15, 100), (120, 30, 80)], 500: [(300, 100, 200)]}, 30, 'gaussian', (0, 0.1), 3000)
series = pattern.generate(param)
# pattern = ArrivalPattern('stability')
# series = pattern.generate((30, 'gaussian', (0, 1), 3000))
query_data = pd.read_csv('./dataset/query_data_test.csv')
data = pd.date_range('2022-01-01', '2022-01-08', freq='5T')
data = data[:-1]
query_data = query_data.set_index(data, drop=True)
# print(query_data.mean())
# print(type(query_data.mean()))
series_len = query_data.shape[0]
warm_up = 1200
predict_interval = 12
time_series = query_data.drop(['time'], axis=1)
time_series, mean, std = normalize_pd(time_series)
train_data = time_series[:warm_up]
print(train_data)
ts_train = TimeSeries.from_dataframe(train_data)

model = RNNModel(input_chunk_length=12,
                 model='LSTM',
                 hidden_dim=20,
                 n_rnn_layers=2,
                 training_length=24,
                 n_epochs=30
                 )
linear = LinearRegressionModel(lags=32)
pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0]}
print('Model training start')
start_time = time.time()
model.fit(ts_train)
# linear.fit(ts_train)
print('Model training consume {}'.format(time.time()-start_time))
current = warm_up
while current < series_len - 288:
    # ts_train.plot()
    test = time_series[current-288: current]
    ts_test = TimeSeries.from_dataframe(test)
    ts_pred = model.predict(n=predict_interval, series=ts_test)
    pd_ts_pred = ts_pred.pd_dataframe()
    print(pd_ts_pred)
    ts_pred = TimeSeries.from_dataframe(pd_ts_pred[['1', '9']])
    ts_pred.plot()
    ts_actual = TimeSeries.from_dataframe(time_series[current-288: current+predict_interval])
    pd_ts_actual = ts_actual.pd_dataframe()
    ts_actual = TimeSeries.from_dataframe(pd_ts_actual[['1', '9']])
    ts_actual.plot(label=['forecast-1', 'forecast-9'])
    plt.show()
    current += predict_interval