import numpy as np
import math
import time
import os
import random

import pandas as pd

from forecaster_models import LSTM_Model, Linear_Model, DLinear_Model, Transformer_Model, ARIMA_Model, Holt_Winters_Model
from darts import TimeSeries
from darts import TimeSeries
import matplotlib.pyplot as plt

def normalize_pd(data):
    std = data.std()
    mean = data.mean()
    normalized_data = (data-mean) / std
    return normalized_data, mean, std

def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def mse(y_true, y_pred):
    return np.mean(np.square(np.abs(y_pred - y_true)))

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

class Forecaster(object):
    def __init__(self, args, methods, weight_configs, data):
        self.args = args
        self.data = data
        self.train_num = int(self.data.shape[0] * self.args.warm_up)
        self.methods = methods
        self.weight_configs = weight_configs

        self.deployed_models = dict()
        self.real_predict = []
        self.normalize_predict = []
        self.state = 'waiting for training'


    def get_model(self, method):
        if method == 'lstm':
            return LSTM_Model(self.args)
        elif method == 'linear':
            return Linear_Model(self.args)
        elif method == 'transformer':
            return Transformer_Model(self.args)
        elif method == 'DLinear':
            return DLinear_Model(self.args)
        elif method == 'ARIMA':
            return ARIMA_Model(self.args)
        elif method == 'Holt_Winters':
            return Holt_Winters_Model(self.args)
        else:
            raise NotImplementedError

    def train(self):
        assert self.state == 'waiting for training'

        train_data = self.data[:self.train_num]
        for method in self.methods:
            model = self.get_model(method)
            start_time = time.time()
            model.fit(train_data)
            self.deployed_models[method] = model
            print('  training {} model ----- time consume {}'.format(method, time.time()-start_time))

        self.state = 'waiting for prediction'

    # Note : output of predict is normalized series
    #        output of transform_predict is transformed to the real series
    def predict(self, method, test, weight_config):
        assert self.state == 'waiting for prediction'
        transform_predict, predict = self.deployed_models[method].predict(timeseries=test)
        ac = predict
        predicts = np.zeros(shape=(self.args.predict_horizon, 99))
        for method in self.methods:
          predicts += np.array(ac) * weight_config[method]
        predicts = list(predicts)
        self.real_predict.append(transform_predict)
        self.normalize_predict.append(predicts)


    # def ensemble(self, ac, weight_config):
    #     predicts = np.zeros(shape=self.args.predict_horizon)
    #     for method in self.methods:
    #         predicts += np.array(ac) * weight_config[method]
    #     predicts = list(predicts)
    #     return predicts
    """
    A wrapping version of self.predict
    """
    def forecast(self):
        series_len = self.data.shape[0]
        current = self.train_num
        weight_config = self.weight_configs
        while current < series_len - 288:
            test = self.data[current - 288: current]
            for method in self.methods:
                self.predict(method, test, weight_config)
                current += self.args.predict_horizon
            # ac = self.predict(self.methods, test)
            # predicts = self.ensemble(ac, self.weight_configs)
            # return predicts



    def score(self, predicts, left, right):

        mae_score = 0
        mse_score = 0
        rmse_score = 0
        for idx in range(self.args.query_num):
            y_true = self.data[left:right][str(idx+1)].to_numpy()
            y_pred = predicts[str(idx+1)].to_numpy()
            mae_score += mae(y_true, y_pred)
            mse_score += mse(y_true, y_pred)
            # rmse_score += rmse(y_true, y_pred)

        mae_score = mae_score / self.args.query_num
        mse_score = mse_score / self.args.query_num
        rmse_score = math.sqrt(mse_score)

        return mae_score, mse_score, rmse_score

    def model_score(self):
        series_len = self.data.shape[0]
        current = self.train_num
        idx = 0
        max_num = len(self.normalize_predict)
        mae = []
        mse = []
        rmse = []
        while current < series_len - 288 and idx < max_num:
            left = current
            right = current + self.args.predict_horizon
            pd_real_pred = self.real_predict[idx]
            mae_score, mse_score, rmse_score = self.score(pd_real_pred, left ,right)
            mae.append(mae_score)
            mse.append(mse_score)
            rmse.append(rmse_score)
            current += self.args.predict_horizon
            idx+=1

        return mae, mse, rmse

    def visualize(self, query_number):
        forecast_query_number = []
        for num in query_number:
            forecast_query_number.append("forecast-"+num)
        series_len = self.data.shape[0]
        current = self.train_num
        idx = 0
        max_num = len(self.normalize_predict)
        while current < series_len - 288 and idx < max_num:
            pd_ts_pred = self.normalize_predict[idx]
            ts_pred = TimeSeries.from_dataframe(pd_ts_pred[query_number])
            ts_pred.plot(label=forecast_query_number)
            pd_ts_actual, _, _ = normalize_pd(self.data[current - 288: current + self.args.predict_horizon])
            ts_actual = TimeSeries.from_dataframe(pd_ts_actual[query_number])
            ts_actual.plot()
            plt.show()
            current += self.args.predict_horizon
            idx+=1

    def visualize_pie(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        series_len = self.data.shape[0]
        current = self.train_num
        idx = 0
        max_num = len(self.real_predict)
        index = ['real', 'forecast']
        columns = []
        for id in range(99):
            columns.append(str(id+1))
        while current < series_len - 288 and idx < max_num:
            pd_ts_pred = self.real_predict[idx]
            ts_pred_lst = pd_ts_pred.mean().tolist()
            pd_ts_actual = self.data[current: current + self.args.predict_horizon]
            ts_actual_lst = pd_ts_actual.mean().tolist()
            data = []
            data.append(ts_actual_lst)
            data.append(ts_pred_lst)
            df = pd.DataFrame(data, columns=columns, index=index)
            df = df.T
            fig1 = plt.figure()
            sp1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
            sp1.set_title('real vs forecast')
            sp1.pie(df['real'], labels=df.index, autopct='%.1f%%', startangle=90,
                    wedgeprops={'width': 0.3, 'edgecolor': 'w', 'linewidth': 5}, pctdistance=0.85)
            sp1.pie(df['forecast'], labels=df.index, autopct='%.1f%%', startangle=90,
                    wedgeprops={'width': 0.3, 'edgecolor': 'w', 'linewidth': 5}, pctdistance=0.80)
            sp1.legend()
            plt.show()
            current += self.args.predict_horizon
            idx+=1

    def visualize_pie_forecast(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        series_len = self.data.shape[0]
        current = self.train_num
        idx = 0
        max_num = len(self.real_predict)
        index = ['forecast']
        while current < series_len - 288 and idx < max_num:
            pd_ts_pred = self.real_predict[idx]
            ts_pred_lst = pd_ts_pred.mean().tolist()
            xx = np.asarray(ts_pred_lst)
            top_idx = xx.argsort()[-1:-11:-1]
            TopBER = xx[top_idx]
            TopBER = TopBER.tolist()
            other = sum(ts_pred_lst) - sum(TopBER)
            TopBER.append(other)
            columns = []
            for id in top_idx:
                columns.append(str(id+1))
            columns.append("others")
            data = []
            data.append(TopBER)
            df = pd.DataFrame(data, columns=columns, index=index)
            df = df.T
            fig1 = plt.figure()
            sp1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
            sp1.set_title('forecast')
            sp1.pie(df['forecast'], labels=df.index, autopct='%.1f%%', startangle=90,
                    wedgeprops={'width': 0.3, 'edgecolor': 'w', 'linewidth': 5}, pctdistance=0.85)
            sp1.legend()
            plt.show()
            current += self.args.predict_horizon
            idx+=1