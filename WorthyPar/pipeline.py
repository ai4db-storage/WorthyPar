from forecaster import Forecaster
import argparse
import pandas as pd

DAY_SIZE = 24*60  #m
WEEK_SIZE = 7*24*60  #m

def parse_arg():
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', default='query_data.csv')
    args.add_argument('--path', default='./dataset')
    args.add_argument('--warm_up', default=0.75)

    args.add_argument('--monitor_interval')
    args.add_argument('--monitor_points')
    args.add_argument('--query_num', default=99)
    # params for forecast model
    args.add_argument('--predict_horizon', default=a, type=int, help='predict steps for each forecaster model')
    args.add_argument('--predict_interval', default=12, type=int, help='time interval for forecasting')
    args.add_argument('--epochs', default=30, type=int)
    # args for LSTM model
    args.add_argument('--lstm_input_chunk_length', default=24, type=int)
    args.add_argument('--hidden_dim', default=20, type=int)
    args.add_argument('--n_rnn_layer', default=2, type=int)
    args.add_argument('--training_length', default=24, type=int)
    # args for Linear model
    args.add_argument('--lags', default=2, type=int)
    # args for DLinear model
    args.add_argument('--DLinear_input_chunk_length', default=24, type=int)
    args.add_argument('--DLinear_output_chunk_length', default=12, type=int)
    # args for Transformer model
    args.add_argument('--Transformer_input_chunk_length', default=24, type=int)
    args.add_argument('--Transformer_output_chunk_length', default=12, type=int)
    # args for ARIMA model
    # for AutoARIMA model, there is no need for us to config model parameters ahead
    # args for Holt Winters
    args.add_argument('--seasonal_periods', default=288)
    args.add_argument('--damped', default=True)
    args = args.parse_args()


    return args

def run(a):
    args = parse_arg()

    # methods = ['transformer', 'DLinear']
    methods = ['linear']
    # weight_configs = {'transformer': 1, 'DLinear': 0}
    weight_configs = {'linear': 1}
    query_data = pd.read_csv(args.path + '/' + args.dataset)
    data_time = pd.date_range('2022-01-01', '2022-01-08', freq='5T')
    data_time = data_time[:-1]
    query_data = query_data.set_index(data_time, drop=True)
    data = query_data.drop(['time', 'Unnamed: 0'], axis=1)

    forecaster = Forecaster(args=args, methods=methods, weight_configs=weight_configs, data=data)

    forecaster.train()

    forecaster.forecast()


    mae_score, mse_score, rmse_score = forecaster.model_score()
    print(mae_score)
    print(mse_score)
    print(rmse_score)
    print("mae" + str(sum(mae_score[:13]) / 13))
    print("mse" + str(sum(mse_score[:13]) / 13))
    print("rmse" + str(sum(rmse_score[:13]) / 13))

    # forecaster.visualize(["5", "7", "55"])

    # forecaster.visualize_pie_forecast()

    # methods = ['liner']


if __name__ == '__main__':

    b = [3, 12, 24, 36, 144, 288, 526, 2016]
    for a in b:
        print("predict_horizon = ", a)
        run(a)