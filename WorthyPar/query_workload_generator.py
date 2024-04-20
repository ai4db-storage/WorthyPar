from arrivalPattern import ArrivalPattern
from workload_generator import Workload_Generator
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import random

DAY_SIZE = 24*60  #min
WEEK_SIZE = 7*24*60  #min

class Query_Workload_Generator(object):
    def __init__(self, args):
        self.args = args
        if args.time_size == 'DAY_SIZE':
            self.time_size = DAY_SIZE
        elif args.time_size == 'WEEK_SIZE':
            self.time_size = WEEK_SIZE
        self.query_num = args.query_num

        self.temp_workload_generator = Workload_Generator(args, self.time_size, self.query_num)
        self.query_log = dict()

    def query_generate(self):

        # 调用Workload_Generator类生成查询对应的到达率信息
        self.temp_workload_generator.workload_generate()

        interval_value = int(re.findall(r"\d+", self.args.monitor_interval)[0])
        interval_unit = re.findall(r'[A-Za-z]', self.args.monitor_interval)[0]

        for idx in range(1, self.query_num+1):
            arrival_pattern, arrival_rate_param = self.temp_workload_generator.templates[idx]['arrival_pattern']
            size = int(self.time_size / interval_value)

            arrival_rate_curve = self.arrival_rate_curve_generate(arrival_pattern, arrival_rate_param, size)
            self.query_log[idx] = arrival_rate_curve

    def arrival_rate_curve_generate(self, pattern, params, size):
        arrival_rate_generator = ArrivalPattern(pattern)
        if pattern == 'stability':
            bias, noise_type, noise_param = params
            series = arrival_rate_generator.generate(params=(bias, noise_type, noise_param, size))
        elif 'spike' in pattern:
            bias, noise_type, noise_param, spike_param = params
            series = arrival_rate_generator.generate(params=(bias, noise_type, noise_param, spike_param, size))
        elif pattern == 'chaos':
            method, method_param = params
            series = arrival_rate_generator.generate(params=(method, method_param, size))
        elif pattern == 'cycle':
            cycle_type, cycle_param, bias, noise_type, noise_param = params
            series = arrival_rate_generator.generate(
                params=(cycle_type, cycle_param, bias, noise_type, noise_param, size))
        else:
            raise NotImplementedError
        series[series < 0] = 0
        series[series == np.nan] = 0
        return series

    def query_visualize(self):
        for temp_idx in self.query_log.keys():
            for query in self.query_log[temp_idx]:
                print(query)
                print()

    def query_stats(self):
        for temp_idx in self.query_log.keys():
            plt.plot(self.query_log[temp_idx])
            plt.xlabel('Time')
            plt.ylabel('Queries/5min')
            plt.show()

    def save_queries(self, path):
        interval_value = int(re.findall(r"\d+", self.args.monitor_interval)[0])

        df = pd.DataFrame(columns=['time'], data=[[0]])
        for i in range(int(self.time_size/interval_value) - 1):
            df = df.append({'time': (i + 1) * interval_value}, ignore_index=True)

        for temp_idx in self.query_log.keys():
            temp_data = pd.Series(self.query_log[temp_idx])
            df.insert(df.shape[1], str(temp_idx), temp_data, allow_duplicates=False)

        df.to_csv(path + '/query_data.csv')
        # print(df)

    def save_queries_np(self, path):
        lst = []
        for temp_idx in self.query_log.keys():
            lst.append(self.query_log[temp_idx])

        np.savetxt(path + '/query_data_np.csv', lst, delimiter=",")
def shift_list(arr):
    temp_arr = []
    for i in range(len(arr[0])):
        b = []
        for j in range(len(arr)):
            b.append(arr[j][i])
        temp_arr.append(b)
    return temp_arr