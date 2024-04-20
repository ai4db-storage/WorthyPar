import random
import numpy as np
from arrivalPattern import ArrivalPattern
from config import Config

DAY_SIZE = 24*60
WEEK_SIZE = 7*24*60
class Workload_Generator(object):

    def __init__(self, args, time_size, query_num):
        self.args = args
        self.time_size = time_size
        self.query_num = query_num

        self.templates = dict()
        self.configs = None

    def workload_generate(self):
        self.configs = Config(pattern=self.args.workload_pattern)

        query2pattern = None
        if self.args.fix_pattern:
            query2pattern = self.configs.fix_arrival_pattern(range(1, self.query_num+1))

            assert query2pattern is not None
        # print(sensor2pattern)
        # print(pattern2sensor)
        for idx in range(1, self.query_num+1):
            workload_pattern = query2pattern[idx]
            arrival_pattern, arrival_pattern_params = \
                self.arrival_pattern_generate(self.configs, workload_pattern)
            self.templates[idx] = dict()
            self.templates[idx]['arrival_pattern'] = (arrival_pattern, arrival_pattern_params)

    def arrival_pattern_generate(self, configs, workload_pattern):
        freq_pattern = np.random.choice(
            a=[key for key in configs.configuration['freq_pattern'].keys()],
            size=1,
            p=[prob for prob in configs.configuration['freq_pattern'].values()]
        )[0]

        if workload_pattern == 'stability':
            bias = random.randint(
                a=configs.configuration[workload_pattern][freq_pattern]['bias']['lower_bound'],
                b=configs.configuration[workload_pattern][freq_pattern]['bias']['upper_bound']
            )
            noise_type = 'gaussian'
            noise_param_holder = [value for value in
                                  configs.configuration[workload_pattern][freq_pattern]['noise'][noise_type].values()]
            noise_param = (noise_param_holder[0], noise_param_holder[1], noise_param_holder[2])
            params = (bias, noise_type, noise_param)

        elif workload_pattern == 'spike_random':
            bias = random.randint(
                a=configs.configuration[workload_pattern][freq_pattern]['bias']['lower_bound'],
                b=configs.configuration[workload_pattern][freq_pattern]['bias']['upper_bound']
            )
            noise_type = 'gaussian'
            noise_param = (value for value in
                           configs.configuration[workload_pattern][freq_pattern]['noise'][noise_type].values())
            spike_scale = configs.configuration[workload_pattern]['spike_scale']
            spike_extent_type = 'sigmoid'
            spike_type = 'random'
            spike_last_time_bound = (
                configs.configuration[workload_pattern]['spike_last_time']['lower_bound'],
                configs.configuration[workload_pattern]['spike_last_time']['upper_bound'],
            )
            params = (
                bias, noise_type, noise_param, (spike_type, spike_extent_type, spike_last_time_bound, spike_scale))

        elif workload_pattern == 'spike_cycle':
            bias = random.randint(
                a=configs.configuration[workload_pattern][freq_pattern]['bias']['lower_bound'],
                b=configs.configuration[workload_pattern][freq_pattern]['bias']['upper_bound']
            )
            noise_type = 'gaussian'
            noise_param = (value for value in
                           configs.configuration[workload_pattern][freq_pattern]['noise'][noise_type].values())
            spike_scale = configs.configuration[workload_pattern]['spike_scale']
            spike_extent_type = 'sigmoid'
            spike_type = 'cycle'
            spike_last_time_bound = (
                configs.configuration[workload_pattern]['spike_last_time']['lower_bound'],
                configs.configuration[workload_pattern]['spike_last_time']['upper_bound'],
            )
            cycle = np.random.choice(
                a=[key for key in configs.configuration[workload_pattern]['cycle_pattern'].keys()],
                size=1,
                p=[prob for prob in configs.configuration[workload_pattern]['cycle_pattern'].values()]
            )[0]
            cycle_period = configs.configuration[workload_pattern]['cycles'][cycle]
            shift = configs.configuration[workload_pattern]['shift'][cycle]
            params = (bias, noise_type, noise_param,
                      (spike_type, spike_extent_type, (cycle_period, shift, spike_last_time_bound), spike_scale))

        elif workload_pattern == 'cycle':
            cycle_type = 'normal'
            cycle_param = dict()
            bias = random.randint(
                a=configs.configuration[workload_pattern][freq_pattern]['bias']['lower_bound'],
                b=configs.configuration[workload_pattern][freq_pattern]['bias']['upper_bound']
            )
            noise_type = 'gaussian'
            noise_param_holder = [value for value in
                                  configs.configuration[workload_pattern][freq_pattern]['noise'][noise_type].values()]
            noise_param = (noise_param_holder[0], noise_param_holder[1], noise_param_holder[2])
            cycle_prob = random.random()
            if cycle_prob < 0.4:
                cycle_periods = ['day']
            elif cycle_prob < 0.6:
                cycle_periods = ['week']
            else:
                cycle_periods = ['day', 'week']
            for cycle in cycle_periods:
                cycle_size = configs.configuration[workload_pattern]['cycles'][cycle]
                unit_interval = cycle_size / len(configs.configuration[workload_pattern]['centers'][cycle].keys())
                cycle_param[cycle_size] = []
                period_num = min(np.random.zipf(a=2, size=1)[0], 3)
                temp = [prob for prob in configs.configuration[workload_pattern]['centers'][cycle].values()]
                temp = temp / np.sum(temp)
                centers = np.random.choice(
                    a=[key for key in configs.configuration[workload_pattern]['centers'][cycle].keys()],
                    p=temp,
                    size=period_num
                )
                for unit in range(period_num):
                    center = centers[unit]
                    shift = configs.configuration[workload_pattern]['shift'][cycle]
                    center = random.randint(
                        a=max(0, center * unit_interval - shift),
                        b=min(cycle_size, center * unit_interval + shift))
                    width = random.randint(
                        a=configs.configuration[workload_pattern]['width'][cycle]['lower_bound'],
                        b=configs.configuration[workload_pattern]['width'][cycle]['upper_bound']
                    )
                    scale = random.randint(
                        a=configs.configuration[workload_pattern][freq_pattern]['scale'][cycle]['lower_bound'],
                        b=configs.configuration[workload_pattern][freq_pattern]['scale'][cycle]['upper_bound']
                    )
                    cycle_param[cycle_size].append((center, width, scale))
            params = (cycle_type, cycle_param, bias, noise_type, noise_param)

        else:
            raise NotImplementedError

        return workload_pattern, params