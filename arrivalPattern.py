import math
import random

import numpy as np
import matplotlib.pyplot as plt

"""
Arrival Patterns
[1] Cycles:
    ~ Cycles per day
    ~ Cycles per week
    ~ Cycles per week and day
[2] Stability:
[3] Spike:

"""


class ArrivalPattern(object):
    def __init__(self, pattern):
        self.pattern = pattern

    """
    Core problem:
        How to determine the value of center
    Returns: (numpy.array) generated arrival patterns
    """

    def stability(self, center, noise_type, noise_param, size):
        arrival_pattern = center * np.ones(shape=size)
        if noise_type == 'gaussian':
            mu, sigma, scale = noise_param
            arrival_pattern = arrival_pattern + scale * np.random.normal(mu, sigma, size)
            arrival_pattern = np.trunc(arrival_pattern)
        elif noise_type == 'random':
            low, high = noise_param
            arrival_pattern = arrival_pattern + np.random.randint(low, high, size)
        arrival_pattern[arrival_pattern < 0] = 0
        return arrival_pattern

    """
    Core problem:
        How to model the scale and lasting time of spikes
    Returns: (numpy.array) generated arrival patterns
    spike_param: spike_type, spike_extent_type, spike_last_time
    spike_type: 'cycle' or 'random'
    spike_extent_type
    spike_last_time: last_time_bound (ie.last_time_low, last_time_up) for spike_random
                     Python tuple of (cycle_period, shift, last_time_bound)
    spike_scale
    """

    def spike(self, center, noise_type, noise_param, spike_param, size):
        inputs = self.stability(center, noise_type, noise_param, size)
        spike_type, spike_extent_type, spike_last_time, spike_scale = spike_param
        if spike_type == 'cycle':
            arrival_patterns = self.spike_cycle(inputs, center, spike_extent_type, spike_last_time, spike_scale)
        elif spike_type == 'random':
            arrival_patterns = self.spike_random(inputs, center, spike_extent_type, spike_last_time, spike_scale)
        else:
            raise NotImplementedError
        return arrival_patterns

    def spike_cycle(self, series, center, spike_extent_type, spike_last_time, spike_scale):
        cycle_period, shift, last_time_bound = spike_last_time
        cursor = 0
        while (cursor + cycle_period - shift) < len(series):
            spike_time = cursor + cycle_period + random.randint(-shift, shift)
            if spike_extent_type == 'sigmoid':
                level = np.random.normal()
                spike = center * 1 / (1 + np.exp(level)) * spike_scale
                last_time = np.random.randint(low=last_time_bound[0], high=last_time_bound[1])
                Pi = math.pi
                for i in range(last_time):
                    if (i + spike_time) >= len(series):
                        break
                    series[i + spike_time] += spike * np.cos(Pi / (last_time * 2) * i)
                    # series[i+cursor] += spike * np.exp(-i/3)
            else:
                raise NotImplementedError
            cursor = spike_time
        return series

    def spike_random(self, series, center, spike_extent_type, spike_last_time_bound, spike_scale):
        cursor = 0
        while (cursor + spike_last_time_bound[0]) < len(series):
            spike_time = np.random.randint(low=spike_last_time_bound[0], high=len(series) - cursor)
            cursor += spike_time
            if spike_extent_type == 'sigmoid':
                level = np.random.normal()
                spike = center * 1 / (1 + np.exp(level)) * spike_scale
                last_time = np.random.randint(low=spike_last_time_bound[0], high=spike_last_time_bound[1])
                for i in range(last_time):
                    if (i + cursor) >= len(series):
                        break
                    series[i + cursor] += spike * np.exp(-i / 3)
            else:
                raise NotImplementedError
        return series

    """
    cycle_param : {periodicity1: [(center1, width1, scale1), ...], ...}
    """

    def cycle(self, cycle_type, cycle_param, bias, noise_type, noise_param, size):
        series = np.full(fill_value=float(bias), shape=size)
        if cycle_type == 'normal':
            for item in cycle_param.keys():
                series += self.cycle_unit(item, cycle_param[item], noise_type, noise_param, size)
        else:
            raise NotImplementedError
        return series

    """
    periodicity : int
    component_param : a Python list of (center, width, scale)
    modify the specific cyclical pattern by stacking multiple component_cycles
    """

    def cycle_unit(self, periodicity, component_param, noise_type, noise_param, size):

        def unit(periodicity, component_param):
            x = np.arange(periodicity)
            series = np.zeros(shape=periodicity)
            for item in component_param:
                center, width, scale = item
                series += self.component_cycle(center, width, scale, x)
            return series

        def unit_1(periodicity, component_param, noise_param):
            x = np.arange(periodicity)
            series = np.zeros(shape=periodicity)
            for item in component_param:
                center, width, scale = item
                series += (self.component_cycle(center, width, scale, x) + np.random.normal(noise_param[0],
                                                                                            noise_param[1],
                                                                                            periodicity))
            return series

        def unit_2(periodicity, component_param, noise_param):
            x = np.arange(periodicity)
            series = np.zeros(shape=periodicity)
            for item in component_param:
                center, width, scale = item
                series += (self.component_cycle(center, width, scale, x) + np.random.randint(noise_param[0],
                                                                                             noise_param[1],
                                                                                             periodicity))
            return series

        if noise_type == 'gaussian':
            series_unit = unit_1(periodicity, component_param, noise_param)
        elif noise_type == 'random':
            series_unit = unit_2(periodicity, component_param, noise_param)
        else:
            series_unit = unit(periodicity, component_param)
        if size <= periodicity:
            return series_unit[:size]
        series = series_unit
        num_units = int(np.trunc(size / periodicity))
        for i in range(num_units - 1):
            series = np.append(series, series_unit)

        remains = size - num_units * periodicity
        series = np.append(series, series_unit[:remains])
        return series

    def component_cycle(self, center, width, scale, x):
        width /= 2
        return scale / (np.sqrt(2 * np.pi) * width) * np.exp(-(x - center) ** 2 / (2 * width ** 2))

    def param_extractor(self, params):
        if self.pattern == 'stability':
            bias, noise_type, noise_param = params
            return self.pattern, noise_type, ('noise_param', bias, noise_param)
        elif 'spike' in self.pattern:
            bias, noise_type, noise_param, spike_param = params
            spike_type, spike_extent_type, spike_last_time, spike_scale = spike_param
            return self.pattern, noise_type, ('noise_param', bias, noise_param), \
                   ('spike_param', spike_extent_type, spike_last_time, spike_scale)
        elif self.pattern == 'chaos':
            method, method_param = params
            return self.pattern, method, ('chaos_param', method_param)
        elif self.pattern == 'cycle':
            cycle_type, cycle_param, bias, noise_type, noise_param = params
            return self.pattern, noise_type, ('noise_param', bias, noise_param), cycle_type, (
            'cycle_param', cycle_param)
        else:
            return

    def generate(self, params):
        if self.pattern == 'stability':
            center, noise_type, noise_param, size = params
            series = self.stability(center, noise_type, noise_param, size)
        elif 'spike' in self.pattern:
            center, noise_type, noise_param, spike_param, size = params
            series = self.spike(center, noise_type, noise_param, spike_param, size)
        elif self.pattern == 'cycle':
            cycle_type, cycle_param, bias, noise_type, noise_param, size = params
            series = self.cycle(cycle_type, cycle_param, bias, noise_type, noise_param, size)
        else:
            raise NotImplementedError
        return series

    def visualize(self, series):
        plt.plot(series)
        plt.xlabel('Time')
        plt.ylabel('Queries/h')
        plt.show()

# pattern = ArrivalPattern('stability')
# series = pattern.generate((30, 'gaussian', (0, 1), 1000))

# pattern = ArrivalPattern('spike')
# series = pattern.generate((30, 'gaussian', (0, 1), ('cycle', 'sigmoid', (100, 10, (10, 20)), 2), 1000))
# series = pattern.generate((30, 'gaussian', (0, 1), ('random', 'sigmoid', (10, 20), 2), 1000))

# pattern = ArrivalPattern('cycle')
# param = ('normal', {200: [(40, 15, 100), (120, 30, 80)], 500: [(300, 100, 200)]}, 30, 'gaussian', (0, 0.1), 3000)
# series = pattern.generate(param)
# pattern.visualize(series)
