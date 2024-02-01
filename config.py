import numpy as np


class Config(object):
    def __init__(self, pattern):
        self.pattern = pattern
        self.configuration = dict()
        if pattern == 'Cycle':
            self.configuration['arrival_pattern'] = {
                'cycle': 0.72,
                'stability': 0.21,
                'spike_random': 0.02,
                'spike_cycle': 0.05,
            }

            self.configuration['freq_pattern'] = {
                'high': 0.8, 'mid': 0.1, 'low': 0.1
            }

            self.configuration['time_pattern'] = {
                'range': 0.6, 'till_now': 0.4
            }

            """
            configs for stability
            """
            self.configuration['stability'] = {
                'high': {
                    'bias': {'upper_bound': 200, 'lower_bound': 100},
                    'noise': {'gaussian': {'mean': 0.1, 'std': 0.1, 'scale': 30}}
                },
                'mid': {
                    'bias': {'upper_bound': 80, 'lower_bound': 40},
                    'noise': {'gaussian': {'mean': 0.1, 'std': 0.1, 'scale': 20}}
                },
                'low': {
                    'bias': {'upper_bound': 20, 'lower_bound': 10},
                    'noise': {'gaussian': {'mean': 0.2, 'std': 0.1, 'scale': 10}}
                }
            }

            """
            configs for spike_random
            """
            self.configuration['spike_random'] = {
                'high': {
                    'bias': {'upper_bound': 200, 'lower_bound': 100},
                    'noise': {'gaussian': {'mean': 0, 'std': 0.1, 'scale': 50}},
                },
                'mid': {
                    'bias': {'upper_bound': 80, 'lower_bound': 40},
                    'noise': {'gaussian': {'mean': 0, 'std': 0.1, 'scale': 20}},
                },
                'low': {
                    'bias': {'upper_bound': 20, 'lower_bound': 10},
                    'noise': {'gaussian': {'mean': 0, 'std': 0.1, 'scale': 10}},
                },
                'spike_scale': 3,
                'spike_last_time': {'upper_bound': 12, 'lower_bound': 3}
            }

            """
            configs for spike_cycle
            """
            self.configuration['spike_cycle'] = {
                'high': {
                    'bias': {'upper_bound': 200, 'lower_bound': 100},
                    'noise': {'gaussian': {'mean': 0, 'std': 0.1, 'scale': 50}},
                },
                'mid': {
                    'bias': {'upper_bound': 800, 'lower_bound': 400},
                    'noise': {'gaussian': {'mean': 0, 'std': 0.1, 'scale': 20}},
                },
                'low': {
                    'bias': {'upper_bound': 200, 'lower_bound': 100},
                    'noise': {'gaussian': {'mean': 0, 'std': 0.1, 'scale': 10}},
                },
                'spike_scale': 3,
                'spike_last_time': {'upper_bound': 10, 'lower_bound': 4},
                'cycle_pattern': {'day': 0.4, 'week': 0.6},
                'cycles': {'day': 288, 'week': 2016},
                'shift': {'day': 6, 'week': 36}
            }

            """
            configs for cycle
            """
            self.configuration['cycle'] = {
                'high': {
                    'bias': {'upper_bound': 300, 'lower_bound': 150},
                    'noise': {'gaussian': {'mean': 0, 'std': 0.1, 'scale': 40}},
                    'scale': {'day': {'upper_bound': 2200, 'lower_bound': 600},
                              'week': {'upper_bound': 4400, 'lower_bound': 1200}}
                },
                'mid': {
                    'bias': {'upper_bound': 100, 'lower_bound': 50},
                    'noise': {'gaussian': {'mean': 0, 'std': 0.1, 'scale': 20}},
                    'scale': {'day': {'upper_bound': 600, 'lower_bound': 150},
                              'week': {'upper_bound': 1800, 'lower_bound': 300}}
                },
                'low': {
                    'bias': {'upper_bound': 30, 'lower_bound': 10},
                    'noise': {'gaussian': {'mean': 0, 'std': 0.1, 'scale': 10}},
                    'scale': {'day': {'upper_bound': 150, 'lower_bound': 60},
                              'week': {'upper_bound': 600, 'lower_bound': 120}}
                },
                'cycles': {'day': 288, 'week': 2016},
                'width': {'day': {'upper_bound': 24, 'lower_bound': 6},
                          'week': {'upper_bound': 48, 'lower_bound': 12}},
                'shift': {'day': 2, 'week': 8},
                'centers': {'day': {0: 2, 1: 1, 2: 2, 3: 2, 4: 1, 5: 2, 6: 4, 7: 8,
                                    8: 12, 9: 15, 10: 12, 11: 8, 12: 7, 13: 5, 14: 5, 15: 6, 16: 7,
                                    17: 9, 18: 12, 19: 13, 20: 12, 21: 8, 22: 7, 23: 4},
                            'week': {1: 2, 2: 1, 3: 1, 4: 1, 5: 3, 6: 12, 7: 8}}
            }

        else:
            raise NotImplementedError

    """
    约束各个模板的到达率模式
    """
    def fix_arrival_pattern(self, tags_value):
        template_arrival_map = dict()
        for tag in tags_value:
            workload_pattern = np.random.choice(
                a=[key for key in self.configuration['arrival_pattern'].keys()],
                size=1,
                p=[prob for prob in self.configuration['arrival_pattern'].values()])[0]
            template_arrival_map[tag] = workload_pattern
        return template_arrival_map

