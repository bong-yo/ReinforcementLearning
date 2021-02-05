import numpy as np
from typing import Tuple


class StateHandler:
    def __init__(self, os_low: int, os_high: int,
                 discrete_os_size: Tuple[int, int] = [1, 1]):
        self.os_low = os_low
        self.os_high = os_high
        self.os_size = abs((os_high - os_low))
        self.os_means = self.os_size / 2
        self.discrete_os_window = (os_high - os_low) / discrete_os_size

    def get_discrete_state(self, state):
        discrete = (state - self.os_low) // self.discrete_os_window
        return tuple(discrete.astype(np.int))

    def normalize_state(self, state):
        '''normalize every dimension of the state between 0 and 1, with avg 0'''
        return (state - self.os_means) / self.os_size
