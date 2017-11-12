from ..tdagent import TDAgent
import numpy as np

class SP(TDAgent):
    '''Switch Portfolio'''
    def __init__(self, gamma=0.25, last_b=None):
        super(SP, self).__init__()
        self.gamma = gamma
        self.last_b = last_b

    def decide_by_history(self, x, last_b):
        self.record_history(x)
        nx = self.history[-1,:].ravel()
        if self.last_b is None:
            self.last_b = np.ones(nx.size) / nx.size
        b = self.last_b * (1-self.gamma-self.gamma/nx.size) + self.gamma/nx.size
        b = b / np.sum(b)
        self.last_b = b
        return b
