# -*- coding: utf-8 -*-
from ..tdagent import TDAgent
import numpy as np

class EG(TDAgent):
    """ Exponentiated Gradient (EG) algorithm by Helmbold et al.

    Reference:
        Helmbold, David P., et al.
        "On‐Line Portfolio Selection Using Multiplicative Updates."
        Mathematical Finance 8.4 (1998): 325-347.
        http://www.cis.upenn.edu/~mkearns/finread/helmbold98line.pdf
    """

    def __init__(self, eta=0.05, b=None, last_b=None):
        """
        :params eta: Learning rate. Controls volatility of weights.
        """
        super(EG, self).__init__()
        self.eta = eta
        self.b = b
        self.last_b = last_b

    def init_pw(self, x):
        self.b = np.ones(x.size)

    def decide_by_history(self, x, last_b):
        self.record_history(x)
        x = self.history[-1,:].ravel()
        if self.last_b is None:
            self.last_b = np.ones(x.size) / x.size
        if self.b is None:
            self.init_pw(x)
        else:
            self.b = self.last_b * np.exp(self.eta * x.T / np.dot(x,last_b))
        b = self.b / np.sum(self.b)
        self.last_b = b
        return b

