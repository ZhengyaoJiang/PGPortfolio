import numpy as np
import pandas as pd
from pgportfolio.tdagent.algorithms.olmar import OLMAR

class RMR(OLMAR):
    '''universal-portfolio implementation'''
    def __init__(self, window=5, eps=10, tau=1e-3):
        super(RMR, self).__init__(window, eps)
        self.tau = tau

    def decide_by_history(self, x, last_b):
        self.record_history(x)
        close = pd.DataFrame(self.get_close())
        nx = close.iloc[-1,:]
        #print close.shape
        y = close.mean()
        y_last = None
        while y_last is None or norm(y-y_last)/norm(y_last)>self.tau:
            y_last=y
            d=norm(close-y)
            y = close.div(d, axis=0).sum() / (1./d).sum()
        return y/nx

def norm(x):
    if isinstance(x, pd.Series):
        axis=0
    else:
        axis=1
    return np.sqrt((x**2).sum(axis=axis))
