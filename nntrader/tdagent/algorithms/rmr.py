from ..tdagent import TDAgent
import numpy as np

class RMR(TDAgent):
    ''' Robust Median Reversion

    Reference:


    '''
    def __init__(self, eps=5, W=5, b=None):
        '''
        :param eps: the parameter control the reversion threshold
        :pram W: the length of window
        '''
        super(RMR, self).__init__()
        self.eps = eps
        self.W = W
        self.b = b

    def decide_by_history(self, x, last_b):
        self.record_history(x)
        data_close = self.get_close()
        b = self.update(data_close, self.history, last_b, self.eps, self.W)
        return b

    def update(self, data_close, data, last_b, eps, W):
        t1 = data.shape[0]
        if t1 < W+2:
            x_t1 = data[t1-1, :]
        else:
            x_t1 = self.l1_median_VaZh(data_close[(t1-W):(t1-1),:]) / data_close[t1-1,:]

        if np.linalg.norm(x_t1 - x_t1.mean())**2 == 0:
            tao = 0
        else:
            tao = min(0, (x_t1.dot(last_b)-eps) / np.linalg.norm(x_t1 - x_t1.mean())**2)
        if self.b is None:
            self.b = np.ones(data.shape[1])/data.shape[1]
        else:
            self.b -= tao * (x_t1 - x_t1.mean() * np.ones(x_t1.shape))
            self.b = self.euclidean_proj_simplex(self.b)
        return self.b
