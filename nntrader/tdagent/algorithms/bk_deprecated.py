# -*- coding: utf-8 -*-
from ..tdagent import TDAgent
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging


class BK(TDAgent):
    """ Kernel based strategy. It tries to find similar sequences of price in history and then maximize objective function (that is profit) on the days following them.

    Reference:
        L. Gyorfi, G. Lugosi, and F. Udina. Nonparametric kernel based sequential
        investment strategies. Mathematical Finance 16 (2006) 337–357.
    """
    def __init__(self, k=5, l=10):
        """
        :param k: Sequence length.
        :param l: Number of nearest neighbors.
        """

        super(BK, self).__init__()
        self.k = k
        self.l = l

    def decide_by_history(self, x, last_b):
        self.record_history(x)
        history = pd.DataFrame(self.history)
        # find indices of nearest neighbors throughout history
        ixs = self.find_nn(history, self.k, self.l)

        # get returns from the days following NNs
        J = history.iloc[[history.index.get_loc(i) + 1 for i in ixs]]

        # get best weights
        return opt_weights(J)


    def find_nn(self, H, k, l):
        """ Note that nearest neighbors are calculated in a different (more efficient) way than shown
        in the article.

        param H: history
        """
        # calculate distance from current sequence to every other point
        D = H * 0
        for i in range(1, k+1):
            D += (H.shift(i-1) - H.iloc[-i])**2
        D = D.sum(1).iloc[:-1]

        # sort and find nearest neighbors
        D.sort_values(inplace=True)
        return D.index[:l]


def opt_weights(X, max_leverage=1):
    x_0 = max_leverage * np.ones(X.shape[1]) / float(X.shape[1])
    objective = lambda b: -np.sum(np.log(np.maximum(np.dot(X-1, b)+1,1e-4)))
    cons = ({'type': 'eq', 'fun': lambda b: max_leverage-sum(b)},)
    bnds = [(0., max_leverage)]*len(x_0)
    while True:
        res = minimize(objective, x_0, bounds=bnds, constraints=cons, method='slsqp')
        eps = 1e-7
        if (res.x < 0-eps).any() or (res.x > max_leverage+eps).any():
            X = X + np.random.randn(1)[0] * 1e-5
            logging.debug('Optimal weights not found, trying again')
            continue
        elif res.success:
            break
        else:
            if np.isnan(res.x).any():
                logging.warning('Solution not found')
                res.x = np.ones(X.shape[1]) / X.shape[1]
            else:
                logging.warning("converged but not successfully")
            break

    return res.x
