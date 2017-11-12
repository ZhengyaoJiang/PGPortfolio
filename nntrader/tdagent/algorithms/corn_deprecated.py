from ..tdagent import TDAgent
import numpy as np
from scipy.stats.stats import pearsonr
from scipy.optimize import minimize
import logging

class CORN(TDAgent):
    """
    Correlation-driven nonparametric learning approach. Similar to anticor but instead
    of distance of return vectors they use correlation.
    In appendix of the article, universal property is proven.

    Two versions are available. Fast which provides around 2x speedup, but uses more memory
    (linear in window) and slow version which is memory efficient. Most efficient would
    be to rewrite it in sweave or numba.

    Reference:
        B. Li, S. C. H. Hoi, and V. Gopalkrishnan.
        Corn: correlation-driven nonparametric learning approach for portfolio selection, 2011.
        http://www.cais.ntu.edu.sg/~chhoi/paper_pdf/TIST-CORN.pdf
    """
    def __init__(self, w=5, rho=0.1):
        """
        :param w: Window parameter.
        :param rho: Correlation coefficient threshold. Recommended is 0.
        """
        # input check
        if not(-1 <= rho <= 1):
            raise ValueError('rho must be between -1 and 1')
        if not(w >= 2):
            raise ValueError('window must be greater than 2')
        super(CORN, self).__init__()
        self.w = w
        self.rho = rho


    def decide_by_history(self, x, last_b):
        self.record_history(x)
        x = self.get_last_rpv(x)

        T, N = self.history.shape
        m = 0
        histdata = np.zeros((T,N))

        if T <= self.w+1:
            '''use uniform portfolio weight vector'''
            return np.ones(x.size) / x.size

        if self.w==0:
            histdata = self.history
            m = T
        else:
            for i in np.arange(self.w+1, T+1):
                d1 = self.history[i-self.w:i-1,:]
                d2 = self.history[T-self.w+1:T,:]

                datacorr = np.corrcoef(d1,d2)[0,1]

                if datacorr >= self.rho:
                    m += 1
                    histdata[m,:] = self.history[i-1,:] #minus one to avoid out of bounds issue

        if m==0:
            return np.ones(x.size) / x.size

        #sqp according to OLPS implementation
        x_0 = np.ones(x.size) / x.size
        objective = lambda b: -np.prod(np.dot(histdata, b))
        cons = ({'type': 'eq', 'fun': lambda b: 1-np.sum(b, axis=0)},)
        bnds = [(0.,1)]*x.size
        while True:
            res = minimize(objective, x_0, bounds=bnds, constraints=cons, method='slsqp')
            eps = 1e-7
            if (res.x < 0-eps).any() or (res.x > 1+eps).any():
                x = x + np.random.randn(1)[0] * 1e-5
                logging.debug('Optimal portfolio weight vector not found, trying again...')
                continue
            elif res.success:
                break
            else:
                if np.isnan(res.x).any():
                    logging.warning('Solution does not exist, use uniform pwv')
                    res.x = np.ones(x.size) / x.size
                else:
                    logging.warning('Converged but not successfully.')
                break

        return res.x
