from ..tdagent import TDAgent
import numpy as np
import logging
from scipy.optimize import minimize

class CORNK(TDAgent):
    '''
    Correlation driven non parametric Uniform
    '''
    def __init__(self, K=5, L=10, pc=0.1, exp_w=None):
        '''
        :param K: maximum window size
        :param L: splits into L parts, in each K
        '''
        super(CORNK, self).__init__()
        self.K = K
        self.L = L
        self.pc = pc
        self.exp_ret = np.ones((K,L))
        self.exp_w = exp_w


    def decide_by_history(self, X, last_b):
        self.record_history(X)

        n, m = self.history.shape

        if self.exp_w is None:
            self.exp_w = np.ones((self.K*self.L, m)) / m

        for k in np.arange(self.K):
            for l in np.arange(self.L):
                rho = l / self.L
                self.exp_w[(k-1)*self.L+l,:] = self.update(self.history, k+1, rho)

        nc = np.ceil(self.pc*self.K*self.L)
        exp_ret_vec = self.exp_ret.ravel()
        exp_ret_sort = np.sort(exp_ret_vec, kind='heapsort')
        ret_rho = exp_ret_sort[int(self.K*self.L-nc+1)]

        numerator = 0
        denominator = 0

        p = 1./(self.K*self.L)

        for k in np.arange(self.K):
            for l in np.arange(self.L):
                p = 1 if self.exp_ret[k,l] >= ret_rho else 0
                numerator += p * self.exp_ret[k,l] * self.exp_w[(k-1)*self.L+l,:]
                denominator += p * self.exp_ret[k,l]

        b = np.divide(numerator.T , denominator)

        for k in range(self.K):
            for l in range(self.L):
                self.exp_ret[k,l] *= np.dot(self.history[-1,:], self.exp_w[(k-1)*self.L+l,:].T)

        return b

    def update(self, data, w, c):
        '''
        :param w: window sze
        :param c: correlation coefficient threshold
        '''
        T, N = data.shape
        m = -1
        histdata = np.zeros((T,N))

        if T <= w+1:
            return np.ones(N) / N

        if w==0:
            histdata = data[:T,:]
            m = T
        else:
            for i in np.arange(w, T):
                d1 = data[i-w:i,:].ravel()
                d2 = data[T-w:T,:].ravel()

                datacorr = np.corrcoef(d1,d2)[1,0]


                if datacorr >= c:
                    m += 1
                    histdata[m,:] = data[i,:] #minus one to avoid out of bounds issue

        if m==-1:
            return np.ones(N) / N

        b = opt(histdata[:m+1,:])
        return b

def opt(X):
    x_0 = np.ones(X.shape[1]) / X.shape[1]
    objective = lambda b: -np.prod(X.dot(b))
    cons = ({'type': 'eq', 'fun': lambda b: 1-np.sum(b)},)
    bnds = [(0,1)]*len(x_0)
    res = minimize(objective, x0=x_0,  bounds=bnds, constraints=cons, method='slsqp', )
    return res.x
