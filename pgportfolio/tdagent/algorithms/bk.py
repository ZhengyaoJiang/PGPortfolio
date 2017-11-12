from ..tdagent import TDAgent
import numpy as np
import logging
from scipy.optimize import minimize

class BK(TDAgent):
    '''
    anti-correlation olps
    '''
    def __init__(self, K=5, L=10, c=1, exp_w=None):
        super(BK, self).__init__()
        self.K = K
        self.L = L
        self.c = c
        self.exp_ret = np.ones((K,L+1))
        self.exp_w = exp_w

    def decide_by_history(self, x, last_b):
        self.record_history(x)

        data = self.history

        n, m = data.shape

        if self.exp_w is None:
            self.exp_w = np.ones((self.K*(self.L+1),m)) / m

        self.exp_w[self.K*self.L,:] = self.update(data, 0, 0, self.c)

        for k in np.arange(self.K):
            for l in np.arange(self.L):
                self.exp_w[(k-1)*self.L+l,:] = self.update(data, k, l, self.c)

        p = 1./(self.K*self.L)
        numerator = p * self.exp_ret[0,self.L] * self.exp_w[self.K*self.L,:]
        denominator = p * self.exp_ret[0, self.L]

        for k in np.arange(self.K):
            for l in np.arange(self.L):
                numerator += p*self.exp_ret[k, l] * self.exp_w[(k-1)*self.L+l,:]
                denominator += p*self.exp_ret[k,l]

        weight = numerator.T / denominator

        self.exp_ret[0, self.L] *= np.dot(self.history[-1,:], self.exp_w[self.K*self.L,:].T)

        for k in np.arange(self.K):
            for l in np.arange(self.L):
                self.exp_ret[k,l] *= np.dot(self.history[-1,:], self.exp_w[(k-1)*self.L+l,:])


        return weight

    def update(self, data, k, l, c):
        '''
        :param w: window sze
        :param c: correlation coefficient threshold
        '''
        T, N = data.shape
        m = -1
        histdata = np.zeros((T,N))

        if T <= k+1:
            return np.ones(N) / N

        if k==0 and l==0:
            histdata = data[:T,:]
            m = T
        else:
            for i in np.arange(k+1, T):
                #print 'i is %d k is %d T is %d\n' % (i,k,T)
                data2 = data[i-k-1:i,:] - data[T-k-1:T,:]
                #print data2

                if np.sqrt(np.trace(np.dot(data2,data2.T))) <= c/l:
                    m += 1
                    histdata[m,:] = data[i,:] #minus one to avoid out of bounds issue

        if m==-1:
            return np.ones(N) / N

        b = opt_weights(histdata[:m+1,:])
        #print b
        #print 'w is %d\t T is %d\n' % (w,T)
        return b

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
