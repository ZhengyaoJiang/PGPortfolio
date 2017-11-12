from ..tdagent import TDAgent
import numpy as np
import logging
from scipy.optimize import minimize

class BNN(TDAgent):
    '''
    Non-parametric
    '''
    def __init__(self, K=5, L=10, exp_w=None):
        super(BNN, self).__init__()
        self.K = K
        self.L = L
        self.exp_ret = np.ones((K,L+1))
        self.exp_w = exp_w

    def get_b(self, x, last_b):
        self.record_history(x)

        data = self.history
        n, m = data.shape

        if self.exp_w is None:
            self.exp_w = np.ones((self.K*(self.L+1),m)) / m

        self.exp_w[self.K*self.L,:] = self.update(data, 0, 0)

        for k in np.arange(self.K):
            for l in np.arange(self.L):
                pl = 0.02+0.5*(l-1)/(self.L-1)
                self.exp_w[(k-1)*self.L+l,:] = self.update(data, k, pl)

        p = 1./(self.K*self.L)
        numerator = p * self.exp_ret[0,self.L] * self.exp_w[self.K*self.L,:]
        denominator = p * self.exp_ret[0, self.L]

        for k in np.arange(self.K):
            for l in np.arange(self.L):
                numerator += p*self.exp_ret[k, l] * self.exp_w[(k-1)*self.L+l,:]
                denominator += p*self.exp_ret[k,l]

        weight = numerator.T / denominator

        for k in np.arange(self.K):
            for l in np.arange(self.L):
                self.exp_ret[k,l] *= np.dot(self.history[-1,:], self.exp_w[(k-1)*self.L+l-1,:])

        return weight

    def update(self, data, k, l):
        T, N = data.shape
        m = 0
        histdata = np.zeros((T,N))

        if T <= k+1:
            return np.ones((1,N)) / N

        if k==0 and l==0:
            histdata = data[:T,:]
            m = T
        else:
            normid = np.zeros((T-k,1))
            histdata = data[:T,:]
            normid[:k] = 0
            for i in np.arange(k+1,T):
                data2 = data[i-k:i-1,:] - data[T-k+1:T,:]
                normid[:i] = np.sqrt(np.trace(data2.dot(data2.T)))
                sortpos = np.sort(normid)
                sortpos = sortpos.astype(int)
                m = int(np.floor(l*T))
                for j in np.arange(m):
                    histdata = np.vstack((histdata,histdata[int(sortpos[j]),:]))
        if m == 0:
            return np.ones((1,N)) / N

        b = opt_weights(histdata)
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
