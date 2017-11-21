from __future__ import absolute_import
import numpy as np
import logging
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, euclidean

class TDAgent(object):
    '''Traditional Agent.
    parent class for algorithms(new-style)
    '''

    def __init__(self, history=None, cum_ret=None, last_b=None):
        '''init
        :param X: input
        :param history: a history list of relative price vector
        '''
        self.history = history
        self.cum_ret = cum_ret
        self.last_b = last_b

    @property
    def agent(self):
        return self._agent


    def decide_by_history(self, x, last_b):
        '''calculate new portfolio weight vector.
        :param x: input x
        :last_b: last portfolio weight vector
        '''
        raise NotImplementedError('subclass must implement this!')

    def get_last_rpv(self, x):
        '''remove dimension of input. Return last relative price vector.
        :param x: matrix with shape (1, window_size, coin_number+1)
        '''
        if x.ndim == 3:
            #print x.shape
            last_rpv = x[0,:,-1] # output a vector with shape (x.size,)
        else:
            last_rpv = x #if it has already been processed just return x
        return last_rpv

    def get_first_history(self, x):
        '''get history in first period
        :param x: input matrix
        '''
        if x.ndim == 3:
            first = x[0,:,:] # array size (#assets, #periods)

        #return (#periods, #assets) for convention
        return first.T

    def record_history(self, x):
        nx = self.get_last_rpv(x)
        nx = np.reshape(nx, (1,nx.size))
        if self.history is None:
            #self.history = self.get_first_history(x)
            self.history = nx
        else:
            self.history = np.vstack((self.history, nx))

    def get_close(self):
        '''get close data from relative price
        :param x: relative price data
        '''
        close = np.ones(self.history.shape)
        for i in range(1,self.history.shape[0]):
            close[i,:] = close[i-1] * self.history[i,:]
        return close

    def simplex_proj(self, y):
        '''projection of y onto simplex. '''
        m = len(y)
        bget = False

        s = sorted(y, reverse = True)
        tmpsum = 0.

        for ii in range(m-1):
            tmpsum = tmpsum + s[ii]
            tmax = (tmpsum - 1) / (ii + 1)
            if tmax >= s[ii+1]:
                bget = True
                break

        if not bget:
            tmax = (tmpsum + s[m-1] - 1) / m

        return np.maximum(0, y-tmax)

    def get_last_return(self, last_b):
        '''Caulate daily retrun. No need to calculate transaction cost there.
        '''
        last_x = self.history[-1,:]
        self.ret = last_b * last_x #element-wise
        return np.squeeze(self.ret)

    def cal_cum_ret(self, ret):
        '''Calculate the cumulative return.
        :param ret: newest retrun
        '''
        if self.cum_ret is None:
            self.cum_ret = ret
        else:
            self.cum_ret = self.cum_ret * ret #element-wise
        return self.cum_ret

    def find_bcrp(self, X, max_leverage=1):
        x_0 = max_leverage * np.ones(X.shape[1]) / np.float(X.shape[1])
        objective = lambda b: -np.prod(np.dot(X, b))
        cons = ({'type': 'eq', 'fun': lambda b: max_leverage - np.sum(b, axis=0)},)
        bnds = [(0., max_leverage)]*len(x_0)
        while True:
            res = minimize(objective, x_0, bounds=bnds, constraints=cons, method='slsqp')
            eps = 1e-7
            if (res.x < 0-eps).any() or (res.x > max_leverage+eps).any():
                X = X + np.random.randn(1)[0] * 1e-5
                logging.debug('Optimal weights not found, trying again...')
                continue
            elif res.success:
                break
            else:
                if np.isnan(res.x).any():
                    logging.warning('Solution does not exist, use uniform portfolio weight vector.')
                    res.x = np.ones(X.shape[1]) / X.shape[1]
                else:
                    logging.warning('Converged but not successfully.')
                break

        return res.x


    def euclidean_proj_simplex(self, v, s=1):
        '''Compute the Euclidean projection on a positive simplex
        :param v: n-dimensional vector to project
        :param s: int, radius of the simple

        return w numpy array, Euclidean projection of v on the simplex

        Original author: John Duchi
        '''
        assert s>0, "Radius s must be positive (%d <= 0)" % s

        n, = v.shape # raise ValueError if v is not 1D
        # check if already on the simplex
        if v.sum() == s and np.alltrue( v>= 0):
            return v

        # get the array of cumulaive sums of a sorted copy of v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        # get the number of >0 components of the optimal solution
        rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
        # compute the Lagrange multiplier associated to the simplex constraint
        theta = (cssv[rho] - s) / (rho + 1.)
        w = (v-theta).clip(min=0)
        return w

    def l1_median_VaZh(self, X, eps=1e-5):
        '''calculate the L1_median of X with the l1median_VaZh method
        '''
        y = np.mean(X, 0)

        while True:
            D = cdist(X, [y])
            nonzeros = (D != 0)[:, 0]

            Dinv = 1 / D[nonzeros]
            Dinvs = np.sum(Dinv)
            W = Dinv / Dinvs
            T = np.sum(W * X[nonzeros], 0)
            num_zeros = len(X) - np.sum(nonzeros)
            if num_zeros == 0:
                y1 = T
            elif num_zeros == len(X):
                return y
            else:
                R = (T - y) * Dinvs
                r = np.linalg.norm(R)
                rinv = 0 if r==0 else num_zeros/r
                y1 = max(0, 1-rinv)*T + min(1, rinv)*y

            if euclidean(y, y1) < eps:
                return y1

            y = y1

    def corn_expert(self, data, w, c):
        '''
        :param w: window sze
        :param c: correlation coefficient threshold
        '''
        T, N = data.shape
        m = 0
        histdata = np.zeros((T,N))

        if T <= w+1:
            '''use uniform portfolio weight vector'''
            return np.ones(N) / N

        if w==0:
            histdata = data[:T,:]
            m = T
        else:
            for i in np.arange(w, T):
                d1 = data[i-w:i,:]
                d2 = data[T-w:T,:]
                datacorr = np.corrcoef(d1,d2)[0,1]

                if datacorr >= c:
                    m += 1
                    histdata[m,:] = data[i-1,:] #minus one to avoid out of bounds issue

        if m==0:
            return np.ones(N) / N

        #sqp according to OLPS implementation
        x_0 = np.ones((1,N)) / N
        objective = lambda b: -np.prod(np.dot(histdata, b))
        cons = ({'type': 'eq', 'fun': lambda b: 1-np.sum(b, axis=0)},)
        bnds = [(0.,1)]*N
        while True:
            res = minimize(objective, x_0, bounds=bnds, constraints=cons, method='slsqp')
            eps = 1e-7
            if (res.x < 0-eps).any() or (res.x > 1+eps).any():
                data += np.random.randn(1)[0] * 1e-5
                logging.debug('Optimal portfolio weight vector not found, trying again...')
                continue
            elif res.success:
                break
            else:
                if np.isnan(res.x).any():
                    logging.warning('Solution does not exist, use uniform pwv')
                    res.x = np.ones(N) / N
                else:
                    logging.warning('Converged but not successfully.')
                break

        return res.x
