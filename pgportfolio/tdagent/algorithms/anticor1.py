from ..tdagent import TDAgent
import numpy as np
import logging

class ANTICOR1(TDAgent):
    '''
    anti-correlation olps
    '''
    def __init__(self, window=30, exp_w=None):
        super(ANTICOR1, self).__init__()
        self.window = window
        self.exp_ret = np.ones((window-1,1))
        self.exp_w = exp_w

    def decide_by_history(self, x, last_b):
        self.record_history(x)
        n, m = self.history.shape
        if self.exp_w is None:
            self.exp_w = np.ones((self.window-1,m)) / m

        for k in np.arange(1,self.window):
            self.exp_w[k-1,:] = self.update(self.history, self.exp_w[k-1,:], k+1)
        numerator = 0
        denominator = 0

        for k in np.arange(1,self.window):
            numerator += self.exp_ret[k-1] * self.exp_w[k-1,:]
            denominator += self.exp_ret[k-1]

        weight = numerator.T / denominator

        for k in np.arange(1, self.window):
            self.exp_ret[k-1] = np.dot(self.exp_ret[k-1]*self.history[-1,:], self.exp_w[k-1,:].T)

        self.exp_ret[:,0] /= np.sum(self.exp_ret[:,0])

        return weight

    def update(self, data,last_b, w):
        T, N = data.shape
        b = last_b

        if T >= 2*w :
            data1 = data[T-2*w:T-w,:]
            data2 = data[T-w:T,:]
            #print data1
            LX1 = np.log(data1)
            LX2 = np.log(data2)

            mu1 = np.mean(LX1, axis=0)
            mu2 = np.mean(LX2, axis=0)

            n_LX1 = LX1 - mu1
            n_LX2 = LX2 - mu2


            sig1 = np.diag(np.dot(n_LX1.T, n_LX1).T) / (w-1)
            sig2 = np.diag(np.dot(n_LX2.T, n_LX2).T) / (w-1)

            sig1 = sig1[:,None]
            sig2 = sig2[:,None]

            sigma = np.dot(sig1,sig2.T) #(N,N)

            mCov = n_LX1.T.dot(n_LX2) / (w-1)
            mCorr = np.zeros((N,N))

            mCorr = np.zeros((N,N))
            new_sigma = np.multiply(sigma, sigma!=0)
            new_sigma_zero_index = new_sigma==0
            new_sigma[new_sigma_zero_index] = 1e-8
            mCorr = np.multiply(mCov, sigma!=0) / np.sqrt(new_sigma)

            claim = np.zeros((N,N))
            w_mu2 = np.tile(mu2[None,...].T, (1,N))
            w_mu1 = np.tile(mu2[None,...], (N,1))

            s12 = np.multiply(w_mu2 >= w_mu1, mCorr>0)
            claim = np.multiply(claim, s12) + np.multiply(mCorr, s12)

            diag_mCorr = np.diag(mCorr)
            cor1 = np.maximum(0, np.tile(-diag_mCorr[...,None], (1,N)))
            cor2 = np.maximum(0, np.tile(-diag_mCorr[...,None].T, (N,1)))
            claim +=  np.multiply(cor1, s12) + np.multiply(cor2, s12)
            claim = np.multiply(claim, s12)

            transfer = np.zeros((N,N))
            s_claim = np.sum(claim, axis=1)
            sum_claim = np.tile(s_claim[...,None],(1,N))


            s1 = np.absolute(sum_claim) > 0

            w_b = np.tile(b[...,None], (1,N))
            mul_bc = np.multiply(w_b, s1) * np.multiply(claim, s1)
            deno = np.multiply(sum_claim, s1)
            deno_zero_index = deno==0
            deno[deno_zero_index] = 1e-8
            transfer = np.divide(mul_bc, deno)
            transfer = np.where(np.isnan(transfer), 0, transfer)

            transfer_ij = transfer.T - transfer
            sum_ij = np.sum(transfer_ij, axis=0)

            b = np.subtract(b, sum_ij.T)

        return b
