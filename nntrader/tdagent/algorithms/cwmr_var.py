from ..tdagent import TDAgent
import numpy as np
import scipy.stats
from numpy.linalg import inv
from numpy import diag, sqrt, log, trace

class CWMR_VAR(TDAgent):
    """ First variant of a CWMR outlined in original article. It is
    only approximation to the posted problem. """
    def __init__(self, eps=-0.5, confidence=0.95, sigma=None):
        """
        :param eps: Mean reversion threshold (expected return on current day must be lower
                    than this threshold). Recommended value is -0.5.
        :param confidence: Confidence parameter for profitable mean reversion portfolio. Recommended value is 0.95.
        """
        super(CWMR_VAR, self).__init__()
        # input check
        if not (0 <= confidence <= 1):
            raise ValueError('confidence must be from interval [0,1]')

        self.eps = eps
        self.theta = scipy.stats.norm.ppf(confidence)
        self.sigma = sigma

    def init_portfolio(self, X):
        m = X.shape[1]
        self.sigma = np.matrix(np.eye(m) / m**2)


    def decide_by_history(self, x, last_b):
        x = self.get_last_rpv(x)
        x = np.reshape(x, (1,x.size))
        last_b = np.reshape(last_b, (1,last_b.size))
        if self.sigma is None:
            self.init_portfolio(x)
        # initialize
        m = len(x)
        mu = np.matrix(last_b).T
        sigma = self.sigma
        theta = self.theta
        eps = self.eps
        x = np.matrix(x).T    # matrices are easier to manipulate

        # 4. Calculate the following variables
        M = (mu.T * x).mean()
        V = x.T * sigma * x
        x_upper = sum(diag(sigma) * x) / trace(sigma)

        # 5. Update the portfolio distribution
        mu, sigma = self.update(x, x_upper, mu, sigma, M, V, theta, eps)

        # 6. Normalize mu and sigma
        mu = self.simplex_proj(mu)
        sigma = sigma / (m**2 * trace(sigma))
        """
        sigma(sigma < 1e-4*eye(m)) = 1e-4;
        """
        self.sigma = sigma

        return np.ravel(mu)

