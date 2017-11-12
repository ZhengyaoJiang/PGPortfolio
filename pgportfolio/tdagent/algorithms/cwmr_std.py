from ..tdagent import TDAgent
import numpy as np
import scipy.stats
from numpy.linalg import inv
from numpy import diag, sqrt, log, trace


class CWMR_STD(TDAgent):
    """ Confidence weighted mean reversion.

    Reference:
        B. Li, S. C. H. Hoi, P.L. Zhao, and V. Gopalkrishnan.
        Confidence weighted mean reversion strategy for online portfolio selection, 2013.
        http://jmlr.org/proceedings/papers/v15/li11b/li11b.pdf
    """
    def __init__(self, eps=-0.5, confidence=0.95, sigma=None):
        """
        :param eps: Mean reversion threshold (expected return on current day must be lower
                    than this threshold). Recommended value is -0.5.
        :param confidence: Confidence parameter for profitable mean reversion portfolio. Recommended value is 0.95.
        """
        super(CWMR_STD, self).__init__()
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

    def update(self, x, x_upper, mu, sigma, M, V, theta, eps):
        # lambda from equation 7
        foo = (V - x_upper * x.T * np.sum(sigma, axis=1)) / M**2 + V * theta**2 / 2.
        a = foo**2 - V**2 * theta**4 / 4
        b = 2 * (eps - np.log(M)) * foo
        c = (eps - np.log(M))**2 - V * theta**2

        a,b,c = a[0,0], b[0,0], c[0,0]

        lam = np.amax([0,
                  (-b + sqrt(b**2 - 4 * a * c)) / (2. * a),
                  (-b - sqrt(b**2 - 4 * a * c)) / (2. * a)])
        # bound it due to numerical problems
        lam = np.minimum(lam, 1E+7)

        # update mu and sigma
        U_sqroot = 0.5 * (-lam * theta * V + sqrt(lam**2 * theta**2 * V**2 + 4*V))
        mu = mu - lam * sigma * (x - x_upper) / M
        sigma = inv(inv(sigma) + theta * lam / U_sqroot * diag(x)**2)
        """
        tmp_sigma = inv(inv(sigma) + theta*lam/U_sqroot*diag(xt)^2);
        % Don't update sigma if results are badly scaled.
        if all(~isnan(tmp_sigma(:)) & ~isinf(tmp_sigma(:)))
            sigma = tmp_sigma;
        end
        """
        return mu, sigma

