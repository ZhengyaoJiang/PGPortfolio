from ..tdagent import TDAgent
import numpy as np

class UP(TDAgent):
    """ Universal Portfolio by Thomas Cover enhanced for "leverage" (instead of just
        taking weights from a simplex, leverage allows us to stretch simplex to
        contain negative positions).

    Reference:
        T. Cover. Universal Portfolios, 1991.
        http://www-isl.stanford.edu/~cover/papers/paper93.pdf
    """
    def __init__(self, eval_points=10000, leverage=1., W=None):
        """
        :param eval_points: Number of evaluated points (approximately). Complexity of the
            algorithm is O(time * eval_points * nr_assets**2) because of matrix multiplication.
        :param leverage: Maximum leverage used. leverage == 1 corresponds to simplex,
            leverage == 1/nr_stocks to uniform CRP. leverage > 1 allows negative weights
            in portfolio.
        """
        super(UP, self).__init__()
        self.eval_points = eval_points
        self.leverage = leverage
        self.W = W

    def init_portfolio(self, X):
        """ Create a mesh on simplex and keep wealth of all strategies. """
        m = X.shape[1]
        # create set of CRPs
        self.W = np.matrix(mc_simplex(m - 1, self.eval_points))
        self.S = np.matrix(np.ones(self.W.shape[0])).T

        # stretch simplex based on leverage (simple calculation yields this)
        leverage = max(self.leverage, 1./m)
        stretch = (leverage - 1./m) / (1. - 1./m)
        self.W = (self.W - 1./m) * stretch + 1./m


    def decide_by_history(self, x, last_b):
        # calculate new wealth of all CRPs
        x = self.get_last_rpv(x)
        x = np.reshape(x, (1,x.size))

        if self.W is None:
            self.init_portfolio(x)

        self.S = np.multiply(self.S, self.W * np.matrix(x).T)
        b = self.W.T * self.S
        pv = b / np.sum(b)
        pvn = np.ravel(pv)
        return pvn #squeeze not working there



def mc_simplex(d, points):
    '''Sample random points from a simplex with dimension d
    :param d: Number of dimensions
    :param points: Total number of points.
    '''
    a = np.sort(np.random.random((points,d)))
    a = np.hstack([np.zeros((points,1)), a, np.ones((points,1))])
    return np.diff(a)



