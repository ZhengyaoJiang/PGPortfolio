from ..tdagent import TDAgent
import numpy as np


class CRP(TDAgent):
    """ Constant rebalanced portfolio = use fixed weights all the time. Uniform weights are commonly used as a benchmark.

    Reference:
        T. Cover. Universal Portfolios, 1991.
        http://www-isl.stanford.edu/~cover/papers/paper93.pdf
    """

    def __init__(self, b=None):
        """
        :params b: Constant rebalanced portfolio weights. Default is uniform.
        """
        super(CRP, self).__init__()
        self.b = b

    def decide_by_history(self, x, last_b):
        x = self.get_last_rpv(x)

        # init b to default if necessary
        if self.b is None:
            self.b = np.ones(len(x)) / len(x)
        return self.b

