from ..tdagent import TDAgent
import numpy as np


class PAMR(TDAgent):
    """ Passive aggressive mean reversion strategy for portfolio selection.
    There are three variants with different parameters, see original article
    for details.

    Reference:
        B. Li, P. Zhao, S. C.H. Hoi, and V. Gopalkrishnan.
        Pamr: Passive aggressive mean reversion strategy for portfolio selection, 2012.
        http://www.cais.ntu.edu.sg/~chhoi/paper_pdf/PAMR_ML_final.pdf
    """
    def __init__(self, eps=0.5, C=500, variant=2, b=None):
        """
        :param eps: Control parameter for variant 0. Must be >=0, recommended value is
                    between 0.5 and 1.
        :param C: Control parameter for variant 1 and 2. Recommended value is 500.
        :param variant: Variants 0, 1, 2 are available.
        """
        super(PAMR, self).__init__()

        # input check
        if not(eps >= 0):
            raise ValueError('epsilon parameter must be >=0')

        if variant == 0:
            if eps is None:
                raise ValueError('eps parameter is required for variant 0')
        elif variant == 1 or variant == 2:
            if C is None:
                raise ValueError('C parameter is required for variant 1,2')
        else:
            raise ValueError('variant is a number from 0,1,2')

        self.eps = eps
        self.C = C
        self.variant = variant
        self.b = b

    def decide_by_history(self, x, last_b):
        x = self.get_last_rpv(x)
        # calculate return prediction
        if self.b is None:
            self.b = np.ones(x.size) / x.size
        last_b = self.b
        b = self.update(last_b, x, self.eps, self.C)
        b = b.ravel()
        self.b = b
        return self.b


    def update(self, b, x, eps, C):
        """ Update portfolio weights to satisfy constraint b * x <= eps
        and minimize distance to previous weights. """
        x_mean = np.mean(x)

        le = np.maximum(0., np.dot(b, x) - eps)

        denominator = np.square(np.linalg.norm(x-x_mean))

        if self.variant == 0:
            tau = le / denominator
        elif self.variant == 1:
            tau = np.minimum(C, le / denominator)
        elif self.variant == 2:
            tau = le / (denominator + 0.5 / C)

        # limit lambda to avoid numerical problems
        tau = np.minimum(100000, tau)

        # update portfolio
        b = b - tau * (x - x_mean)

        # project it onto simplex
        return self.simplex_proj(b)


