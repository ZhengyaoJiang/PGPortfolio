from ..tdagent import TDAgent
from pgportfolio.tdagent.algorithms.crp import CRP
import numpy as np
from scipy.optimize import minimize

class BCRP(CRP):
    """ Best Constant Rebalanced Portfolio = Constant Rebalanced Portfolio constructed with hindsight. It is often used as benchmark.

    Reference:
        T. Cover. Universal Portfolios, 1991.
        http://www-isl.stanford.edu/~cover/papers/paper93.pdf
    """

    def __init__(self, last_b=None):
        super(BCRP, self).__init__()
        self.last_b = last_b

    def get_weight(self, data):
        """ Find weights which maximize return on X in hindsight! """
        weights = opt_weights(data)
        return weights

    def decide_by_history(self, x, last_b):
        if self.last_b is None:
            from pgportfolio.tools.trade import get_test_data
            from pgportfolio.tools.configprocess import preprocess_config
            import json
            with open("pgportfolio/net_config.json") as file:
                config = json.load(file)
            config = preprocess_config(config)
            data = get_test_data(config)
            self.last_b = self.get_weight(data.T)

        return self.last_b


def opt_weights(X, max_leverage=1):
    x_0 = max_leverage * np.ones(X.shape[1]) / float(X.shape[1])
    objective = lambda b: -np.prod(X.dot(b))
    cons = ({'type': 'eq', 'fun': lambda b: max_leverage-np.sum(b)},)
    bnds = [(0., max_leverage)]*len(x_0)
    res = minimize(objective, x_0, bounds=bnds, constraints=cons, method='slsqp', options={'ftol': 1e-07})
    return res.x


if __name__ == '__main__':
    from pgportfolio.tools.backtest import get_test_data
    from pgportfolio.tools.configprocess import preprocess_config
    import json
    with open("pgportfolio/net_config.json") as file:
        config = json.load(file)
    config = preprocess_config(config)
    data = get_test_data(config)
    bcrp = BCRP()
    result = bcrp.get_weight(data.T)

