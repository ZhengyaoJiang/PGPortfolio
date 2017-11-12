from __future__ import division,absolute_import,print_function
from pgportfolio.trade.backtest import BackTest
from pgportfolio.tdagent.algorithms import crp, ons, olmar, up, anticor1, pamr,\
    best, bk, cwmr_std, eg, sp, ubah, wmamr, bcrp, cornk, m0, rmr

# the dictionary of name of algorithms mapping to the constructor of tdagents
ALGOS = {"crp": crp.CRP, "ons": ons.ONS, "olmar": olmar.OLMAR, "up": up.UP,
         "anticor": anticor1.ANTICOR1, "pamr": pamr.PAMR,
         "best": best.BEST, "bk": bk.BK, "bcrp": bcrp.BCRP,
         "corn": cornk.CORNK, "m0": m0.M0, "rmr": rmr.RMR,
         "cwmr": cwmr_std.CWMR_STD, "eg": eg.EG, "sp": sp.SP, "ubah": ubah.UBAH,
         "wmamr": wmamr.WMAMR}

def execute_backtest(algo, config):
    """
    @:param algo: string representing the name the name of algorithms
    @:return: numpy array of portfolio changes
    """
    agent, agent_type, net_dir = _construct_agent(algo)
    backtester = BackTest(config, agent=agent, agent_type=agent_type, net_dir=net_dir)
    backtester.start_trading()
    return backtester.test_pc_vector


def _construct_agent(algo):
    if algo.isdigit():
        agent = None
        agent_type = "nn"
        net_dir = "./train_package/" + algo + "/netfile"
    elif algo in ALGOS:
        agent = ALGOS[algo]()
        agent_type = "traditional"
        net_dir = None
    else:
        message = "The algorithm name "+algo+" is not support. Supported algos " \
                                             "are " + str(list(ALGOS.keys()))
        raise LookupError(message)
    return agent, agent_type, net_dir