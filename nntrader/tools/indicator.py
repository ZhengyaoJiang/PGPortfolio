from __future__ import division, print_function, absolute_import
import numpy as np


def max_drawdown(pc_array):
    """calculate the max drawdown with the portfolio changes
    @:param pc_array: all the portfolio changes during a trading process
    @:return: max drawdown
    """
    portfolio_values = []
    drawdown_list = []
    max_benefit = 0
    for i in range(pc_array.shape[0]):
        if i > 0:
            portfolio_values.append(portfolio_values[i - 1] * pc_array[i])
        else:
            portfolio_values.append(pc_array[i])
        if portfolio_values[i] > max_benefit:
            max_benefit = portfolio_values[i]
            drawdown_list.append(0.0)
        else:
            drawdown_list.append(1.0 - portfolio_values[i] / max_benefit)
    return max(drawdown_list)


def sharpe(pc_array):
    """calculate sharpe ratio with the portfolio changes
    @:param pc_array: all the portfolio changes during a trading process
    @:return: sharpe ratio
    """
    pc_array = pc_array-1.0
    return np.mean(pc_array)/np.std(pc_array)


def moving_accumulate(pc_array, n=48):
    acc = np.cumprod(pc_array)
    acc[n:] = acc[n:] / acc[:-n]
    return acc


def positive_count(pc_array):
    return np.sum(pc_array>1)


def negative_count(pc_array):
    return np.sum(pc_array<1)
