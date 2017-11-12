from __future__ import division,absolute_import,print_function
import numpy as np
import pandas as pd


def pricenorm3d(m, features, norm_method, fake_ratio=1.0, with_y=True):
    """normalize the price tensor, whose shape is [features, coins, windowsize]
    @:param m: input tensor, unnormalized and there could be nan in it
    @:param with_y: if the tensor include y (future price)
        logging.debug("price are %s" % (self._latest_price_matrix[0, :, -1]))
    """
    result = m.copy()
    if features[0] != "close":
        raise ValueError("first feature must be close")
    for i, feature in enumerate(features):
        if with_y:
            one_position = 2
        else:
            one_position = 1
        pricenorm2d(result[i], m[0, :, -one_position], norm_method=norm_method,
                    fake_ratio=fake_ratio, one_position=one_position)
    return result


# input m is a 2d matrix, (coinnumber+1) * windowsize
def pricenorm2d(m, reference_column,
                norm_method="absolute", fake_ratio=1.0, one_position=2):
    if norm_method=="absolute":
        output = np.zeros(m.shape)
        for row_number, row in enumerate(m):
            if np.isnan(row[-one_position]) or np.isnan(reference_column[row_number]):
                row[-one_position] = 1.0
                for index in range(row.shape[0] - one_position + 1):
                    if index > 0:
                        row[-one_position - index] = row[-index - one_position + 1] / fake_ratio
                row[-one_position] = 1.0
                row[-1] = fake_ratio
            else:
                row = row / reference_column[row_number]
                for index in range(row.shape[0] - one_position + 1):
                    if index > 0 and np.isnan(row[-one_position - index]):
                        row[-one_position - index] = row[-index - one_position + 1] / fake_ratio
                if np.isnan(row[-1]):
                    row[-1] = fake_ratio
            output[row_number] = row
        m[:] = output[:]
    elif norm_method=="relative":
        output = m[:, 1:]
        divisor = m[:, :-1]
        output = output / divisor
        pad = np.empty((m.shape[0], 1,))
        pad.fill(np.nan)
        m[:] = np.concatenate((pad, output), axis=1)
        m[np.isnan(m)] = fake_ratio
    else:
        raise ValueError("there is no norm morthod called %s" % norm_method)


def get_chart_until_success(polo, pair, start, period, end):
    is_connect_success = False
    chart = {}
    while not is_connect_success:
        try:
            chart = polo.marketChart(pair=pair, start=int(start), period=int(period), end=int(end))
            is_connect_success = True
        except Exception as e:
            print(e)
    return chart


def get_type_list(feature_number):
    """
    :param feature_number: an int indicates the number of features
    :return: a list of features n
    """
    if feature_number == 1:
        type_list = ["close"]
    elif feature_number == 2:
        type_list = ["close", "volume"]
        raise NotImplementedError("the feature volume is not supported currently")
    elif feature_number == 3:
        type_list = ["close", "high", "low"]
    elif feature_number == 4:
        type_list = ["close", "high", "low", "open"]
    else:
        raise ValueError("feature number could not be %s" % feature_number)
    return type_list


def panel2array(panel):
    """convert the panel to datatensor (numpy array) without btc
    """
    without_btc = np.transpose(panel.values, axes=(2, 0, 1))
    return without_btc


def count_periods(start, end, period_length):
    """
    :param start: unix time, excluded
    :param end: unix time, included
    :param period_length: length of the period
    :return: 
    """
    return (int(end)-int(start)) // period_length


def get_volume_forward(time_span, portion, portion_reversed):
    volume_forward = 0
    if not portion_reversed:
        volume_forward = time_span*portion
    return volume_forward


def panel_fillna(panel, type="bfill"):
    """
    fill nan along the 3rd axis
    :param panel: the panel to be filled
    :param type: bfill or ffill
    """
    frames = {}
    for item in panel.items:
        if type == "both":
            frames[item] = panel.loc[item].fillna(axis=1, method="bfill").\
                fillna(axis=1, method="ffill")
        else:
            frames[item] = panel.loc[item].fillna(axis=1, method=type)
    return pd.Panel(frames)

