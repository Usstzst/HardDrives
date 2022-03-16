#-*- coding: utf-8 -*-

import re
import math
import sys
import time
from operator import itemgetter
import copy

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt


def _num_dd(varseries):
    """
        :param varseries: 
        :type varseries: pd.Series

        :return: 返回一个字典, 字典的 key 有 
                ks-pvalue, max-or-bot1, mean-or-top1, median-or-bot5, min-or-top2, n-miss, n-unique, n-valid
                 'p1-or-top3', 'p25-or-top5', 'p5-or-top4', 'p75-or-bot4': 3.0, 'p95-or-bot3', 'p99-or-bot2'
        :rtype: dict
    """
    statsDict = {}
    statsDict['n-valid'] = varseries.count()
    statsDict['n-miss'] = len(varseries) - varseries.count()
    statsDict['n-unique'] = len(varseries.unique())

    if len(varseries.unique()) == 1 and str(varseries.unique()[0]) == 'nan':
        statsDict['mean-or-top1'] = '.'
        statsDict['min-or-top2'] = '.'
        statsDict['p1-or-top3'] = '.'
        statsDict['p5-or-top4'] = '.'
        statsDict['p25-or-top5'] = '.'
        statsDict['median-or-bot5'] = '.'
        statsDict['p75-or-bot4'] = '.'
        statsDict['p95-or-bot3'] = '.'
        statsDict['p99-or-bot2'] = '.'
        statsDict['max-or-bot1'] = '.'
        statsDict['ks-pvalue'] = '.'
    else:
        statsDict['mean-or-top1'] = varseries.mean()
        statsDict['min-or-top2'] = varseries.min()

        temp = varseries[varseries.notnull()]
        statsDict['p1-or-top3'] = temp.quantile(0.01)
        statsDict['p5-or-top4'] = temp.quantile(0.05)
        statsDict['p25-or-top5'] = temp.quantile(0.25)
        statsDict['median-or-bot5'] = temp.quantile(0.5)
        statsDict['p75-or-bot4'] = temp.quantile(0.75)
        statsDict['p95-or-bot3'] = temp.quantile(0.95)
        statsDict['p99-or-bot2'] = temp.quantile(0.99)
        statsDict['max-or-bot1'] = varseries.max()
        mu = varseries.mean()
        sigma = np.std(varseries)
        stat_val, p_value = stats.kstest(temp, 'norm', (mu, sigma))
        statsDict['ks-pvalue'] = str(p_value)
    return statsDict


def _char_dd(varseries):
    """
        :param varseries: 
        :type varseries: pd.Series
    """
    statsDict = {}
    varseriesFillna = varseries.fillna('__NULL__')
    freqStatsDf = pd.DataFrame(varseriesFillna.value_counts())
    freqStatsDf.columns = ['count']
    freqStatsDf.sort_values(by='count',
                            ascending=False,
                            inplace=True)
    if '__NULL__' in freqStatsDf.index:
        statsDict['n-miss'] = freqStatsDf.ix['__NULL__', 'count']
        statsDict['n-valid'] = freqStatsDf['count'].sum() - \
            freqStatsDf.ix['__NULL__', 'count']
    else:
        statsDict['n-miss'] = 0
        statsDict['n-valid'] = freqStatsDf['count'].sum()

    statsDict['n-unique'] = len(freqStatsDf.index.unique())
    charStatList = ['mean-or-top1', 'min-or-top2', 'p1-or-top3', 'p5-or-top4', 'p25-or-top5',
                    'median-or-bot5', 'p75-or-bot4', 'p95-or-bot3', 'p99-or-bot2', 'max-or-bot1']
    topCatStats = freqStatsDf[:5]
    bottomCatStats = freqStatsDf[-5:]

    for cherry in range(len(topCatStats)):
        statsDict[charStatList[cherry]] = freqStatsDf.ix[cherry].name + \
            "::" + str(freqStatsDf.ix[cherry, 'count'])

    for cherry in range(len(bottomCatStats)):
        cherry -= len(bottomCatStats)
        statsDict[charStatList[cherry]] = freqStatsDf.ix[cherry].name + \
            "::" + str(freqStatsDf.ix[cherry, 'count'])

    statsDict['ks-pvalue'] = '.'
    return statsDict


def py_data_desc(df):
    """
        :param df: 
        :type df: pd.DataFrame

        todo:
        需要做一个更强的数据类型分类
    """
    df = df[np.isnan(df["serial_number"])==False]
    dtypes = df.dtypes

    # char_lst = list(dtypes[dtypes == "object"].index)
    char_lst = list(
        dtypes[dtypes.isin([np.dtype("O"), np.dtype("<M8[ns]")])].index)
    # num_lst = list(dtypes[dtypes != "object"].index)
    num_lst = list(
        dtypes[~dtypes.isin([np.dtype("O"), np.dtype("<M8[ns]")])].index)

    result = []
    for x in char_lst:
        # tmp = _char_dd(df[x])
        tmp = _char_dd(df[x].astype(unicode))
        tmp["variable"] = x
        result.append(tmp)

    for x in num_lst:
        tmp = _num_dd(df[x])
        tmp["variable"] = x
        result.append(tmp)

    return pd.DataFrame(result)[["variable", "n-miss", "n-unique", "n-valid", "mean-or-top1", "min-or-top2", "p1-or-top3", "p5-or-top4", "p25-or-top5", "median-or-bot5", "p75-or-bot4", "p95-or-bot3", "p99-or-bot2", "max-or-bot1", "ks-pvalue"]]


######################################################################################

def ex_sinple():
    df = pd.DataFrame({"a": ["b", "c", "d", "e", "f"], "t": range(5)})
    print(py_data_desc(df))


def app_ex():
    df = pd.read_csv("../res/train_data_2015_2020.csv", encoding="utf-8")
    print(py_data_desc(df))
    dft = py_data_desc(df)
    dft.to_csv("../res/train_data_2015_2020_desc.csv", encoding="utf-8")


if __name__ == "__main__":
    app_ex()
