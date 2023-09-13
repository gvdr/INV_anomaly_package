
# general
import pandas as pd 
import numpy as np

import functions_t_e_s_t as FT

import outlierdetection.univariate as UOD

import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
#PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, TEST_DIR)

print(TEST_DIR)









def test_clear_peak():

    df_bare = pd.read_csv(TEST_DIR + '/data/INV_ES_0_resistencia_daily_clear_peak_last.csv')
    df_bare['Date'] = df_bare['Date'].apply(pd.to_datetime)
    df_bare = df_bare.set_index('Date')

    # insert some NaNs into the training data to check whether this is handled
    df_bare.iloc[10:50, :] = np.nan
    df_bare.iloc[100, :] = np.nan
    df_bare.iloc[200:230, :] = np.nan

    ts = df_bare.loc[:, 'ES_0_resistencia_daily'].to_list()
    ts_dates = df_bare.index.to_list()

    isOutlier, max_strength, message_detail, result_responses = UOD.process_last_point(ts, ts_dates)

    print(max_strength)
    print(message_detail)
    print(result_responses)


    assert isOutlier == True, "Clear extremal value outlier not detected by process_last_point(ts, ts_dates). Daily data from INVTEREST."
    assert message_detail
    assert result_responses


def test_seasonal_outlier():

    df = pd.read_csv(TEST_DIR + '/data/NAB/data/artificialWithAnomaly/art_daily_nojump.csv')

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    df = df.loc[:pd.to_datetime('2014-04-11 09:00:00'), :] # this value is a clear seasonal anomaly that is not a value outlier

    print(df.tail())


    ts = df.loc[:, 'value'].to_list()
    ts_dates = df.index.to_list()

    isOutlier, max_strength, message_detail, result_responses = UOD.process_last_point(ts, ts_dates)

    print(max_strength)
    print(message_detail)
    print(result_responses)


    assert isOutlier == True, "Clear seasonal outlier not detected by process_last_point(ts, ts_dates). 5 minute data from NAB."
    assert message_detail
    assert result_responses



