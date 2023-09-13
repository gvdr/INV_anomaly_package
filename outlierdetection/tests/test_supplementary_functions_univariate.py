
# general
import pandas as pd 
import numpy as np
import datetime


import functions_t_e_s_t as FT

import outlierdetection.univariate as UOD

import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
#PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, TEST_DIR)

print(TEST_DIR)

def MakePastFuture(base, numdays_past, numdays_future):
    past = [base + datetime.timedelta(days=x) for x in range(numdays_past)]
    future = [past[-1] + datetime.timedelta(days=x) for x in range(1, numdays_future+1)]
    return past, future



def test_clear_peak():

    df_bare = pd.read_csv(TEST_DIR + '/data/INV_ES_0_resistencia_daily_clear_peak_last.csv')
    df_bare['Date'] = df_bare['Date'].apply(pd.to_datetime)
    df_bare = df_bare.set_index('Date')

    # insert some NaNs into the training data to check whether this is handled
    df_bare.iloc[10:50, :] = np.nan
    df_bare.iloc[100, :] = np.nan
    df_bare.iloc[200:230, :] = np.nan

    ts_panda = pd.Series(df_bare['ES_0_resistencia_daily'])

    length_past = int(len(ts_panda) * 3 / 4)

    length_future = len(ts_panda) - length_past

    past, future = MakePastFuture(pd.to_datetime(df_bare.index[0]), length_past, length_future)

    OD = UOD.UnivariateOutlierDetection(ts_panda)
    OD.AutomaticallySelectDetectors(detector_window_length = length_future)
    print(OD.GetDetectorNames())
    assert isinstance(OD.GetSeries(), pd.Series)


    last_point_scores = OD.LastOutlierScore().iloc[0,:]
    result = OD.InterpretPointScore(last_point_scores)
    assert result[0] == True, "Clear extremal value outlier not detected."

    scores = OD.WindowOutlierScore(past, future)
    final_score = np.repeat(0, length_future)
    for k in range(length_future):
        isOutlier, _, _, _ = OD.InterpretPointScore(scores.loc[pd.to_datetime(future[k]), :])
        if isOutlier:
            final_score[k] = 1

    final_score = pd.Series(index=future, data=final_score)

    assert final_score[-1] == 1, "Clear extremal value outlier not detected."

    assert len(scores.index) == len(future)
    assert isinstance(scores, pd.DataFrame)
    assert len(scores.columns) > 0, "No detectors were selected for this data series."

