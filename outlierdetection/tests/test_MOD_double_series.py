
# general
import pandas as pd 
import numpy as np

import functions_t_e_s_t as FT

import outlierdetection.multivariate as MOD

import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
#PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, TEST_DIR)

print(TEST_DIR)

df_bare = pd.read_csv(TEST_DIR + '/data/two_time_series.csv')
df_bare['Date'] = df_bare['Date'].apply(pd.to_datetime)
df_bare = df_bare.set_index('Date')

time_series = df_bare.dropna()

#print(time_series)

# Many functions below use the same t


def test_scores_no_nan():

    time_series_changed = time_series.copy()

    t = MOD.MultivariateOutlierDetection(time_series_changed)

    t.ClearDetectors()
    t.AutomaticallySelectDetectors()
    
    past, future = FT.MakePastFuture(pd.to_datetime('2010-01-01'), 3000, 400)    
    scores = t.WindowOutlierScore(past, future)

    assert len(scores.index) == 400
    assert isinstance(scores, pd.DataFrame)
    assert len(scores.columns) > 0
    assert scores.isna().sum().sum() == 0



def test_scores_training_nans():

    # Test with a time series where NaNs are injected into training data 

    time_series_changed = time_series.copy()


    past, future = FT.MakePastFuture(pd.to_datetime('2010-01-01'), 2700, 150)

    time_series_changed.iloc[100:250] = np.nan
    time_series_changed.iloc[500:700] = np.nan

    t = MOD.MultivariateOutlierDetection(time_series_changed)
    t.ClearDetectors()
    t.AutomaticallySelectDetectors()


    scores = t.WindowOutlierScore(past, future)

    assert len(scores.index) == 150
    assert isinstance(scores, pd.DataFrame)
    assert len(scores.columns) > 0
    assert scores.isna().sum().sum() == 0

    

def test_scores_training_nans_and_test_nans():

    # Test with a time series where nan are injected into training data and test data


    past, future = FT.MakePastFuture(pd.to_datetime('2010-01-01'), 2700, 150)

    time_series_changed = time_series.copy()

    # training nans
    time_series_changed.iloc[10:15] = np.nan
    time_series_changed.iloc[100:150] = np.nan

    # total of 52 test nans
    time_series_changed.iloc[2750:2800] = np.nan
    time_series_changed.iloc[2846] = np.nan
    time_series_changed.iloc[2848] = np.nan

    t = MOD.MultivariateOutlierDetection(time_series_changed)

    t.SetStandardDetectors()


    scores = t.WindowOutlierScore(past, future)

    print(scores)

    assert len(scores.index) == 150
    assert isinstance(scores, pd.DataFrame)
    assert len(scores.columns) > 0
    assert scores.isna().sum().sum() == len(scores.columns) * 52



def test_scores_training_only_nans():

    # Test with a time series where training data contains only nans

    past, future = FT.MakePastFuture(pd.to_datetime('2010-01-01'), 2700, 150)

    time_series_changed = time_series.copy()

    time_series_changed.iloc[0:2700] = np.nan

    t = MOD.MultivariateOutlierDetection(time_series_changed)

    t.ClearDetectors()
    t.AutomaticallySelectDetectors()


    scores = t.WindowOutlierScore(past, future)

    assert len(scores.index) == 150
    assert isinstance(scores, pd.DataFrame)
    assert len(scores.columns) > 0
    assert scores.isna().sum().sum() == len(scores.columns) * 150 # There should be only NAN now in the test data as no training data was given 


def test_scores_test_only_nans():

    # Test with a time series where test data contains only nans

    past, future = FT.MakePastFuture(pd.to_datetime('2010-01-01'), 2700, 150)

    time_series_changed = time_series.copy()

    time_series_changed.iloc[2700:] = np.nan

    t = MOD.MultivariateOutlierDetection(time_series_changed)

    t.ClearDetectors()
    t.AutomaticallySelectDetectors()


    scores = t.WindowOutlierScore(past, future)

    assert len(scores.index) == 150
    assert isinstance(scores, pd.DataFrame)
    assert len(scores.columns) > 0
    assert scores.isna().sum().sum() == len(scores.columns) * 150 # There should be only NAN now in the test data as no non-Nan test data was given 



def test_scores_training_has_too_few_points():

    # Test with a time series where the training data is too few points

    past, future = FT.MakePastFuture(pd.to_datetime('2010-01-01'), 3, 150)

    

    t = MOD.MultivariateOutlierDetection(time_series)

    t.ClearDetectors()
    t.AutomaticallySelectDetectors()


    scores = t.WindowOutlierScore(past, future)

    assert len(scores.index) == 150
    assert isinstance(scores, pd.DataFrame)
    assert len(scores.columns) > 0
    assert scores.isna().sum().sum() == len(scores.columns) * 150 # There should be only NAN now in the test data as no non-Nan test data was given 


def test_scores_test_data_empty():

    # Test with a time series where the test data labels are empty. The error that is produced here is an error of the calling function. The scoring data frame returned is empty, as it should be. 

    past, future = FT.MakePastFuture(pd.to_datetime('2010-01-01'), 300, 0)

    t = MOD.MultivariateOutlierDetection(time_series)

    t.ClearDetectors()
    t.AutomaticallySelectDetectors()


    scores = t.WindowOutlierScore(past, future)

    assert len(scores.index) == 0
    assert isinstance(scores, pd.DataFrame)
    assert len(scores.columns) > 0
