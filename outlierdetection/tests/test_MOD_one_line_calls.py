# general
import pandas as pd 
import numpy as np
import json

import functions_t_e_s_t as FT

import outlierdetection.multivariate as MOD

import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
#PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, TEST_DIR)

print(TEST_DIR)

f = open(TEST_DIR + '/data/multivariate_data_input.json')
json_format_data = json.load(f)


#print(time_series)

# Many functions below use the same t


def test_multivariate_json_format():

    # this  test is used only to test whether the input data format is correclty handled by the one line call. 

    isOutlier, max_strength, message_detail, result_responses = MOD.process_last_point_with_window(json_format_data, window_size=500)

    print(max_strength)
    print(message_detail)
    print(result_responses)

   #assert False

    #assert isOutlier == True, "Clear extremal value outlier not detected by process_last_point(ts, ts_dates). Daily data from INVTEREST."
    #assert isOutlier == False
    assert max_strength > 0
    #assert message_detail
    assert result_responses



    #assert len(scores.index) == 400
    #assert isinstance(scores, pd.DataFrame)
    #assert len(scores.columns) > 0
    #assert scores.isna().sum().sum() == 0
