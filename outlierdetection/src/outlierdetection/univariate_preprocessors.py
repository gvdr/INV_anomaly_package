
"""
univariate_preprocessors.py
====================================
Contains the definition of preprocessors that can be applied to the time series data. 
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


def pp_average(self, processed_series, args):
    critical_error = False
    add_skip = 0
    try:
        width = int(args[0])
        add_skip = width
        processed_series = processed_series.rolling(width, min_periods=int(width / 2), center=False, win_type=None, on=None, axis=0, closed=None, step=None, method='single').mean()
    except Exception as e:
        print("An exception occurred during pp_average: " + str(e))
        critical_error = True
    return processed_series, critical_error, add_skip


def pp_power(self, processed_series, args):
    critical_error = False       
    add_skip = 0         
    try:
        power = int(args[0])
        processed_series = np.power(np.abs(processed_series), power)
    except Exception as e:
        print("An exception occurred during pp_power:" + str(e))
        critical_error = True
    return processed_series, critical_error, add_skip                    


def pp_median(self, processed_series, args):
    critical_error = False       
    add_skip = 0         
    try:
        width = int(args[0])
        processed_series = processed_series.rolling(width, min_periods=1, center=False, win_type=None, on=None, axis=0, closed=None, step=None, method='single').median()
        processed_series.replace([np.inf, -np.inf], np.nan, inplace=True)
        add_skip += width
    except Exception as e:
        print("An exception occurred during pp_median:" + str(e))
        critical_error = True
    return processed_series, critical_error, add_skip   


def pp_volatility(self, processed_series, args):
    critical_error = False       
    add_skip = 0         
    try:
        width = int(args[0])
        add_skip += width
        ma_series = processed_series.rolling(width, min_periods=1, center=False, win_type=None, on=None, axis=0, closed=None, step=None, method='single').median()
        ma_series.replace([np.inf, -np.inf], np.nan, inplace=True)
        processed_series -= ma_series
        processed_series = np.abs(processed_series)
    except Exception as e:
        print("An exception occurred during pp_volatility:" + str(e))
        critical_error = True
    return processed_series, critical_error, add_skip   


def pp_difference(self, processed_series, args):
    critical_error = False       
    add_skip = 0         
    try:
        n_shift = int(args[0])
        add_skip += n_shift
        processed_series = processed_series - processed_series.shift(n_shift)
    except Exception as e:
        print("An exception occurred during pp_difference:" + str(e))
        critical_error = True
    return processed_series, critical_error, add_skip   

      
def pp_season_subtract(self, processed_series, args):
    critical_error = False       
    add_skip = 0         
    try:
        period = int(args[0])
        seasonal_result = seasonal_decompose(processed_series, period=period)
        processed_series -= seasonal_result.seasonal 
    except Exception as e:
        print("An exception occurred during pp_season_subtract:" + str(e))
        critical_error = True
    return processed_series, critical_error, add_skip  
