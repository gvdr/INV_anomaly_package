
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
    imputed_series = processed_series.copy()        
    try:
        #print(args)
        period = int(args[0])
        type = str(args[1])
        nans = np.where(processed_series.isnull())
        #print(nans)
        imputed_series, _, _ = self.pp_fillna_linear(processed_series)
        seasonal_result = seasonal_decompose(imputed_series, model=type, period=period).seasonal
        if period == 365:
            seasonal_result = seasonal_result.rolling(30, min_periods = 3, center=True).mean()
        if type == 'additive':
            imputed_series -= seasonal_result 
        if type == 'multiplicative':
            imputed_series /= seasonal_result 
        if nans:
            imputed_series.iloc[nans] = np.nan
        #print(imputed_series)
    except Exception as e:
        print("An exception occurred during pp_season_subtract:" + str(e))
        critical_error = True
    return imputed_series, critical_error, add_skip  


def pp_fillna_linear(self, processed_series, args=[]):
    critical_error = False       
    add_skip = 0
    try:    
        if not args:
            fill_limit = len(processed_series)
        else:
            fill_limit = args[0]
        processed_series = processed_series.interpolate(method='linear', limit=fill_limit)
        processed_series = processed_series.interpolate(method='linear', limit_direction='backward', limit=fill_limit)
    except Exception as e:
        print("An exception occurred during pp_fillna_linear:" + str(e))
        critical_error = True
    return processed_series, critical_error, add_skip


def pp_get_resid(self, processed_series, args=[]):
    critical_error = False       
    add_skip = 0  
    try:  
        if self.resid is None:
            self.CalculateMSTL()
    except Exception as e:
        print("An exception occurred during pp_get_resid:" + str(e))
        critical_error = True
    return self.resid, critical_error, add_skip


def pp_get_trend(self, processed_series, args=[]):
    critical_error = False       
    add_skip = 0    
    try:
        if self.trend is None:
            self.CalculateMSTL()
    except Exception as e:
        print("An exception occurred during pp_get_trend:" + str(e))
        critical_error = True
    return self.trend, critical_error, add_skip


def pp_get_trend_plus_resid(self, processed_series, args=[]):
    critical_error = False       
    add_skip = 0    
    try:
        if self.trend is None:
            self.CalculateMSTL()
    except Exception as e:
        print("An exception occurred during pp_get_trend_plus_resid:" + str(e))
        critical_error = True
    return self.trend + self.resid, critical_error, add_skip


def pp_skip_from_beginning(self, processed_series, args=[0]):
    critical_error = False       
    try:
        add_skip = args[0] 
    except Exception as e:
        print("An exception occurred during pp_skip_from_beginning:" + str(e))
        critical_error = True
    return processed_series, critical_error, add_skip

def pp_restrict_data_to(self, processed_series, args=[]):
    critical_error = False       
    try:
        training_length = args[0] 
        test_length = args[1]
        add_skip = len(self.series) - training_length - test_length
    except Exception as e:
        print("An exception occurred during pp_restrict_data_to:" + str(e))
        critical_error = True
    return processed_series, critical_error, add_skip
