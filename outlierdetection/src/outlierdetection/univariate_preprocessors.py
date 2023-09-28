
"""
univariate_preprocessors.py
====================================
Contains the definition of preprocessors that can be applied to the time series data. 
"""

import pandas as pd
import numpy as np

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from sklearn.metrics import mean_squared_error

import warnings


def pp_average(self, processed_series, args):
    """
    Rolling average
     
    Performs a rolling average on the series over a window of size args[0] with right boundary at the current point.

    Parameters
    ----------
    processed_series : pd.Series with datetime index
        Time series to be processed
    args : list
        args[0] = window size for averaging w.r.t. time series steps 
    
    Returns
    -------
    processed_series : pd.Series with datetime index
        Processed series
    critical_error : bool
        Did a critical error occur during preprocessing?
    add_skip : int
        How many time steps should be skipped from the beginning of the series due to the preprocessing (here due to boundary effects of the rolling average)
    """

    critical_error = False
    add_skip = 0
    try:
        width = int(args[0])
        add_skip = width
        processed_series = processed_series.rolling(width, min_periods=int(width * 3/4), center=False, win_type=None, on=None, axis=0, closed=None, step=None, method='single').mean()
    except Exception as e:
        print("An exception occurred during pp_average: " + str(e))
        critical_error = True
    return processed_series, critical_error, add_skip


def pp_power(self, processed_series, args):
    """
    Raise to power
     
    Computes a power of the absolute value of the time series

    Parameters
    ----------
    processed_series : pd.Series with datetime index
        Time series to be processed
    args : list
        args[0] = power
    
    Returns
    -------
    processed_series : pd.Series with datetime index
        Processed series
    critical_error : bool
        Did a critical error occur during preprocessing?
    add_skip : int
        How many time steps should be skipped from the beginning of the series due to the preprocessing
    """

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
    """
    Median
     
    Substitutes time series points by the median over a window of size args[0] with right boundary at the current point.

    Parameters
    ----------
    processed_series : pd.Series with datetime index
        Time series to be processed
    args : list
        args[0] = window size
    
    Returns
    -------
    processed_series : pd.Series with datetime index
        Processed series
    critical_error : bool
        Did a critical error occur during preprocessing?
    add_skip : int
        How many time steps should be skipped from the beginning of the series due to the preprocessing
    """

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
    """
    Volatility
     
    Substitutes time series points by a measure of their volatility.
    Subtracts a local median from the data point and then takes the absolute value. 

    Parameters
    ----------
    processed_series : pd.Series with datetime index
        Time series to be processed
    args : list
        args[0] = window size over which median is computed. 
    
    Returns
    -------
    processed_series : pd.Series with datetime index
        Processed series
    critical_error : bool
        Did a critical error occur during preprocessing?
    add_skip : int
        How many time steps should be skipped from the beginning of the series due to the preprocessing
    """

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
    """
    Differencing
     
    Differences the time series with a specified n_shift. output = (1 - BACKSHIFT^n_shift) input 

    Parameters
    ----------
    processed_series : pd.Series with datetime index
        Time series to be processed
    args : list
        args[0] = n_shift.  
    
    Returns
    -------
    processed_series : pd.Series with datetime index
        Processed series
    critical_error : bool
        Did a critical error occur during preprocessing?
    add_skip : int
        How many time steps should be skipped from the beginning of the series due to the preprocessing
    """

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
    """
    Subtracting seasonality
     
    Subtracts a specified seasonality from the time series. 

    Parameters
    ----------
    processed_series : pd.Series with datetime index
        Time series to be processed
    args : list
        args[0] : int
            Peroid over which the seasonality is computed. 
        args[1] : int
            Seasonality is smoothed up this scale, i.e. neglected below the scale. 
        args[2] = str
            Seasonality type, either 'multiplicative' or 'additive. Time series values need to be >0 for multiplicative. 

    Returns
    -------
    processed_series : pd.Series with datetime index
        Processed series
    critical_error : bool
        Did a critical error occur during preprocessing?
    add_skip : int
        How many time steps should be skipped from the beginning of the series due to the preprocessing
    """
     
    critical_error = False       
    add_skip = 0 
    imputed_series = processed_series.copy()        
    try:
        #print(args)
        period = int(args[0])
        average_period = int(args[1])
        type = str(args[2])
        nans = np.where(processed_series.isnull())
        #print(nans)
        imputed_series, _, _ = self.pp_fillna_linear(processed_series)
        seasonal_result = seasonal_decompose(imputed_series, model=type, period=period).seasonal
        if average_period > 1:
            seasonal_result = seasonal_result.rolling(average_period, min_periods = 3, center=True).mean()
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
    """
    Linear imputation of missing values
     
    Fills missing values via Pandas interpolate with linear approximations between available points. 

    Parameters
    ----------
    processed_series : pd.Series with datetime index
        Time series to be processed
    args : list
        args[0] : int
            Maximal subsequent missing values that are imputed. If args=[] or not specified, this is taken as the series length. 
            
    Returns
    -------
    processed_series : pd.Series with datetime index
        Processed series
    critical_error : bool
        Did a critical error occur during preprocessing?
    add_skip : int
        How many time steps should be skipped from the beginning of the series due to the preprocessing
    """

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
    """
    MSTL Residual
     
    Returns the MSTL residual computed with CalcualteMSTL()

    Parameters
    ----------
    processed_series : pd.Series with datetime index
        Time series NOT USED HERE
    args : list
        args ARE NOT USED HERE
            
    Returns
    -------
    processed_series : pd.Series with datetime index
        Processed series
    critical_error : bool
        Did a critical error occur during preprocessing?
    add_skip : int
        How many time steps should be skipped from the beginning of the series due to the preprocessing
    """

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
    """
    MSTL Trend
     
    Returns the MSTL trend computed with CalcualteMSTL()

    Parameters
    ----------
    processed_series : pd.Series with datetime index
        Time series NOT USED HERE
    args : list
        args ARE NOT USED HERE
            
    Returns
    -------
    processed_series : pd.Series with datetime index
        Processed series
    critical_error : bool
        Did a critical error occur during preprocessing?
    add_skip : int
        How many time steps should be skipped from the beginning of the series due to the preprocessing
    """

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
    """
    MSTL Residual + Trend
     
    Returns the MSTL residual + trend computed with CalcualteMSTL()

    Parameters
    ----------
    processed_series : pd.Series with datetime index
        Time series NOT USED HERE
    args : list
        args ARE NOT USED HERE
            
    Returns
    -------
    processed_series : pd.Series with datetime index
        Processed series
    critical_error : bool
        Did a critical error occur during preprocessing?
    add_skip : int
        How many time steps should be skipped from the beginning of the series due to the preprocessing
    """

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
    """
    Skip time steps at beginning
     
    Skips a number of time steps from the beginning of the series via adding its argument to the skip counter

    Parameters
    ----------
    processed_series : pd.Series with datetime index
        Time series to be processed
    args : list
        args[0] : int
            Numer of time steps to be skipped
   
            
    Returns
    -------
    processed_series : pd.Series with datetime index
        Processed series
    critical_error : bool
        Did a critical error occur during preprocessing?
    add_skip : int
        How many time steps should be skipped from the beginning of the series due to the preprocessing
    """

    critical_error = False       
    try:
        add_skip = args[0] 
    except Exception as e:
        print("An exception occurred during pp_skip_from_beginning:" + str(e))
        critical_error = True
    return processed_series, critical_error, add_skip

def pp_restrict_data_to(self, processed_series, args=[]):
    """
    Restricts series to later points
     
    Skips a number of time steps from the beginning so that a total window of args[0] + args[1] at the end of the series is kept

    Parameters
    ----------
    processed_series : pd.Series with datetime index
        Time series to be processed
    args : list
        args[0] : int
            size of training window to be kept
        args[1] : int
            size of test window to be kept
            
    Returns
    -------
    processed_series : pd.Series with datetime index
        Processed series
    critical_error : bool
        Did a critical error occur during preprocessing?
    add_skip : int
        How many time steps should be skipped from the beginning of the series due to the preprocessing
    """

    critical_error = False       
    try:
        training_length = args[0] 
        test_length = args[1]
        add_skip = len(self.series) - training_length - test_length
    except Exception as e:
        print("An exception occurred during pp_restrict_data_to:" + str(e))
        critical_error = True
    return processed_series, critical_error, add_skip


def pp_ARIMA_subtract(self, processed_series, args):
    """
    Subtracting ARIMA
     
    Subtracts a specified or estimated ARIMA model from the time series. 

    Parameters
    ----------
    processed_series : pd.Series with datetime index
        Time series to be processed
    args : list
        args[0] : int
            Peroid over which the seasonality is computed. 
        args[1] : int
            Seasonality is smoothed up this scale, i.e. neglected below the scale. 
        args[2] = str
            Seasonality type, either 'multiplicative' or 'additive. Time series values need to be >0 for multiplicative. 

    Returns
    -------
    processed_series : pd.Series with datetime index
        Processed series
    critical_error : bool
        Did a critical error occur during preprocessing?
    add_skip : int
        How many time steps should be skipped from the beginning of the series due to the preprocessing
    """
     
    critical_error = False       
    add_skip = 0 
    imputed_series = processed_series.copy() 
    counter = 0     
    try:
        
        if args[1] == 'stat':
            imputed_series, critical_error, add_skip_diff, counter  = self.pp_difference_until_stationary(imputed_series, args=[0, 0.05])
            add_skip += add_skip_diff
        else: 
            d = int(args[1])
            if d > 0:
                for _ in range(d):
                    imputed_series = imputed_series - imputed_series.shift(1)
                    add_skip += 1
                    counter += 1

        #print(add_skip)

        #print(args)
        if args[0] == 'pacf':
            use_pacf = True
        else: 
            use_pacf = False
            p_arg = int(args[0])
            if p_arg >= 0:
                p_range = [p_arg]
            elif p_arg < 0:
                p_range = np.arange(int(-1 * p_arg))
        
        if args[1] != 0:
            # difference until 
            pass

        q_arg = int(args[2])
        if q_arg >= 0:
            q_range = [q_arg]
        elif q_arg < 0:
            q_range = np.arange(int(-1 * q_arg))

        nans = np.where(processed_series.isnull())

        imputed_series, _, _ = self.pp_fillna_linear(processed_series)

        if use_pacf:
            max_lags = min(10, int(len(self.series) / 2))
            crit_val = 0.05
            _, conf = pacf(imputed_series, alpha=0.01, nlags = max_lags)
            p = 0
            while(p < max_lags):
                if conf[p+1][0] < crit_val:
                    break
                else:
                    p += 1
            if p == 0:
                p_range = [0,1]
            else:
                p_range = [p] # require at least 1 AR term to test 


        def error_metric(model, error):
            return model.aic * error # custom error metric that seems somewhat better than just simple AIC

        best_order = [0, 0, 0]
        best_fit = None
        best_metric = None
        best_model = None
        for p in p_range:
            for q in q_range:
                order = [p,0,q]
                warnings.filterwarnings("ignore")
                model = ARIMA(imputed_series, order=order).fit()
                predictions = model.fittedvalues
                error = mean_squared_error(imputed_series, predictions)
                #print(order)
                if best_metric == None:
                    best_metric = error_metric(model,  error)
                    best_order = order
                    best_fit = predictions
                    best_model = model
                else:
                    if error_metric(model,  error) < best_metric:
                        best_metric = error_metric(model,  error)
                        best_order = order
                        best_fit = predictions
                        best_model = model

        best_order[1] = counter

        R2 = 1 - (imputed_series - best_fit).var() / imputed_series.var()

        

        def print_ARIMA_result_formatted(best_order, best_model, R2):
            print("----- ARIMA estimation ------")
            print(f"Best ARIMA order estimated as {best_order}.")
            print(best_model.specification)
            ar_list = []
            ma_list = []
            for arc in best_model.polynomial_ar[1:]:
                ar_list.append( - arc)
            for mac in best_model.polynomial_ma[1:]:
                ma_list.append( - mac)
            if ar_list:
                ar_list = ["%.3f" % elem for elem in ar_list]
                print(f"AR: {', '.join(ar_list)}")
            if ma_list:
                ma_list = ["%.3f" % elem for elem in ma_list]
                print(f"MA: {', '.join(ma_list)}")
            print(f"R2 value: {R2:0.3f}")
            print("-----------------------------")

        print_ARIMA_result_formatted(best_order, best_model, R2)


        crit_R2 = 0.1
        if R2 < crit_R2:
            raise Exception(f"R2 value of ARIMA model is below critical value {crit_R2}.")
            critical_error = True

        #print(best_model.specification)
        #print("AR:")
        #print(best_model.polynomial_ar)
        #print("MA:")
        #print(best_model.polynomial_ma)

        imputed_series -= best_fit

        if nans:
            imputed_series.iloc[nans] = np.nan
        #print(imputed_series)
    except Exception as e:
        print("An exception occurred during pp_ARIMA_subtract:" + str(e))
        critical_error = True
    return imputed_series, critical_error, add_skip  


def pp_difference_until_stationary(self, processed_series, args=[0, 0.05]):
    """
    Difference time series until stationary
     
    Skips a number of time steps from the beginning of the series via adding its argument to the skip counter

    Parameters
    ----------
    processed_series : pd.Series with datetime index
        Time series to be processed
    args : list
        args[0] : int
            >0 Difference at most args[0]
            =0 difference until stationary, arbitrarily many times
        args[1] : float
            critical ADF p-value under which stationarity is concluded 
   
            
    Returns
    -------
    processed_series : pd.Series with datetime index
        Processed series
    critical_error : bool
        Did a critical error occur during preprocessing?
    add_skip : int
        How many time steps should be skipped from the beginning of the series due to the preprocessing
    counter : number of differentiations performed
    """

    critical_error = False     
    add_skip = 0  
    try:
        max_difference = args[0]
        threshold = args[1]
        counter = 0
        adf_pvalue = adfuller(processed_series.dropna())[1]
        #print(adf_pvalue)
        #print(threshold)
        while(adf_pvalue >= threshold):
            print(f"Series not stationary. ADF =  {adf_pvalue} > {threshold}. Differencing. ")
            processed_series = processed_series - processed_series.shift(1)
            add_skip += 1
            counter += 1
            if max_difference == counter:
                break
            adf_pvalue = adfuller(processed_series.dropna())[1]
         
    except Exception as e:
        print("An exception occurred during pp_difference_until_stationary:" + str(e))
        critical_error = True
    return processed_series, critical_error, add_skip, counter

