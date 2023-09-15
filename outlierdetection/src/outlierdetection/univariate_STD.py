import numpy as np
import pandas as pd

def WindowScoreSTDWithErrors(self, processed_series, training, test, args = [1]):

    z = float(args[0])
    
    m = np.mean(processed_series.loc[training])
    s = np.std(processed_series.loc[training]) * z
    test_mean = pd.DataFrame(index=test, data={'mean' : m})
    test_high = pd.DataFrame(index=test, data={'high' : m+s})
    test_low = pd.DataFrame(index=test, data={'low' : m-s})

    return (processed_series.loc[test] - m ) / s, test_mean, test_high, test_low, (processed_series.loc[training] - m ) / s


def STD(self, processed_series, training, test, args = [1]):
    """
    Standard deviation detector
     
    Computes mean and standard deviation of training values. Outlier score is the number of standard deviations a data point is away from the training mean. 

    Parameters
    ----------
    processed_series : pd.Series with datetime index
        Time series to be processed
    args : list
        args[0] : int
            Relative number of standard deviations that enters the score. Should be set to 1 unless the ramifications of another value are understood and handled properly.  
   
            
    Returns
    -------
    result : float
        Outlier score. Number of training standard deviations a test data point deviates from the training mean. 
    """

    z = float(args[0])
    
    m = np.mean(processed_series.loc[training])
    s = np.std(processed_series.loc[training]) * z

    result = (processed_series.loc[test] - m ) / s

    #result.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return result
