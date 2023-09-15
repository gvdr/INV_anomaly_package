import numpy as np
import pandas as pd


def PRE(self, processed_series, training, test, args = [1]):
    """
    Previous value comparison detector
     
    Computes a histogram of the previously seen (training) data and evalutes the test data based on whether it has been seen before, is within the previously seen range, or significanlty lies outside the previously seen range. 

    Parameters
    ----------
    processed_series : pd.Series with datetime index
        Time series to be processed
    args : list
        args[0] : int
            Number of bins used in constructing the histogram
   
            
    Returns
    -------
    result : float
        Outlier score. 
            Smaller than 1: Value was seen before (is within a histogram bin) and the probability of the bin is 1 - result. 

            Equal to 1: The value was not seen before (is not within a non-zero histogram bin), but falls within the range (min to max) of previously seen values. 
            
            Larger than 1: The value has not been seen before and differs result * 100% from the previously seen range (max - min)
    """

    z = float(args[0])

    n_bins = args[0]

    high = np.max(processed_series.loc[training])
    low = np.min(processed_series.loc[training])

    if high == np.nan or low == np.nan:
        return np.nan
    
    if high == low:
        result = np.repeat(0.0, len(test))
        for i in range(len(test)):
            if processed_series.loc[test[i]] > high:
                result[i]=np.inf
            elif processed_series.loc[test[i]] < high:
                result[i]= - np.inf
        return result

    diff = high - low

    h = np.histogram(processed_series.loc[training].dropna(), bins = n_bins, density=True)
    
    bin_edges = h[1]

    result = pd.Series(np.nan, index = test)

    for idx in test:
        if processed_series[idx] > high:
            result[idx] = np.abs(processed_series[idx] - low) / diff
        elif processed_series[idx] < low:
            result[idx] = -1.0 * np.abs(high - processed_series[idx]) / diff
        else:
            for i in range(n_bins):
                if processed_series[idx] <= bin_edges[i+1]:
                    result[idx] = 1.0 - h[0][i] * diff / n_bins # score closer to 1 for lower probability of before seen value
                    break
    
    return result
