import numpy as np
import pandas as pd


def PRE(self, processed_series, training, test, args = [1]):

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
