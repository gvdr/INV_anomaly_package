from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

def IF(self, processed_series, training, test, args):

        outliers_fraction = float(args[0])

        # Normal scale training and then test from these values
        m = processed_series.loc[training].mean()
        s = processed_series.loc[training].std()

        training_data_scaled = (processed_series.loc[training].dropna() - m )/s
        
        model = IsolationForest(contamination=outliers_fraction)
        model.fit(np.array(training_data_scaled).reshape(-1, 1))

        results_no_nan = np.array((model.predict(np.array((processed_series.loc[test].dropna() - m)/s).reshape(-1, 1)) -1) / (-2)).astype(int) # 1 for outlier, zero for no outlier
        
        # Add NaN test values as NaN by hand. Loop through the test series and replace non-NaN values by the IF result.
        results = processed_series.loc[test].copy()
        j=0
        for i in range(len(test)):
            if not np.isnan(results.iloc[i]):
                results[i] = results_no_nan[j]
                j+=1
        return results