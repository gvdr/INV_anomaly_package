from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

def IF(self, processed_data, training, test, args=[0.05]):

    try:
        outliers_fraction = args[0]

        # Normal scale training and then test from these values
        m = processed_data.loc[training].mean()
        s = processed_data.loc[training].std()

        training_data_scaled = (processed_data.loc[training].dropna() - m )/s

        model =  IsolationForest(contamination=outliers_fraction)
        model.fit(training_data_scaled)

        test_no_na = np.array(processed_data.loc[test].dropna())

        results = np.zeros(len(test))

        if len(test_no_na) > 0:
            results_no_nan = np.array((model.predict(np.array((processed_data.loc[test].dropna() - m)/s)) -1) / (-2)).astype(bool) # 1 for outlier, zero for no outlier
            j=0
            for i in range(len(test)):
                if not processed_data.loc[test].iloc[i,:].isnull().values.any():
                    results[i] = results_no_nan[j]
                    j+=1
                else:
                    results[i] = np.nan
            return results
        else:
            return np.repeat(False, len(test))
    
    except Exception as e:
        print("An exception occurred during IF:" + str(e))
        return np.repeat(np.nan, len(test))           

  