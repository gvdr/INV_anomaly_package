
# general
import numpy as np 
import pandas as pd 
import warnings


class MultivariateOutlierDetection:

    from .multivariate_IF import IF
    from .multivariate_MD import MD

    
    def __init__(self, data):

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Passed data is not a pd.DataFrame.")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("Index of passed data is not a pd.DatetimeIndex.")

        for x in data:
            if not pd.api.types.is_numeric_dtype(data[x]):
                raise TypeError(f"Column {x} of passed data is not numeric.")
            
        # Check for duplicate indices:
        duplicated_indices = data.index.duplicated()
        if duplicated_indices.sum() > 0:
            warnings.warn(f"Warning: passed data contains duplicated indices. Removing duplicates, keeping firsts only.\n")
            data = data[~duplicated_indices]
            
        self.data = data    

        nan_times = data[~data.index.isin(data.dropna().index)].index.to_list()

        if len(nan_times) > 0:
            warnings.warn(f"Warning: passed series contains {len(nan_times)} nans:\n{nan_times}\n")

        self.SetStandardDetectors()

        self.min_training_data = 5


    def GetData(self):
        return self.data
    

    def AddDetector(self, new_detector):
        if new_detector[0] == 'MD':
            new_detector[0] = self.MD
        if new_detector[0] == 'IF':
            new_detector[0] = self.IF

        self.detectors.append(new_detector)


    def ClearDetectors(self):
        self.detectors = []


    def SetStandardDetectors(self):
        self.ClearDetectors()
        self.AddDetector(['MD', [1], []])
        self.AddDetector(['IF', [0.05], []])


    def LastOutlierScore(self, window = None):
        if window == None:
            window = self.data.index.to_list()
        training = window[:-1]
        test = window[-1:]
        return self.WindowOutlierScore(training, test)


    def WindowOutlierScore(self, training, test):

        perform_detection = True

        df = self.data.loc[training, :]
        nan_times = df[~df.index.isin(df.dropna().index)].index.to_list()

        if len(nan_times) > 0:
            warnings.warn(f"Warning: training data contains {len(nan_times)} out of {len(training)} NaNs:\n{nan_times}\n")
        if len(nan_times) == len(training):
            warnings.warn(f"Warning: No valid training data! Output will be NaN only!\n")
            perform_detection = False
        if len(training) - len(nan_times) < self.min_training_data:
            warnings.warn(f"Warning: Non NaN training data less than {self.min_training_data}. No testing performed. Output will be NaN only!\n")
            perform_detection = False

        df = self.data.loc[test, :]
        nan_times = df[~df.index.isin(df.dropna().index)].index.to_list()

        if len(nan_times) > 0:
            warnings.warn(f"Warning: test data contains {len(nan_times)} out of {len(test)} nans:\n{nan_times}\n")
        if len(nan_times) == len(test):
            warnings.warn(f"Warning: No valid test data! Output will be NaN only!\n")
            perform_detection = False

        result = pd.DataFrame(index = test)

        for detector_tuple in self.detectors:
            detector = detector_tuple[0]
            arguments = detector_tuple[1]
            preprocessor = detector_tuple[2]
            name = detector.__name__ + "["
            for x in arguments:
                name = name + str(x) + ","
            name = name[:-1]
            name += "]"
            if perform_detection:
                if preprocessor:
                    processed_data = self.data.copy()
                    name += "["
                    for p in preprocessor:
                        type = p[0]
                        if type == 'A':
                            width = int(p[1:])
                            processed_data = processed_data.rolling(width, min_periods=1, center=True, win_type=None, on=None, axis=0, closed=None, step=None, method='single').mean()
                        if type == 'S':
                            n_shift = int(p[1:])
                            processed_data = processed_data - processed_data.shift(n_shift)
                        name += p
                    name += "]"
                    result[name] = detector(processed_data, training, test, arguments)
                else:
                    result[name] = detector(self.data, training, test, arguments)
            else:
                result[name] = np.nan

        return result

      






####### Older code, maybe reuse        

    # For now test set should be directly after the training set
#    def WindowScoreKatsVAR(self, training, test, alpha = 0.05):

        # Seems that the KatsVAR code cannot handle missing values. 

#        all_indices = training + test

#        multi_anomaly_df = self.data.loc[all_indices, :].copy()
#        multi_anomaly_df['time'] = multi_anomaly_df.index
        #print(multi_anomaly_df)
        
#        multi_anomaly_df_to_ts = multi_anomaly_df.copy()
#        multi_anomaly_ts = TimeSeriesData(multi_anomaly_df_to_ts)
        #print(multi_anomaly_df_to_ts['time'])
#        test_idx=-1
        #Find first test index

#        for x in test:
            #print(x.strftime('%Y-%m-%d'))
#            if x.strftime('%Y-%m-%d') in multi_anomaly_df_to_ts['time']:
#                test_idx = multi_anomaly_df_to_ts.index.get_loc(x)
#                break

        #print(f"First test index is {test_idx}")


        #print(multi_anomaly_ts)

        # Plot the data
        #multi_anomaly_ts.plot(cols=multi_anomaly_ts.value.columns.tolist())
#        np.random.seed(10)
#        params = VARParams(maxlags=2)
#        d = MultivariateAnomalyDetector(multi_anomaly_ts, params, training_days=test_idx)
#        anomaly_score_df = d.detector()
        #print(anomaly_score_df)
        
        #d.plot()


        ##### Anomaly p-values are calcualted below. For now, alpha is not used

        #anomalies = d.get_anomaly_timepoints(alpha=alpha)
        
        #print(anomalies[0])
        #print(d.get_anomalous_metrics(anomalies[0], top_k=3))

#        return anomaly_score_df['overall_anomaly_score']
        #return anomaly_score_df['p_value']

        


        
        