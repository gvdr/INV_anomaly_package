
# general
import numpy as np 
import pandas as pd 
import warnings
from datetime import timedelta
import json

class MultivariateOutlierDetection:

    from .multivariate_IF import IF
    from .multivariate_MD import MD

    from .univariate_preprocessors import pp_average, pp_power, pp_median, pp_volatility, pp_difference, pp_season_subtract, pp_fillna_linear, pp_get_resid, pp_get_trend, pp_get_trend_plus_resid, pp_skip_from_beginning, pp_restrict_data_to, pp_ARIMA_subtract, pp_difference_until_stationary


    
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

        self.preprocessor_returns = dict()

        self.min_training_data = 5

        nan_times = data[~data.index.isin(data.dropna().index)].index.to_list()

        if len(nan_times) > 0:
            warnings.warn(f"Warning: passed series contains {len(nan_times)} nans:\n{nan_times}\n")

        self.ClearDetectors()
        self.SetStandardDetectors()

        self.min_training_data = 5


    def GetData(self):
        return self.data
    
    def PrintPreprocessorReturns(self):
        print(self.preprocessor_returns)
    

    #def AddDetector(self, new_detector):
    #    if new_detector[0] == 'MD':
    #        new_detector[0] = self.MD
    #    if new_detector[0] == 'IF':
    #        new_detector[0] = self.IF

    #    self.detectors.append(new_detector)

    def AddDetector(self, new_detector):
        """
        Appends a new detector to the list of already existing detectors. 

        Parameters
        ----------
        new_detector : list, e.g. ['STD', [relative_sigma], [['season_subtract', [season, average_period, seasonality_type]]], sigma_STD]
            [0]: string 
                type of the detector, e.g. 'STD', 'PRE', ...
            [1]: list
                detector parameters
            [2]: list
                series preprocessor directives
            [3]: float
                detector threshold

        """
        self.num_detectors += 1
        new_detector.append(new_detector[0]) # store the name before turning into a function
        new_detector.append(self.num_detectors) # 1, 2, 3 ... detector ID
        new_detector[0] = getattr(self, new_detector[0])
        self.detectors.append(new_detector)


    def ClearDetectors(self):
        self.detectors = []
        self.num_detectors = 0


    def SetStandardDetectors(self):
        self.ClearDetectors()
        self.AddDetector(['MD', [1], [], 4])
        self.AddDetector(['IF', [0.05], [], None])


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

            name = "D_" + str(detector_tuple[5])

            preprocessor_return_list = []
            
            if perform_detection:
                if preprocessor:
                    perform_detection_after_preprocessing = True
                    processed_data = self.data.copy()
                    #name += "["
                    skip = 0
                    critical_error = False
                    for p in preprocessor:
                        pp_type = p[0]
                        pp_args = p[1]
                        pp_func = getattr(self, 'pp_' + pp_type)

                        preprocessor_return_list_data = []

                        add_skip = 0

                        

                        for s in processed_data:
                            processed_data[s], critical_error_series, add_skip_series, pp_return_series = pp_func(processed_data[s], pp_args)

                            if pp_return_series:
                                preprocessor_return_list_data.append(pp_return_series)

                            critical_error = critical_error or critical_error_series

                            add_skip = max(add_skip, add_skip_series)

                        if preprocessor_return_list_data:
                            preprocessor_return_list.append(preprocessor_return_list_data)

                        skip += add_skip
                        if critical_error:
                            perform_detection_after_preprocessing = False

                    if len(training) - len(nan_times) - skip < self.min_training_data:
                        warnings.warn(f"Warning: Non NaN training data less than {self.min_training_data}. No testing performed. Output will be NaN only!\n")
                        perform_detection_after_preprocessing = False

                    if perform_detection_after_preprocessing:
                        # check if the data that needs to be skipped is anyway skipped due to training starting not at beginning of series and remove those from skip
                        first_train_iloc = self.data.index.get_loc(training[0])
                        skip = max(skip - first_train_iloc, 0)

                        result[name] = detector(processed_data, training[skip:], test, arguments)
                    else:
                        result[name] = np.nan
                        #type = p[0]
                        #if type == 'A':
                        #    width = int(p[1:])
                        #    processed_data = processed_data.rolling(width, min_periods=1, center=True, win_type=None, on=None, axis=0, closed=None, step=None, method='single').mean()
                        #if type == 'S':
                        #    n_shift = int(p[1:])
                        #    processed_data = processed_data - processed_data.shift(n_shift)
                        #name += p
                    #name += "]"
                    #result[name] = detector(processed_data, training, test, arguments)
                else:
                    result[name] = detector(self.data, training, test, arguments)
            else:
                result[name] = np.nan

            self.preprocessor_returns.update({name : preprocessor_return_list})

        return result
    

    def AutomaticallySelectDetectors(self, sigma_STD = 4, deviation_PRE = 0.1, periods_necessary_for_season = 3, average_periods_necessary = 10, detector_window_length = 1, threshold_R2_seasonality = 0.05):
        """
        Automatically selects a set of detectors fit for the stored time series.  

        Parameters
        ----------
        sigma_STD : float
            Threshold for STD based detectors. Outliers occur when the deviation from the training data mean is at least sigma_STD standard deviations. 
        deviation_PRE : float
            Threshold for PRE based detectors. Outlier will be detected if observed value deviates at least deviation_PRE * 100 % from the previously seen range. 
        periods_necessary_for_season : int
            At least this many full periods need to be present in the time series in order to apply seasonality modeling. 
            Must be at least 2 for statsmodels routines to work. 
        detector_window_length : int
            Length of the intended testing window w.r.t. time series steps. (E.g. 10 for testing the last 10 points of the series.)
        threshold_R2_seasonality : float
            If the R2 value of a seasonal model is at least this value, then the seasonaliy is included in the detector list
        average_periods_necessary : int
            This many full average periods need to be present in the data to add a detector with averaging. 
        """

        self.ClearDetectors()

        # find most often occurring time difference in nanoseconds and store in time_diff_ns
        time_differences_ns = np.diff(self.data.index).astype(int)
        unique, counts = np.unique(time_differences_ns, return_counts=True)
        dic = dict(zip(unique, counts))
        max_value = - 1
        time_diff_ns = 0
        for key, value in dic.items():
            if value > max_value:
                max_value = value
                time_diff_ns = key


        time_diff_min = time_diff_ns / 1000 / 1000 / 1000 / 60

        #time_diff_hours = time_diff_min / 60

        #print(f"Time difference: {time_diff_min} minutes")

        #min_per_hour = 60
        #min_per_day = 24 * 60
        #min_per_week = 24 * 60 * 7
        #min_per_month = 24 * 60 * 30.5
        #min_per_year = 24 * 60 * 365

        num_data = len(self.data)


        # Is base series stationary?

        counter = 0

        for x in self.data:
            _, _, _, counter_dict_series = self.pp_difference_until_stationary(self.data[x])
            counter = max(counter, counter_dict_series['counter'])

        if counter > 0:
            stationary = False
        else: stationary = True
    
        seasonality_candidates = [60, 24*60, 24*60*7, 24*60*365]

        seasonality_periods_to_test = []
        for s in seasonality_candidates:
            period = s / time_diff_min
            if period.is_integer() and period > 1 and num_data >= periods_necessary_for_season * period:
                seasonality_periods_to_test.append(int(period))


        average_candidates = [60, 24*60, 24*60*7, 24*60*365]

        average_periods = []
        for ac in average_candidates:
            if ac > 2 * time_diff_min and num_data * time_diff_min >= average_periods_necessary * ac:
                average_periods.append(int(ac / time_diff_min))
                break
                
        #if stationary:
            # Standard detectors   
        self.AddDetector(['MD', [1], [], sigma_STD])
        self.AddDetector(['MD', [1], [['difference_until_stationary', [0, 0.05]]], sigma_STD])
        self.AddDetector(['MD', [1], [['ARIMA_subtract', ['pacf', 'stat', -3]]], sigma_STD])
        for a in average_periods:
            self.AddDetector(['MD', [1], [['average', [a]]], sigma_STD])
        #else:
        #    self.AddDetector(['MD', [1], [['difference_until_stationary', [0, 0.05]]], 4])
        #    self.AddDetector(['MD', [1], [['ARIMA_subtract', ['pacf', 'stat', -3]]], 4])
    





    

    def GetOutlierTypes(self, preprocessor):
        """
        Create a descriptive string of the outlier type that a detector with the given preprocessor directives would detect. 

        Parameters
        ----------
        preprocessor : list 
            preprocessor directives for a detector

        Returns
        -------
        answer : string
            description of outlier type to be used further in creating messages describing the outlier type. 
        """

        density = False
        average_period = -1
        seasonal = False
        seasonal_period = -1
        restrict = False
        restrict_period = 0
        ARIMA = False

        for p in preprocessor:
            type = p[0]
            if type == 'season_subtract':
                seasonal = True
                seasonal_period = max(seasonal_period, p[1][0])
            if type == 'average':
                density = True
                average_period = max(average_period, p[1][0])
            if type == 'restrict_data_to':
                restrict = True
                restrict_period += p[1][0]
            if type == 'ARIMA_subtract':
                ARIMA = True
                ARIMA_param = f"[{p[1][0]},{p[1][1]},{p[1][2]}]"

        answer = ""

        if seasonal:
            answer += f" seasonal ({seasonal_period})"
        if density:
            answer += f" density ({average_period})"
        if ARIMA:
            answer += f" ARIMA ({ARIMA_param})"
        answer += " outlier"
        if restrict:
            answer += f" with comparison window restricted to last {restrict_period} datapoints"

        answer = answer.lstrip()

        answer = answer[0].upper() + answer[1:]
        #if answer[0].islower():
        #    answer[0] = answer[0].capitalize()

        return answer


    def IsOutlierCluster(self, previous_outliers, dropoff = 0.1, cluster_density_threshold = 3.0):
        """
        Check whether a given 

        Parameters
        ----------
        previous_outliers : list of int (0 or 1)
            0 or 1 encoding of whether the previous points were outliers. End of list is neighboring the currently tested point that has to be an outlier.  
        dropoff : float
            exponential dampening factor. outliers with distance i away from the currently tested point contribute exponentially less to the overall score as += np.exp( - i * dropoff)
        cluster_density_threshold : float
            returns true if the summed conributions of all previous outliers in the list are at least this value. 

        Returns
        -------
        answer : bool
            True if there is an outlier cluster, False otherwise. 
        """

        if not previous_outliers:
            return False
        
        # weigh outliers closer to the last one exponentially more
        previous_outliers.reverse() # so that closer outliers are weighted more wiht the weighting below

        score = 0
        i=0
        for x in previous_outliers:
            score += x * np.exp( - i * dropoff)
            i += 1

        #print(f"Previous score: {score}")

        return score >= cluster_density_threshold
    
    def GetDetectorNames(self):
        """
        Get descriptive names of all stored detectors. 

        Returns
        -------
        names : list of strings
            A list of the full detector names including parameters and preprocessor directives in order in which they are stored. 
        """

        names = []
        for detector in self.detectors:
            
            type = detector[4]
            ID = detector[5]
            threshold = detector[3]
            preprocessor = detector[2]
            args = detector[1]

            full_name = f"D{ID}: " + type + "_" + "".join(str(e) for e in args) + "_" + "".join(str(e) for e in preprocessor) + "_" + str(threshold)
            names.append(full_name)
        return names


    def InterpretPointScore(self, scores, previous_outliers = []):
        """
        Interprets the outlier scores of the set of detectors stored detectors. 

        Parameters
        ----------
        scores : pd.Dataframe 
            Has to contain exactly one row. 
            Column names are 'D_1', 'D_2', ... labelling the detectors w.r.t. to the detector list. 
            Values are the return values of those detectors. 

        previous_outliers : list of int
            0 or 1 encoding of whether the previous points were outliers. 
            End of list is neighboring the currently tested point that has to be an outlier.
            If empty, no cluster testing is performed. 
            

        Returns
        -------
        isOutlier : bool
            Is the last point an outlier?
        max_level : float 
            overall anomaly score
        message_detail : list
            List of strings containing supplementary information about the anomaly in case isOutlier == True
        detector_responses : json
            Json containing the detailed detector parameters and responses 
        """

        message_detail = []

        isOutlier = False

        detector_responses = []

        max_level_MD = 0.0

        for detector in self.detectors:

            # response structure for each detector: [value, algorithm_name, algorithm_parameters, isOutlier]

            current_response = []

            isOutlierCurrent = False

            type = detector[4]
            ID = detector[5]
            threshold = detector[3]
            preprocessor = detector[2]
            args = detector[1]

            column = "D_" + str(ID)

            max_lag = 0

            #print(f"Processing detector {column}. Return list: {self.preprocessor_returns[column]}")


            deactivate_detector_for_score = False
            for elist in self.preprocessor_returns[column]:
                
                for e in elist:
                    #print(e)
                    if e['pp_type'] == 'ARIMA_subtract':
                        max_lag = max(max(e['pdq']), max_lag)
                        #print(max_lag)
                        if sum(previous_outliers[-max_lag:]) > 0:
                            deactivate_detector_for_score = True

            if deactivate_detector_for_score:
                val = np.nan
                #print("detector deactivated")
            else:
                val = scores[column]

                if type == "MD":
                    max_level_MD = max(max_level_MD, np.abs(val))
                

                #full_name = type + "_" + "".join(str(e) for e in args) + "_" + "".join(str(e) for e in preprocessor) + "_" + str(threshold)

        
                if type == 'MD':
                    val = np.abs(val)
                    if val >= threshold:
                        isOutlier = True
                        isOutlierCurrent = True
                        m = []
                        types = self.GetOutlierTypes(preprocessor)
                        m.append(f"{types} detected: {val:.1f} sigma. ")
                        message_detail.append(' '.join(m))
            
                if type == 'IF':
                    val = np.abs(val)
                    if val == 1:
                        types = self.GetOutlierTypes(preprocessor)
                        message_detail.append(f"{types} detected with isolation forest. ")
                        isOutlier = True
                        isOutlierCurrent = True


            current_response.append(val)
            current_response.append(type)
            current_response.append([args, preprocessor, threshold])
            current_response.append(isOutlierCurrent)
            # The structure of result.responses should be as follows:
            # [
            #   {
            #     "Value": 0.1, # numeric score from the 
            #     "Algorithm": "Prophet",
            #     "Detail": {
            #       "type": "15"
            #     },
            #     "Anomaly": true
            #   },   

            detail_dic = {"Arguments": args, "Preprocesor": preprocessor, "Threshold": threshold}

            current_response_dic ={ 
                "Value": val, 
                "Algorithm": type, 
                "Detail": detail_dic, 
                "Anomaly": isOutlierCurrent
            } 

            detector_responses.append(current_response_dic)



        if isOutlier:
            if self.IsOutlierCluster(previous_outliers):

                time_diff_ns = self.GetTimeStep()

                steps = len(previous_outliers)+1

                outlier_window_size_str = str(timedelta(microseconds = (steps) * time_diff_ns / 1000 ))

                message_detail.append(f'The outlier appears to be part of an outlier cluster. (Tested in a window of size {outlier_window_size_str} (= {steps} time steps) ending here.)')


        # preliminary anomaly strength counter. 
        max_level = max_level_MD

        return isOutlier, max_level, message_detail, json.dumps(detector_responses, indent = 4)
    

    def GetTimeStep(self):
        """
        Extract the time step in the stored time series. 

        In case of multiple occuring time steps, the most frequent one is returend.

        Returns
        -------
        time_diff_ns : int
            Most frequent time difference in the series index in nanoseconds 
        """

        time_differences_ns = np.diff(self.series.index).astype(int)
        unique, counts = np.unique(time_differences_ns, return_counts=True)
        dic = dict(zip(unique, counts))
        max = - 1
        time_diff_ns = 0
        for key, value in dic.items():
            if value > max:
                max = value
                time_diff_ns = key
        return time_diff_ns


    


      






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

        


        
        