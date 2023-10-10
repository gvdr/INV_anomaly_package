
"""
univariate.py
====================================
Contains the definition of the UnivariateOutlierDetection class. 
"""

#__version__ = '0.1.0'

# general
import numpy as np 
import pandas as pd 
import json
from datetime import datetime, timedelta

from statsmodels.tsa.seasonal import MSTL



import warnings
warnings.filterwarnings(action='ignore',category=FutureWarning)

def process_last_point(ts, ts_dates, sigma_STD = 4, deviation_PRE = 0.1, periods_necessary_for_season = 3, average_periods_necessary = 10, detector_window_length = 1, threshold_R2_seasonality = 0.05):
    """
    One-line call for last point.

    Simplest one-line call, takes time series values and dates as lists.
    Training data is the whole time series except the last point. 
    Outlier scores are evaluated on the last point only. 

    Parameters
    ----------
    ts : list (or similar object parsable by pandas) of floats 
        Values of the time series. 
    ts_dates : list (or similar object parsable by pandas) of time stamps, need to be parsable by pd.to_datetime()
        Time stamps of the time series. 
        Needs to have the same length as ts

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
    ts_dates = pd.to_datetime(ts_dates)
    ts_panda = pd.Series(index = ts_dates, data = ts)
    OD = UnivariateOutlierDetection(ts_panda)
    OD.AutomaticallySelectDetectors(sigma_STD = sigma_STD, deviation_PRE = deviation_PRE, periods_necessary_for_season = periods_necessary_for_season, average_periods_necessary = average_periods_necessary, detector_window_length = detector_window_length, threshold_R2_seasonality = threshold_R2_seasonality)
    last_point_scores = OD.LastOutlierScore().iloc[0,:]
    result = OD.InterpretPointScore(last_point_scores)
    return result


def process_last_point_with_window(ts, ts_dates, window_size=10, skip_from_beginning = 0, sigma_STD = 4, deviation_PRE = 0.1, periods_necessary_for_season = 3, average_periods_necessary = 10, detector_window_length = 1, threshold_R2_seasonality = 0.05):
    """
    One-line call for last point including cluster analysis. 
    
    Evaluteas and interprets outlier score of last point and checks whether this belongs to a cluster of outliers of window_size. 

    Parameters
    ----------
    ts : list (or similar object parsable by pandas) of floats 
        Values of the time series. 
    ts_dates : list (or similar object parsable by pandas) of time stamps, need to be parsable by pd.to_datetime()
        Time stamps of the time series. 
        Needs to have the same length as ts
    window_size : int
        Size of the window (test data range) relative to the time step in the time series. 
    skip_from_beginning : int
        Skip this many points from the beginning of the time series before training data range starts. 

    Returns
    -------
    isOutlier : bool
        Is the last point an outlier?
    max_level : float 
        overall anomaly score
    message_detail : list
        List of strings containing supplementary information about the anomaly in case isOutlier == True
    detector_responses : json
        Json containing the detailed detector parameters and respondes 
    """
    
    ts_dates = pd.to_datetime(ts_dates)
    ts_panda = pd.Series(index = ts_dates[skip_from_beginning:], data = ts[skip_from_beginning:])
    OD = UnivariateOutlierDetection(ts_panda)    

    lenght_data = len(ts_dates) - skip_from_beginning
    length_past = lenght_data - window_size
    length_future = window_size
    past = ts_dates[skip_from_beginning:(skip_from_beginning+length_past)]
    future = ts_dates[(skip_from_beginning+length_past):]

    OD.AutomaticallySelectDetectors(sigma_STD = sigma_STD, deviation_PRE = deviation_PRE, periods_necessary_for_season = periods_necessary_for_season, average_periods_necessary = average_periods_necessary, detector_window_length = length_future, threshold_R2_seasonality = threshold_R2_seasonality)

    #OD.PrintDetectors()

    # Get outlier true/false for previous points
    scores = OD.WindowOutlierScore(past, future)

    final_score = np.repeat(0, length_future - 1)
    for k in range(length_future - 1):
        isOutlier, _, _, _ = OD.InterpretPointScore(scores.loc[future[k], :])
        if isOutlier:
            final_score[k] = 1
    #final_score = pd.Series(index=future, data=final_score)

    result = OD.InterpretPointScore(scores.loc[future[-1], :], previous_outliers = final_score.tolist())

    return result




class UnivariateOutlierDetection:
    """
    Univariate outlier detection class. 
    """

    # imported outlier detection methods
    from .univariate_STD import STD
    from .univariate_IF import IF
    #from .univariate_PRO import PRO
    from .univariate_PRE import PRE
    from .univariate_preprocessors import pp_average, pp_power, pp_median, pp_volatility, pp_difference, pp_season_subtract, pp_fillna_linear, pp_get_resid, pp_get_trend, pp_get_trend_plus_resid, pp_skip_from_beginning, pp_restrict_data_to, pp_ARIMA_subtract, pp_difference_until_stationary
    

    # series has to have an index in pandas datetime format
    def __init__(self, series):
        """
        Initializes a univariate outlier detector with a time series. 

        Parameters
        ----------
        series : pd.Series
            Pandas Series containing the time series. 
            Index has to be a pd.DatetimeIndex. 
            Values are float and may be np.nan. 
            Duplicate indices are removed, first are kept. 
        """

        self.min_training_data = 5

        if not isinstance(series, pd.Series):
            raise TypeError("Passed series is not a pd.Series.")
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("Index of passed series is not a pd.DatetimeIndex.")
        if not pd.api.types.is_numeric_dtype(series):
            raise TypeError("Passed series is not numeric.")
        
        # Check for duplicate indices:
        duplicated_indices = series.index.duplicated()
        if duplicated_indices.sum() > 0:
            warnings.warn(f"Warning: passed series contains duplicated indices. Removing duplicates, keeping firsts only.\n")
            series = series[~duplicated_indices]
            
        self.series = series  

        self.trend = None
        self.resid = None

        self.preprocessor_returns = dict()

        nans = series.isna() 
        nan_times = nans[nans==True].index.to_list()

        if len(nan_times) > 0:
            warnings.warn(f"Warning: passed series contains {len(nan_times)} nans:\n{nan_times}\n")

        self.ClearDetectors()
        self.SetStandardDetectors()


    # Generally has poor performance as compared to manual season_subtract
    def CalculateMSTL(self, periods = [7, 365]):
        """
        Calculates Season-Trend decomposition using LOESS for multiple seasonalities using MSTL from statsmodels.

        Results are stored in self.trend and self.resid.
        Tends to result in false positives for outliers, pp_season_subtract function tends to be more reliable, use this instead for now. 
        """

        nans = np.where(self.series.isnull())

        imputed_series, _, _ = self.pp_fillna_linear(self.series)
        
        res = MSTL(imputed_series, periods=(365), lmbda=0).fit()

        self.trend = res.trend
        self.resid = res.resid

        if nans:
            self.trend.iloc[nans] = np.nan
            self.resid.iloc[nans] = np.nan





    def GetSeries(self):
        """
        Returns the stored time series. 

        Returns
        -------
        self.series : pd.Series of floats with datetime index
            stored time series
        """
        return self.series
    

    def GetTrend(self):
        """
        Returns the trend of the stored time series after CalculateMSTL() has been called. 

        Returns
        -------
        self.trend : pd.Series of floats with datetime index
            stored time series trend
        """
        return self.trend
    
    def GetResid(self):
        """
        Returns the residual of the stored time series after CalculateMSTL() has been called. 

        Returns
        -------
        self.resid : pd.Series of floats with datetime index
            stored time series residuals
        """
        return self.resid
    
    def GetResidPlusTrend(self):
        """
        Returns the trend plus residual of the stored time series after CalculateMSTL() has been called. 

        This is the analogue of using season_subtract on the time series as the missing component is the seasonality. 

        Returns
        -------
        self.resid + self.trend : pd.Series of floats with datetime index
            stored time series trend + residual
        """
        return self.resid + self.trend
    

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
        """
        Deletes the list of detectors. At least one detector needs to be added afterwards before the outlier detection can be used. 
        """
        self.detectors = []
        self.num_detectors = 0


    def SetStandardDetectors(self):
        """
        Selects a set of standard detectors applicable to generic time series. Better call AutomaticallySelectDetectors() instead for detectors customised for the stored time series.
        """
        self.ClearDetectors()
        self.AddDetector(['STD', [1], [], 5])
        self.AddDetector(['PRE', [10], [], 0.05])
        self.AddDetector(['IF', [0.05], [], None])

    
    
    def LastOutlierScore(self, window = None):
        """
        Short hand call of WindowOutlierScore() that takes all of the stored series (or a window) except the last point as training data and check the last point for being an outlier. No checking for outlier clustering. 

        Parameters
        ----------
        window : list of datetime objects matching parts of the stored series' index
            if not set to None, training + test is taken from this window instead of the whole stored series. 
        """
        if window == None:
            window = self.series.index.to_list()
        #print(window)
        training = window[:-1]
        #print(training[-1])
        test = window[-1:]
        #print(test)
        return self.WindowOutlierScore(training, test)
    

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

        max_level_STD = 0.0
        max_level_PRE = 0.0

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

            #print(f"Processing detector {column}. Return list: {self.preprocessor_returns[column]}")


            deactivate_detector_for_score = False
            for e in self.preprocessor_returns[column]:
                if e['pp_type'] == 'ARIMA_subtract':
                    max_lag = max(e['pdq'])
                    #print(max_lag)
                    if sum(previous_outliers[-max_lag:]) > 0:
                        deactivate_detector_for_score = True

            if deactivate_detector_for_score:
                val = np.nan
                #print("detector deactivated")
            else:
                val = scores[column]

                if type == "STD":
                    max_level_STD = max(max_level_STD, np.abs(val))
                if type == "PRE":
                    max_level_PRE = max(max_level_PRE, np.abs(val))

                #full_name = type + "_" + "".join(str(e) for e in args) + "_" + "".join(str(e) for e in preprocessor) + "_" + str(threshold)

                if type == 'PRE':
                    val = np.abs(val)
                    types = self.GetOutlierTypes(preprocessor)
                    if val >= threshold:
                        
                        message_detail.append(f"{types}: a value this extreme was never seen before. It deviates by at least {((val-1.0)*100):.0f}% from the previously seen value range. ")
                        isOutlier = True
                        isOutlierCurrent = True
                    if val == 1:
                        message_detail.append(f"{types}: a similar value has never been observed before, but it is within the previously observed data range. ")
                        isOutlier = True
                        isOutlierCurrent = True

                if type == 'STD':
                    val = np.abs(val)
                    if val >= threshold:
                        isOutlier = True
                        isOutlierCurrent = True
                        m = []
                        types = self.GetOutlierTypes(preprocessor)
                        m.append(f"{types} detected: {val:.1f} sigma. ")
                        message_detail.append(' '.join(m))
            
                if type == 'PRO':
                    val = np.abs(val)
                    if val >= threshold:
                        message_detail.append(f"Contextual outlier (via Prophet) detected with strength {val:.1f}. ")
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
        max_level = max((max_level_PRE - 1.0) * 5.0, max_level_STD)

        return isOutlier, max_level, message_detail, json.dumps(detector_responses, indent = 4) 



        

    def WindowOutlierScore(self, training, test):
        """
        Calculates the outlier scores given a training and test window. 

        Parameters
        ----------
        training : list of datetime indices
            Datetime indices of a subset of the time series to be used for training the detectors. 

        test : list of datetime indices
            Datetime indices of a subset of the time series to be used for calculating the outlier scores. 
            

        Returns
        -------
        result : pd.DataFrame
            Dataframe containing as column names the detectors ('D_1', 'D_2', ...) and as row indices the datetime indices of the test list.
            Values are the outlier scores of the detectors. 
        """

        perform_detection = True

        nans = self.series[training].isna() 
        nan_times = nans[nans==True].index.to_list()
        if len(nan_times) > 0:
            warnings.warn(f"Warning: training data contains {len(nan_times)} out of {len(training)} NaNs:\n{nan_times}\n")
        if len(nan_times) == len(training):
            warnings.warn(f"Warning: No valid training data! Output will be NaN only!\n")
            perform_detection = False
        if len(training) - len(nan_times) < self.min_training_data:
            warnings.warn(f"Warning: Non NaN training data less than {self.min_training_data}. No testing performed. Output will be NaN only!\n")
            perform_detection = False


        nans = self.series[test].isna() 
        nan_times = nans[nans==True].index.to_list()
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
                    processed_series = self.series.copy()
                    skip = 0 # we skip this many from the beginning of the series in the training data to avoid boundary effects
                    for p in preprocessor:
                        pp_type = p[0]
                        pp_args = p[1]
                        pp_func = getattr(self, 'pp_' + pp_type)
                        processed_series, critical_error, add_skip, pp_return = pp_func(processed_series, pp_args)

                        if pp_return:
                            preprocessor_return_list.append(pp_return)

                        skip += add_skip
                        if critical_error:
                            perform_detection_after_preprocessing = False

                    if len(training) - len(nan_times) - skip < self.min_training_data:
                        warnings.warn(f"Warning: Non NaN training data less than {self.min_training_data}. No testing performed. Output will be NaN only!\n")
                        perform_detection_after_preprocessing = False

                    if perform_detection_after_preprocessing:
                        # check if the data that needs to be skipped is anyway skipped due to training starting not at beginning of series and remove those from skip
                        first_train_iloc = self.series.index.get_loc(training[0])
                        skip = max(skip - first_train_iloc, 0)

                        result[name] = detector(processed_series, training[skip:], test, arguments)
                    else:
                        result[name] = np.nan
                else:
                    result[name] = detector(self.series, training, test, arguments)
            else:
                result[name] = np.nan

            self.preprocessor_returns.update({name : preprocessor_return_list})
            #print(self.preprocessor_returns)


        return result



    def IsSeasonalitySignificant(self, period, average_period, threshold_R2 = 0.1, type = 'multiplicative'):
        """
        Checks whether a given seasonality is significanlty present in the stores seriess. 

        Parameters
        ----------
        period : int
            Length of the period tested for w.r.t. time series steps
        average_period : int
            Averaging period passed to pp_season_subtract. Seasonality is smoothed up this scale, i.e. neglected below the scale. 
        threshold_R2 : float
            Threshold of the R_square statistic above which seasonality is significant (return True)
        type : str
            Type of the seasonality. Either 'multiplicative' or 'additive'. 
            

        Returns
        -------
        result : bool
            Is the seasonality significant (yes = True / no = False)?
        """


        R2 = self.SeasonalityR2(period = period, average_period = average_period, type = type)

        return np.mean(R2) >= threshold_R2


    def SeasonalityR2(self, period, average_period, type = 'multiplicative'):
        """
        Computes the R_square statistic of the improvement a given seasonality yields in modelling the stored series. 

        Parameters
        ----------
        period : int
            Length of the period tested for w.r.t. time series steps
        average_period : int
            Averaging period passed to pp_season_subtract. Seasonality is smoothed up this scale, i.e. neglected below the scale. 
        type : str
            Type of the seasonality. Either 'multiplicative' or 'additive'. 
            

        Returns
        -------
        R2 : float
            R2 statistic of the seasonality model
        """

        nans=[]
        if self.series.isnull().values.any():
            nans = np.where(self.series.isnull())

        deseasoned, _, _, _ = self.pp_season_subtract(self.series, [period, average_period, type])

        if nans:
            deseasoned.iloc[nans] = np.nan

        variance_average_period = period

        var_signal = self.series.rolling(variance_average_period, min_periods=1, center=False, win_type=None, on=None, axis=0, closed=None, step=None, method='single').var()
        var_deseasoned = deseasoned.rolling(variance_average_period, min_periods=1, center=False, win_type=None, on=None, axis=0, closed=None, step=None, method='single').var()

        var_signal = self.series.dropna().var()
        var_deseasoned = deseasoned.dropna().var()

        R2 = 1 - var_deseasoned / var_signal

        #print(f"Seasonality R2 improvement: {np.mean(R2.dropna())}")
        return R2   
        #return np.mean(R2.dropna())                        

        

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
        time_differences_ns = np.diff(self.series.index).astype(int)
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

        num_data = len(self.series)


        # Is base series stationary?

        _, _, _, counter_dict = self.pp_difference_until_stationary(self.series)
        if counter_dict['counter'] > 0:
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

        if stationary:
            # Standard detectors
            self.AddDetector(['STD', [1], [], sigma_STD])
            self.AddDetector(['PRE', [10], [], 1.0 + deviation_PRE])
            for a in average_periods:
                self.AddDetector(['STD', [1], [['average', [a]]], sigma_STD])
                self.AddDetector(['PRE', [10], [['average', [a]]], 1.0 + deviation_PRE])




        # Add average detectors:
        



        # Set up seasonality decomposition

        # We average the computed seasonality of period period over average_period so that too fine grained noise is removed. 
        # we may later in addition subtract seasonality of a finer scale to compensate. 

        seasonality_preprocessors = [] 

        has_negatives = self.series.le(0).any()

        for p in seasonality_periods_to_test:
            average_period = 1
            if has_negatives:
                seasonality_type = 'additive'
                R2_additive = self.SeasonalityR2(period = p, average_period = average_period, type = 'additive')
                print(f'R2_additive tested for p={p}: {R2_additive}')
                if R2_additive > threshold_R2_seasonality:
                    new_seasonality_preprocessor = ['season_subtract', [p, average_period, seasonality_type]]
                    seasonality_preprocessors.append(new_seasonality_preprocessor)
            else:
                R2_additive = self.SeasonalityR2(period = p, average_period = average_period, type = 'additive')
                print(f'R2_additive tested for p={p}: {R2_additive}')
                R2_multiplicative = self.SeasonalityR2(period = p, average_period = average_period, type = 'multiplicative')
                print(f'R2_multiplicative tested for p={p}: {R2_multiplicative}')
                if R2_additive > R2_multiplicative:
                    seasonality_type = 'additive'
                else:
                    seasonality_type = 'multiplicative'
                if max(R2_additive, R2_multiplicative) > threshold_R2_seasonality:
                    new_seasonality_preprocessor = ['season_subtract', [p, average_period, seasonality_type]]
                    seasonality_preprocessors.append(new_seasonality_preprocessor)


        for preprocessor in seasonality_preprocessors:
            self.AddDetector(['STD', [1], [preprocessor], sigma_STD])
            self.AddDetector(['PRE', [10], [preprocessor], 1.0 + deviation_PRE])
            self.AddDetector(['STD', [1], [preprocessor, ['ARIMA_subtract', ['pacf', 'stat', -3]]], sigma_STD - 1])

        if not seasonality_preprocessors:
            self.AddDetector(['STD', [1], [['ARIMA_subtract', ['pacf', 'stat', -3]]], sigma_STD * 0.75])




        season = None
        if not season == None and False:

            if season == 365:
                average_period = 30 # remove spikes that may be spurious in the INVTEREST data. averaging however procudes bad results e.g. with Numenta data. 
                if self.IsSeasonalitySignificant(period = season, average_period = average_period, threshold_R2 = 0.1, type = seasonality_type):
                    self.AddDetector(['STD', [1], [['season_subtract', [season, average_period, seasonality_type]]], sigma_STD])
                    self.AddDetector(['PRE', [10], [['season_subtract', [season, average_period, seasonality_type]]], 1.0 + deviation_PRE])
                    self.AddDetector(['STD', [1], [['season_subtract', [7, 1, seasonality_type]], ['season_subtract', [365, 30, seasonality_type]]], sigma_STD])
                    self.AddDetector(['STD', [1], [['season_subtract', [7, 1, seasonality_type]], ['season_subtract', [365, 30, seasonality_type]], ['restrict_data_to', [365, detector_window_length]]], sigma_STD])
                elif self.IsSeasonalitySignificant(period = 7, average_period = 1, threshold_R2 = 0.2, type = seasonality_type):
                    self.AddDetector(['STD', [1], [['season_subtract', [7, 1, seasonality_type]]], sigma_STD])

                self.AddDetector(['STD', [1], [['restrict_data_to', [365, detector_window_length]]], sigma_STD])

            else:
                average_period = 1
                self.AddDetector(['STD', [1], [['season_subtract', [season, average_period, seasonality_type]]], sigma_STD])
                self.AddDetector(['PRE', [10], [['season_subtract', [season, average_period, seasonality_type]]], 1.0 + deviation_PRE])
                # For weekly seasonality, we additionally look at the restriction of training data to the last month
                if season == 168 and time_diff_min == 60:
                    self.AddDetector(['STD', [1], [['season_subtract', [season, average_period, seasonality_type]], ['restrict_data_to', [168 * 4, detector_window_length]]], sigma_STD])
                    # Additional detector that subtracts seasonality fist with weekly period, then daily period  
                    # This detector gave similar results to the standard [168,1] subtract, so we leave it out for now. 
                    #self.AddDetector(['STD', [1], [['season_subtract', [168, 24, seasonality_type]], ['season_subtract', [24, 1, seasonality_type]]], sigma_STD])

                self.AddDetector(['STD', [1], [['season_subtract', [season, average_period, seasonality_type]], ['ARIMA_subtract', ['pacf', 'stat', -3]]], sigma_STD])
                


            


    def PrintDetectors(self):
        """
        Prints all detectors the standard output stream. 
        """

        for d in self.detectors:
            print(d) 

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
