
"""
univariate.py
====================================
Contains the definition of the UnivariateOutlierDetection class. 
"""

# general
import numpy as np 
import pandas as pd 
import json

from statsmodels.tsa.seasonal import MSTL



import warnings
warnings.filterwarnings(action='ignore',category=FutureWarning)

def process_last_point(ts,ts_dates):
    """
    Simplest one-line call, takes time series values and dates as lists.
    """
    ts_dates = pd.to_datetime(ts_dates)
    ts_panda = pd.Series(index = ts_dates, data = ts)
    OD = UnivariateOutlierDetection(ts_panda)
    OD.AutomaticallySelectDetectors(detector_window_length=1)
    last_point_scores = OD.LastOutlierScore().iloc[0,:]
    result = OD.InterpretPointScore(last_point_scores)
    return result



class UnivariateOutlierDetection:
    """
    Univariate outlier detection class. 
    """

    # imported outlier detection methods
    from .univariate_STD import STD
    from .univariate_IF import IF
    from .univariate_PRO import PRO
    from .univariate_PRE import PRE
    from .univariate_preprocessors import pp_average, pp_power, pp_median, pp_volatility, pp_difference, pp_season_subtract, pp_fillna_linear, pp_get_resid, pp_get_trend, pp_get_trend_plus_resid, pp_skip_from_beginning, pp_restrict_data_to
    

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

        self.min_training_data = 5

        nans = series.isna() 
        nan_times = nans[nans==True].index.to_list()

        if len(nan_times) > 0:
            warnings.warn(f"Warning: passed series contains {len(nan_times)} nans:\n{nan_times}\n")

        self.SetStandardDetectors()


    # Generally has poor performance as compared to manual season_subtract
    def CalculateMSTL(self, periods = [7, 365]):

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
        self.series : stored time series
            pd.Series stored. 
        """
        return self.series
    

    def GetTrend(self):
        return self.trend
    
    def GetResid(self):
        return self.resid
    
    def GetResidPlusTrend(self):
        return self.resid + self.trend
    

    def AddDetector(self, new_detector):
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
        self.AddDetector(['STD', [1], [], 5])
        self.AddDetector(['PRE', [10], [], 0.05])
        self.AddDetector(['IF', [0.05], [], None])

    
    
    def LastOutlierScore(self, window = None):
        if window == None:
            window = self.series.index.to_list()
        #print(window)
        training = window[:-1]
        #print(training[-1])
        test = window[-1:]
        #print(test)
        return self.WindowOutlierScore(training, test)
    

    def GetOutlierTypes(self, preprocessor):

        density = False
        average_period = -1
        seasonal = False
        seasonal_period = -1
        restrict = False
        restrict_period = 0

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


        answer = ""

        if seasonal:
            answer += f" seasonal ({seasonal_period})"
        if density:
            answer += f" density ({average_period})"
        answer += " outlier"
        if restrict:
            answer += f" with comparison window restricted to last {restrict_period} datapoints"

        answer = answer[1:].capitalize()

        return answer

    def InterpretPointScore(self, scores):
        message_detail = []

        isOutlier = False

        type_seasonal = False
        type_density = False

        detector_responses = []



        high_level_description = ""

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

            
            val = scores[column]

            if type == "STD":
                max_level_STD = max(max_level_STD, np.abs(val))
            if type == "PRE":
                max_level_PRE = max(max_level_PRE, np.abs(val))

            full_name = type + "_" + "".join(str(e) for e in args) + "_" + "".join(str(e) for e in preprocessor) + "_" + str(threshold)

            #print(f"Checking detector {full_name}")


            if type == 'PRE':
                val = np.abs(val)
                types = self.GetOutlierTypes(preprocessor)
                if val >= 1.0 + threshold:
                    
                    message_detail.append(f"{types}: a value this extreme was never seen before. It deviates by at least {((val-1.0)*100):.0f}% from the previously seen value range. ")
                    isOutlier = True
                    isOutlierCurrent = True
                if val == 1:
                    message_detail.append(f"{types}: a similar value has never been observed before, but it is within the previously observed data range. ")
                    isOutlier = True
                    isOutlierCurrent = True

            if type == 'STD':
                #print(val)
                val = np.abs(val)
                if val >= threshold:
                    isOutlier = True
                    isOutlierCurrent = True
                    m = []
                    types = self.GetOutlierTypes(preprocessor)
                    m.append(f"{types} detected: {val:.1f} sigma. ")
                    #val2 = np.abs(scores['PRE[10][A10]'])
                    #if val2 < 1:
                    #    m.append(f'But a similar contextual outlier has been seen before in {((1.0 - val2)*100):.0f}% of measurements.')
                    message_detail.append(' '.join(m))
        
            if type == 'PRO':
                val = np.abs(val)
                if val >= threshold:
                    message_detail.append(f"Contextual outlier (via Prophet) detected with strength {val:.1f}. ")
                    isOutlier = True
                    isOutlierCurrent = True

            if isOutlierCurrent:
                for p in preprocessor:
                    if p[0] == 'season_subtract':
                        type_seasonal = True
                    if p[0] == 'average':
                        type_density = True


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

        if type_seasonal and type_density:
            high_level_description = "Seasonal / density outlier"
        elif type_seasonal:
            high_level_description = "Seasonal outlier"
        elif type_density:
            high_level_description = "Density outlier"
        elif isOutlier:
            high_level_description = "Range outlier"

        # preliminary anomaly strength counter. 
        max_level = max((max_level_PRE - 1.0) * 5.0, max_level_STD)

        return isOutlier, max_level, message_detail, json.dumps(detector_responses, indent = 4) 

#        val = np.abs(scores['STD[1][V20M20]'])
#        if val >= thresh_STD_1_V20M20:
#            isOutlier=True
#            m = []
#            m.append(f"Median volatility outlier of strength {val:.1f} detected.")
#            val2 = np.abs(scores['PRE[10][V20M20]'])
#            if val2 < 1:
#                m.append(f'But a similar average volatility outlier has been seen before in {((1.0 - val2)*100):.0f}% of measurements.')
#            message.append(' '.join(m))

#        val = np.abs(scores['PRE[10][V20M20]'])
#        if val >= thresh_PRE_10_V20M20:
#            isOutlier=True
#            message.append(f"Median volatility exceeds previous range by {((val-1.0)*100):.0f}%.")

    

#        val = np.abs(scores['STD[1][s1440]'])
#        if val >= thresh_STD_1_s1440:
#            m = []
#            isOutlier=True
#            m.append(f'Contextual outlier of {val:.1f} sigma detected.')
#            val2 = np.abs(scores['PRE[10][s1440]'])
#            if val2 < 1:
#                m.append(f'But a similar contextual outlier has been seen before in {((1.0 - val2)*100):.0f}% of measurements.')
#            message.append(' '.join(m))

#        val = np.abs(scores['PRE[10][s1440]'])
#        if val >= thresh_PRE_10_s1440:
#            message.append(f"Contextual outlier detected that deviates at least {((val-1.0)*100):.0f}% from the previously seen value range.")
#            isOutlier = True
            

        
        

    def WindowOutlierScore(self, training, test):

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
            
            if perform_detection:
                if preprocessor:
                    perform_detection_after_preprocessing = True
                    processed_series = self.series.copy()
                    skip = 0 # we skip this many from the beginning of the series in the training data to avoid boundary effects
                    for p in preprocessor:
                        pp_type = p[0]
                        pp_args = p[1]
                        pp_func = getattr(self, 'pp_' + pp_type)
                        processed_series, critical_error, add_skip = pp_func(processed_series, pp_args)

                        skip += add_skip
                        if critical_error:
                            perform_detection_after_preprocessing = False

                    if len(training) - len(nan_times) - skip < self.min_training_data:
                        warnings.warn(f"Warning: Non NaN training data less than {self.min_training_data}. No testing performed. Output will be NaN only!\n")
                        perform_detection_after_preprocessing = False

                    if perform_detection_after_preprocessing:
                        result[name] = detector(processed_series, training[skip:], test, arguments)
                    else:
                        result[name] = np.nan
                else:
                    result[name] = detector(self.series, training, test, arguments)
            else:
                result[name] = np.nan

        return result


    def IsSeasonalitySignificant(self, period, average_period, threshold_R2 = 0.1, type = 'multiplicative'):

        R2 = self.SeasonalityR2(period = period, average_period = average_period, type = type)

        return np.mean(R2) >= threshold_R2


    def SeasonalityR2(self, period, average_period, type = 'multiplicative'):

        nans=[]
        if self.series.isnull().values.any():
            nans = np.where(self.series.isnull())

        deseasoned, _, _ = self.pp_season_subtract(self.series, [period, average_period, type])

        if nans:
            deseasoned.iloc[nans] = np.nan

        variance_average_period = period

        var_signal = self.series.rolling(variance_average_period, min_periods=1, center=False, win_type=None, on=None, axis=0, closed=None, step=None, method='single').var()
        var_deseasoned = deseasoned.rolling(variance_average_period, min_periods=1, center=False, win_type=None, on=None, axis=0, closed=None, step=None, method='single').var()
        R2 = 1 - var_deseasoned / var_signal

        #print(f"Seasonality R2 improvement: {np.mean(R2.dropna())}")

        return np.mean(R2.dropna())                        

        

    def AutomaticallySelectDetectors(self, sigma_STD = 4, deviation_PRE = 0.05, periods_necessary_for_average = 3, detector_window_length = 1):
        self.ClearDetectors()
        # find most often occurring time difference in nanoseconds and store in time_diff_ns
        time_differences_ns = np.diff(self.series.index).astype(int)
        unique, counts = np.unique(time_differences_ns, return_counts=True)
        dic = dict(zip(unique, counts))
        max = - 1
        time_diff_ns = 0
        for key, value in dic.items():
            if value > max:
                max = value
                time_diff_ns = key


        time_diff_min = time_diff_ns / 1000 / 1000 / 1000 / 60

        time_diff_hours = time_diff_min / 60

        #print(f"Time difference: {time_diff_min} minutes")

        min_per_hour = 60

        min_per_day = 24 * 60

        min_per_week = 24 * 60 * 7

        min_per_month = 24 * 60 * 7 * 30.5

        min_per_year = 24 * 60 * 365

        num_data = len(self.series)

        average_length = []

        season = None


        if time_diff_min < 60:
            # average over hour
            if num_data >= periods_necessary_for_average * min_per_hour / time_diff_min:
                average_length.append(int(min_per_hour / time_diff_min))
                season = int(min_per_hour / time_diff_min)
            # average over day
            if num_data >= periods_necessary_for_average * min_per_day / time_diff_min:
                #average_length.append(int(min_per_day / time_diff_min))
                season = int(min_per_day / time_diff_min)
            # average over week
            if num_data >= periods_necessary_for_average * min_per_week / time_diff_min:
                #average_length.append(int(min_per_week / time_diff_min))
                season = int(min_per_week / time_diff_min)

        # OSKAR type hourly data
        if time_diff_min == 60:
            # average over day
            if num_data >= periods_necessary_for_average * min_per_day / time_diff_min:
                average_length.append(int(min_per_day / time_diff_min))
                season = int(min_per_day / time_diff_min)
            # average over week
            if num_data >= periods_necessary_for_average * min_per_week / time_diff_min:
                #average_length.append(int(min_per_week / time_diff_min))
                season = int(min_per_week / time_diff_min)
            # average over month
            if num_data >= periods_necessary_for_average * min_per_month / time_diff_min:
                #average_length.append(int(min_per_month / time_diff_min))
                season = int(min_per_month / time_diff_min)

        # INVTEREST type daily data
        if time_diff_min == 24 * 60:
            # average over week
            if num_data >= periods_necessary_for_average * min_per_week / time_diff_min:
                average_length.append(int(min_per_week / time_diff_min))
                season = int(min_per_week / time_diff_min)
            # average over month
            #if num_data >= periods_necessary_for_average * min_per_month / time_diff_min:
                #average_length.append(int(min_per_month / time_diff_min))
                #season = int(min_per_month / time_diff_min)
            # average over year
            if num_data >= periods_necessary_for_average * min_per_year / time_diff_min:
                #average_length.append(int(min_per_year / time_diff_min))
                season = int(min_per_year / time_diff_min)
                #self.AddDetector(['STD', [1], [['get_trend_plus_resid', []]], sigma_STD])
                #self.AddDetector(['PRE', [10], [['get_trend_plus_resid', []]], 1.0 + deviation_PRE])

            

        # decompose strategy from subday to dayly

        # from daily to weekly / monthly / yearly depending on period lengh

        # try differences with a given time series of first doing weekly then yearly, or just yearly. 

        # Maybe downsample to month first, then do yearly seasonality subtraction 

        # Standard detectors
        self.AddDetector(['STD', [1], [], sigma_STD])
        self.AddDetector(['PRE', [10], [], 1.0 + deviation_PRE])

        # Add average detectors:

        for a in average_length:
            self.AddDetector(['STD', [1], [['average', [a]]], sigma_STD])
            self.AddDetector(['PRE', [10], [['average', [a]]], 1.0 + deviation_PRE])


        average_period = int(season/10)

        has_negatives = self.series.lt(0).any()

        if has_negatives:
            seasonality_type = 'additive'
        else:
            R2_additive = self.SeasonalityR2(period = season, average_period = average_period, type = 'additive')
            R2_multiplicative = self.SeasonalityR2(period = season, average_period = average_period, type = 'multiplicative')
            if R2_multiplicative > R2_additive:
                seasonality_type = 'multiplicative'
            else: 
                seasonality_type = 'additive'


        if not season == None:

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
            
                    

            


    def PrintDetectors(self):
        for d in self.detectors:
            print(d) 

    def GetDetectorNames(self):
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
