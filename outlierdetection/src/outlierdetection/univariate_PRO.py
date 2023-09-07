import numpy as np
import pandas as pd
from prophet import Prophet

# kats
#from kats.consts import TimeSeriesData
#from kats.models.prophet import ProphetModel, ProphetParams

def PRO(self, processed_series, training, test, args = [0.95]):

        interval_width = float(args[0])

        m = Prophet(interval_width = interval_width)

        m.fit(pd.DataFrame(data = {'ds' : training, 'y' : processed_series.loc[training]}))

        future = pd.DataFrame(data = {'ds' : test})
        

        forecast = m.predict(future)
        
        forecast = forecast.set_index('ds')
        

        deviations = (forecast['yhat_upper'] - forecast['yhat_lower']) / 2
        # result is deviation from target divided by the deviation expected from the certainty interval set when calling the function
        results = (processed_series.loc[test] - forecast['yhat'] ) / deviations

        
        return results

    
        # Old code from more general version returning also past forecast and training errors. Not needed for core functionality as the forecast errors are automatically predicted by prophet

        #past = pd.DataFrame(data = {'ds' : training})
        #pastcast = m.predict(past)
        #pastcast = pastcast.set_index('ds')

        #training_deviations = (pastcast['yhat_upper'] - pastcast['yhat_lower']) / 2
        #training_errors = (processed_series.loc[training] - pastcast['yhat'] ) / training_deviations
        
        #return results, forecast['yhat'], forecast['yhat_upper'], forecast['yhat_lower'], training_errors


    
#    # assumes for now consecutive training and test intervals
#    def WindowScoreKatsProphet(self, training, test, interval_width=0.95):

#        tsd_train = TimeSeriesData(pd.DataFrame(data = {'time' : training, 'y' : processed_series.loc[training]}))

#        params = ProphetParams(seasonality_mode='multiplicative')

#        m = ProphetModel(tsd_train, params)
#        m.fit()

#        future = pd.DataFrame(data = {'ds' : test})

#        forecast = m.predict(len(future))

#        forecast = forecast.set_index('time')

#        deviations = (forecast['fcst_upper'] - forecast['fcst_lower']) / 2
        # result is deviation from target divided by the deviation expected from the certainty interval set when calling the function
#        results = (processed_series.loc[test] - forecast['fcst'] ) / deviations       

#        return results
    