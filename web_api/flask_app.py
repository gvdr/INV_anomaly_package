from flask import Flask, jsonify, request
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import datetime
import importlib
import matplotlib.gridspec as gridspec
import json
import outlierdetection.univariate as UOD



app = Flask(__name__)

# /process_to_last considers all the time series provided as input until (but not including) the last point
# accepts a json payload of data with the name of the series (`ts_keyword`) and the time series itself (`ts`)
# returns a json payload according to the schema below 
@app.route('/process_to_last', methods=['POST'])
def process_ts():
    ts = request.json['ts']
    ts_keyword = request.json['ts_keyword']
    ts_dates = request.json['ts_dates']

    isAnomaly, anomaly_score, message_detail, result_responses = UOD.process_last_point(ts, ts_dates)

    anomaly_response = jsonify({
         "Anomaly": isAnomaly, # boolean: whether we call an anomaly or not, from the ensemble decision
         "Description": message_detail, # text: any longer description of the anomaly, as aggregated from ensemble method
         "keyword": ts_keyword, # text: name of the ts (for frontend consumption)
         "TimeSeries": ts, # numeric array: observed numeric values the ts (for frontend consumption)
         "TimeSeriesDates": ts_dates,
         "Support": anomaly_score, # numeric value that express how much convinced we are about this anomaly (it is needed for sorting in front end)
         "Raw_Scores": result_responses # nested json: detailed results from the various algorithms used in the ensemble decision
        })
    return anomaly_response

# The structure of request.json will be as follows:
# 
# {
#   "ts_keyword": "facebook_blablabla",
#   "ts": [1.0,1.2,24.2,123.3],
#   "ts_dates": ["2023-09-01", "2023-09-02", "2023-09-03", "2023-09-04"]
# }
# 
#
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
#   {
#     "Value": 0.2,
#     "Algorithm": "Prophet",
#     "Detail": {
#       "type": "15"
#     },
#     "Anomaly": true
#   },
#   {
#     "Value": 0.2,
#     "Algorithm": "Isolation Forest",
#     "Detail": {
#       "type": "15"
#     },
#     "Anomaly": true
#   }
# ]

# /reload reload the library
@app.route('/reload')
def reload():
    importlib.reload(UOD)
    return "outlierdetection reloaded"

if __name__ == '__main__':
    app.run(debug=False)

