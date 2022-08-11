import measures
import pandas as pd
import threading, time
import statsmodels.api as sm
import numpy as np
np.warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8.0, 6.0)

from PyNomaly import loop
from sklearn.cluster import DBSCAN
from benchmark_hyperopt import BenchmarkHyperopter
from benchmark_hyperopt2 import BenchmarkHyperopter2
from model_selection import prediction
from seasonality_trend import SeasonalityDetector
from sys import maxsize 

class Forecast:
    
    def __init__(self, timeseries, factor, steps, evals, ensemble_metric):
        self.timeseries = timeseries
        self.model_dict = {}
        self.trend_seasonal_dict = {}
        self.data_dict = {}
        self.predicted_dict = {}
        self.data_original = np.ones(steps)
        self.kernel_rmse = int(maxsize)
        self.steps = steps
        self.factor = factor
        self.maxevals = evals
        self.ensemble_metric = ensemble_metric
        
    def outlier(self):
        """
        NOTE: 
        - Appropriately handle missing values prior to using LoOP
        - LoOP does not support Pandas DataFrames or Numpy arrays with missing values
        - Any observations with missing values will be returned with empty outlier scores (nan) in the final result
        """
        print("You are here detecting outliers from the dataset")
        # LOOP_DBSCAN Cluster Assignment
        self.timeseries = np.array(self.timeseries['timeseries'])
        db = DBSCAN(eps=0.6, min_samples=50).fit(self.timeseries.reshape(len(self.timeseries),1))
        m = loop.LocalOutlierProbability(self.timeseries, extent=0.95, n_neighbors=10, cluster_labels = db.labels_).fit()
        scores = m.local_outlier_probabilities

        # Detecting outliers
        critical_score = 0.8
        self.timeseries = [np.nan if values > critical_score else self.timeseries[key] for key, values in enumerate(scores)]

        # Reload Time Series Dataset
        self.timeseries = pd.DataFrame(self.timeseries)
        self.timeseries.columns = ['timeseries']
        
                
    def dataset(self, t, s, periodicity):
        """
        t: filtering method to decompose trend component. Eg: hp => hp_filter
        s: filtering method to decompose seasonality component
        """
        print("You are here in forecasting module preparing datasets for different models")
        index = self.timeseries.index
        obj_sesnl = SeasonalityDetector(self.timeseries, periodicity)
        obj_sesnl.trend_seasonal_comp()
        self.trend_seasonal_dict = obj_sesnl.trend_seasonal_dict
        
        # Original Series updated to data dictionary
        new_series = self.timeseries
        series_length,tp99 = new_series.shape
        self.data_dict.update({'original':[new_series[:int(self.factor*series_length)],
                                           new_series[int(self.factor*series_length):]]} )
        
        # New Series with seasonality removed and updated to data dictionary
        new_series = np.array(self.timeseries).flatten()-np.array(self.trend_seasonal_dict[s][1]).flatten()
        new_series = pd.DataFrame(new_series)
        new_series.columns = ['timeseries']
        new_series.index = index
        series_length,tp99 = new_series.shape
        self.data_dict.update({'seasonal_removed':[new_series[:int(self.factor*series_length)],
                                                  new_series[int(self.factor*series_length):]]} )                                                                                                    

        # New Series with trend and seasonality removed and updated to data dictionary
        new_series = np.array(self.timeseries).flatten()-np.array(self.trend_seasonal_dict[s][1]).flatten()-                               np.array(self.trend_seasonal_dict[t][0]).flatten()
        new_series = pd.DataFrame(new_series)
        new_series.columns = ['timeseries']
        new_series.index = index
        series_length,tp99 = new_series.shape
        self.data_dict.update({'both_removed':[new_series[:int(self.factor*series_length)],
                                               new_series[int(self.factor*series_length):]]} )
        
        # New Series with trend removed and updated to data dictionary
        new_series = np.array(self.timeseries).flatten()-np.array(self.trend_seasonal_dict[s][0]).flatten()
        new_series = pd.DataFrame(new_series)
        new_series.columns = ['timeseries']
        new_series.index = index
        series_length,tp99 = new_series.shape
        self.data_dict.update({'trend_removed':[new_series[:int(self.factor*series_length)],
                                                  new_series[int(self.factor*series_length):]]} )
        
    
    def parameter_tuning(self, periodicity):
        print("You are in forecasting module functionality parameter tuning part")
        # BENCH KERNEL 1 
        obj_parameter_tuning = BenchmarkHyperopter(self.steps, self.maxevals)    
        a = time.time()
        
        self.model_dict.update(obj_parameter_tuning.garch(self.data_dict['trend_removed'][0]))    
         
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning.arima(self.data_dict['seasonal_removed'][0]))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning.garch(self.data_dict['trend_removed'][0]))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning.egarch(self.data_dict['both_removed'][0]))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning.egarchm(self.data_dict['both_removed'][0]))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning.segarch(self.data_dict['both_removed'][0]))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning.segarchm(self.data_dict['both_removed'][0]))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning.gas(self.data_dict['seasonal_removed'][0]))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning.gasllev(self.data_dict['seasonal_removed'][0]))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning.gasllt(self.data_dict['seasonal_removed'][0]))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning.llev(self.data_dict['seasonal_removed'][0]))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning.llt(self.data_dict['seasonal_removed'][0]))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning.nllev(self.timeseries))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning.nllt(self.timeseries))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning.lmegarch(self.timeseries))).start()
        b = time.time()
        print("Time for tuning models of BENCH_HYPEROPT_1: ", round((b-a)/60, 2), " minutes")
        
        # BENCH KERNELS 2
        obj_parameter_tuning2 = BenchmarkHyperopter2(self.steps, self.maxevals, periodicity)
        a = time.time()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning2.wma(self.data_dict['original'][0]))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning2.naive(self.data_dict['original'][0]))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning2.drift(self.data_dict['original'][0]))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning2.linear(self.data_dict['original'][0]))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning2.additive(self.data_dict['original'][0]))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning2.multiplicative(self.data_dict['original'][0]))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning2.lin_decay(self.data_dict['original'][0]))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning2.exp_decay(self.data_dict['original'][0]))).start()
        # threading.Thread(target=self.model_dict.update(obj_parameter_tuning2.gaussian_decay(self.data_dict['original'][0]))).start()
        b = time.time()
        print("Time for tuning models of BENCH_HYPEROPT_2: ", round((b-a)/60, 2), " minutes")


    def select_model(self, periodicity):
        print("You are in forecasting module functionlaity select_model part")
#         print('Select your error metric from RMSE/MAPE/MFE/MAE/MFPE/SSE/SMSE/RMSPE/RMSLE/UNLR:')
#         metric = input()
        metric = 'rmse'
#         print('Select MAX/MIN:')
#         operation = input()
        operation = 'MIN'
        obj_select_model = prediction(metric, operation, self.steps, periodicity)
        for name in self.model_dict.keys():
#             print('Select your ensemble weight for', name)
#             Wgt = int(input())
            Wgt = 1
            if name == 'arima' or 'gas' or 'gasllev' or 'gasllt' or 'llev' or 'llt' or 'nllev' or 'nllt':
                self.predicted_dict.update(obj_select_model.predict(name, self.model_dict[name],     
                                                                    self.data_dict['seasonal_removed'], Wgt))
            elif name == 'garch'or 'egarch'or 'egarchm' or 'lmegarch'or'segarch' or'segarchm':
                self.predicted_dict.update(obj_select_model.predict(name, self.model_dict[name],                                                                                                               self.data_dict['both_removed'],Wgt))
            elif name == 'wma'or 'naive'or 'drift' or 'linear'or'additive' or'multiplicative':
                self.predicted_dict.update(obj_select_model.predict(name, self.model_dict[name],                                                                                                               self.data_dict['original'],Wgt))
            elif name == 'garch':
                self.predicted_dict.update(obj_select_model.predict(name, self.model_dict[name],                                                                                                               self.data_dict['trend_removed'],Wgt))
        
        self.predicted_dict.update(obj_select_model.ensemble(self.data_dict['original'][1], self.ensemble_metric) )
                
        print('Best accuracy_measure:' + str(obj_select_model.Algo) + '   Value:'+ str(obj_select_model.bestmetricvalue))
        print('ensemble accuracy_measure model_wise: ' + str(self.predicted_dict['ensemble'][1]))
        ERR = self.predicted_dict['ensemble'][2]
        print('ensemble accuracy_measure overall: ' + str(ERR))
        ERR_1 = self.predicted_dict['ensemble'][3]
        print('ensemble accuracy_measure tdp_(t+5:t+7): ' + str(ERR_1))
        ERR_2 = self.predicted_dict['ensemble'][4]
        print('ensemble accuracy_measure tdp_(t+8:t+10): ' + str(ERR_2))
        ensembled_pred_df = pd.DataFrame(self.predicted_dict['ensemble'][5])

        return obj_select_model.model, obj_select_model.Algo, ERR, ERR_1, ERR_2, ensembled_pred_df