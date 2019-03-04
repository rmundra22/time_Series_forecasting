import pandas as pd
import pyflux as pf
import numpy as np
from sys import exit
from math import sqrt
from scipy import signal
from numpy import convolve, sum
from pandas import DataFrame, Series
from scipy.optimize import fmin_l_bfgs_b
from estimator import estimator

class Benchmarking:
    """
    Class contains benchmarking algorithms which will be used to
    measure the effectiveness of forecasting methods implemented
    in the modeling module
    """

    def __init__(self, timeseries, steps):
        self.timeseries = timeseries
        self.steps = steps
        
        
    def weighted_moving_average(self, N):
        """
        Weighted Moving Average Model
        Here forecast is the weighted average of the previous N terms

        Parameters
        --------
        N : Number of terms which are to be averaged while
            forecasting
        steps : The number of periods the time series is to be
              forecasted
        """
        model = self.timeseries
#         print("\n\nkernels_timeseries", self.timeseries.head(), "\n\n")
        for i in range(self.steps):
            y = sum(np.array(model.tail(N))) * (1/N)
            model = np.array(model).flatten()
            model = np.append(model, y)
            model = pd.DataFrame(model)
        predict = model.tail(self.steps)
        print(predict, len(predict))
        obj = estimator(predict)
        aic = obj.aic()
        return predict, aic


    def naive(self):
        """
        The function returns the last observation as the forecast
        
        Parameters
        --------
        steps : The number of periods the time series is to be
              forecasted
        """
        predict = np.repeat(self.timeseries.tail(1).values,self.steps)
        obj = estimator(predict)
        aic = obj.aic()
        return predict, aic 


    def drift(self):
        """
        This builds forecast using the drift in the data

        Parameters
        --------
        steps : The number of periods the time series is to be
                forecasted
        """
        Y = self.timeseries
        for i in range(self.steps):
            Y_next = Y.iloc[-1] + (Y.iloc[-1] - Y.iloc[0]) / (Y.index.size - 1)
            Y = Y.append(pd.Series(Y_next), ignore_index=True)
        predict = Y.tail(self.steps)
        obj = estimator(predict)
        aic = obj.aic()
        return predict, aic