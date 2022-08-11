import pandas as pd
import numpy as np
import statsmodels.api as sm

from sys import maxsize
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from scipy.optimize import fmin_l_bfgs_b
from math import exp, isnan, sqrt
from kernels import Benchmarking
from estimator import estimator


def RMSE(params, *args):
    """
    Helper Function for holt winters algorithms
    """
    Y = args[0]
    type = args[1]
    rmse = 0

    if type == 'linear':
        alpha, beta = params
        a = [Y[0]]
        b = [Y[1] - Y[0]]
        y = [a[0] + b[0]]

        for i in range(len(Y)):
            a = np.append(a, alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
            b = np.append(b, beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            y = np.append(y, a[i + 1] + b[i + 1])
    else:
        alpha, beta, gamma = params
        m = args[2]     
        a = [sum(Y[0:m]) / float(m)]
        b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]

        if type == 'additive':
            s = [Y[i] - a[0] for i in range(m)]
            y = [a[0] + b[0] + s[0]]
            for i in range(len(Y)):
                a = np.append(a, alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
                b = np.append(b, beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
                s = np.append(s, gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
                y = np.append(y, a[i + 1] + b[i + 1] + s[i + 1])
        elif type == 'multiplicative':
            s = [Y[i] / a[0] for i in range(m)]
            y = [(a[0] + b[0]) * s[0]]
            for i in range(len(Y)):
                a = np.append(a, alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
                b = np.append(b, beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
                s = np.append(s, gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])
                y = np.append(y, (a[i + 1] + b[i + 1]) * s[i + 1])
        else:
            exit('Type must be either linear, additive or multiplicative')

    rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y, y[:-1])]) / len(Y))

    return rmse



class BenchmarkHyperopter2:
    """
    Class contains benchmarking algorithms which will be used to
    measure the effectiveness of forecasting methods implemented
    in the modeling module
    """

    def __init__(self, steps, evals, periodicity):
        
        self.aic = int(maxsize)
        self.modelbest = None
        self.order = 0
        self.steps = steps
        self.maxevals = evals
        self.periodicity = periodicity
        
    
    def wma(self, timeseries):
        """
        The function implements hyper-parameter tuning of Weighted Moving Average (for given training series)
        
        Returns
        --------------------------------------------
        Dictionary {algorithm name: [best_model_prediction(steps), aic value, model parameters_dict]}
        
        """
        self.aic = int(maxsize)
        self.modelbest= None
        avg_order = 4 
        bm = Benchmarking(pd.DataFrame(timeseries), self.steps)
        predict, aic = bm.weighted_moving_average(avg_order)
        if isnan(aic) or predict.isna().sum().sum()>0:
            return {'loss':int(maxsize), 'status':STATUS_OK}
        else:
            self.aic = aic
            params = {'avg_order':avg_order}
            print( 'wma,  AIC:' +str(self.aic) + ' Averaging order '+ str(avg_order) )
            return { 'wma' : [predict , {'AIC': aic, 'RMSE': None}, params] }
    
    
    def naive(self, timeseries):
        """
        The function implements hyper-parameter tuning of Naive method (for given training series)
        
        Returns
        --------------------------------------------
        Dictionary {algorithm name: [best_model_prediction(50), aic value, model parameters_dict]}
        
        """
        self.aic = int(maxsize)
        self.modelbest= None
        bm = Benchmarking(pd.DataFrame(timeseries), self.steps)
        predict, aic = bm.naive()
        if isnan(aic) or np.isnan(predict).sum().sum()>0:
            return {'loss':int(maxsize), 'status':STATUS_OK}
        else:
            self.aic = aic
            self.order = 1
        params = {}
        rmse = None
        print( 'naive,  AIC:' +str(self.aic) )
        return { 'naive' :[predict, rmse, aic, params] }

    
    def drift(self, timeseries):
        """
        The function implements hyper-parameter tuning of Linear Holt Winter's method (for given training series)
        
        Returns
        --------------------------------------------
        Dictionary {algorithm name: [best_model_prediction(50), aic value, model parameters_dict]}
        
        """
        self.aic = int(maxsize)
        bm = Benchmarking(timeseries, self.steps)
        predict, aic = bm.drift()
        if isnan(aic) or predict.isna().sum().sum()>0:
            return {'loss':int(maxsize), 'status':STATUS_OK}
        else:
            if aic < self.aic:
                self.aic = aic
        params = {}
        rmse = None
        print( 'drift,  AIC:' +str(self.aic) )
        return { 'drift' : [predict , rmse, aic, params] }
              
     
    def linear(self, timeseries, fc=None, alpha = None, beta = None):
        """
        Linear Holt Winters Method

        Parameters
        --------------------------------------------------------
        fc : The number of periods the time series is to be
                forecasted
        alpha : data smoothening factor for linear holt winters
        beta : trend smoothening factor for linear holt winters
        
        Returns
        --------------------------------------------
        Dictionary {algorithm name: [best_model_prediction, aic/rmse values, model parameters_dict]}

        """
        fc = self.steps
        Y = np.array(timeseries).flatten().reshape(len(timeseries),1 )
        if (alpha == None or beta == None):
            initial_values = np.array([0.3, 0.1])
            boundaries = [(0, 1), (0, 1)]
            type = 'linear'
            parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (Y, type), bounds = boundaries, 
                                       approx_grad = True)
            alpha, beta = parameters[0]
        a = [Y[0]]
        b = [Y[1] - Y[0]]
        y = [a[0] + b[0]]
        rmse = 0

        for i in range(len(Y) + fc):
            if i == len(Y):
                tmp = np.array([a[-1] + b[-1]])
                Y = np.append(Y, tmp)
            a = np.append(a, alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
            b = np.append(b, beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            y = np.append(y, a[i + 1] + b[i + 1])
        
        rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))
        predict = pd.DataFrame(Y[-fc:])
        obj = estimator(predict)
        aic = obj.aic()
        params = { 'alpha': parameters[0][0], 'beta': parameters[0][1] }
        
        print( 'linear holt winter,  AIC:' +str(aic) + '   '+ str(params))
        return { 'Linear_Holt_Winter' : [predict, rmse, aic, params] }


    def additive(self, timeseries, fc=None, alpha = None, beta = None, gamma = None):
        """
        Additive Seasonal Holt Winters method

        Parameters
        --------
        fc : The number of periods the time series is to be
                forecasted
        alpha : data smoothening factor for additive holt winters
        beta : trend smoothening factor for additive holt winters
        gamma : seasonal smoothening factor for additive holt winters
        m : period of seasonality

        Reference - https://www.otexts.org/fpp/7/5
        
        """
        m = self.periodicity
        fc = self.steps
        Y = np.array(timeseries).flatten().reshape(len(timeseries),1 )
        if (alpha == None or beta == None or gamma == None):
            initial_values = np.array([0.3, 0.1, 0.1])
            boundaries = [(0, 1), (0, 1), (0, 1)]
            type = 'additive'
            parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (Y, type, m), bounds = boundaries, approx_grad = True)
            alpha, beta, gamma = parameters[0]
        a = [sum(Y[0:m]) / float(m)]
        b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
        s = [Y[i] - a[0] for i in range(m)]
        y = [a[0] + b[0] + s[0]]
        rmse = 0

        for i in range(len(Y) + fc):
            if i == len(Y):
                Y = np.append(Y, pd.Series(a[-1] + b[-1] + s[-m]))
            a = np.append(a, alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
            b = np.append(b, beta * ( a[i + 1] - a[i]) + (1 - beta) * b[i])
            s = np.append(s, gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
            y = np.append(y, a[i + 1] + b[i + 1] + s[i + 1])

        rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))
        predict = pd.DataFrame(Y[-fc:])
        obj = estimator(predict)
        aic = obj.aic()
        params = { 'alpha': parameters[0][0], 'beta': parameters[0][1], 'gamma': parameters[0][2], 'periodicity': m }
        
        print( 'additive holt winter,  AIC:' +str(aic) + '   '+ str(params))
        return { 'Additive_Holt_Winter' : [predict, rmse, aic, params] }


    def multiplicative(self, timeseries, fc=None, alpha = None, beta = None, gamma = None):
        """
        Multiplicative Seasonal Holt Winters method

        Parameters
        --------
        fc : The number of periods the time series is to be
                forecasted
        alpha : data smoothening factor for multiplicative holt winters
        beta : trend smoothening factor for multiplicative holt winters
        gamma : seasonal smoothening factor for multiplicative holt winters
        m : period of seasonality        
        
        Reference - https://www.otexts.org/fpp/7/5
        
        """
        m = self.periodicity
        fc = self.steps
        Y = np.array(timeseries).flatten().reshape(len(timeseries),1 )
        if (alpha == None or beta == None or gamma == None):
            initial_values = np.array([0.0, 1.0, 0.0])
            boundaries = [(0, 1), (0, 1), (0, 1)]
            type = 'multiplicative'
            parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (Y, type, m), bounds = boundaries, approx_grad = True)
            alpha, beta, gamma = parameters[0]
        a = [sum(Y[0:m]) / float(m)]
        b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
        s = [Y[i] / a[0] for i in range(m)]
        y = [(a[0] + b[0]) * s[0]]
        rmse = 0

        for i in range(len(Y) + fc):
            if i == len(Y):
                Y = np.append(Y, pd.Series((a[-1] + b[-1]) * s[-m]))
            a = np.append(a, alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
            b = np.append(b, beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            s = np.append(s, gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])
            y = np.append(y, (a[i + 1] + b[i + 1]) * s[i + 1])

        rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))
        predict = pd.DataFrame(Y[-fc:])
        obj = estimator(predict)
        aic = obj.aic()
        params = { 'alpha': parameters[0][0], 'beta': parameters[0][1], 'gamma': parameters[0][2], 'periodicity': m }
        
        print( 'Multiplicative holt winter,  AIC:' +str(aic) + '   '+ str(params))
        return { 'Multiplicative_Holt_Winter' : [predict, rmse, aic, params] }

    
#      def lin_decay(timeseries, window = 3, steps=None):
#         """
#         inputs:
#         -------------------------------------------------------------------
#         timeseries = series of time series data inputs
#         window = window size. should be always odd for a gaussian window
#         steps = how many time points to predict
#         degree = used in exponential kernel, allows the user to choose the degree of the exponential curve
#         std = standard deviation of the gaussian window used in the gaussain kernel. default set to 3

#         outputs:
#         --------------------------------------------------------------------
#         y1= original series with the predicted outputs appended to it
#         predicted= predicted outputs for the time periods specified in pred

#         """
#         steps = self.steps
#         tmp = pd.DataFrame([0.1])
#         tmp.columns = ['timeseries']
#         timeseries = timeseries.append(tmp, ignore_index = False).reset_index(drop=True)
#         target_lag_1 = np.array(timeseries.shift(1)).flatten()

#         window = window + 1
#         sum_window = sum(x for x in range(window))
#         weight_list = []

#         for i in range(1, window):
#             weight_list =np.append(weight_list, float(i)/sum_window)
#         reversed_weight_list = list(reversed(weight_list))

#         y1 = pd.DataFrame(convolve(target_lag_1, reversed_weight_list, 'valid'))
#         y1.columns = ['timeseries']
#         timeseries = timeseries.drop([len(timeseries) - 1])

#         s2 = timeseries
#         s2 = s2.append(y1.tail(1)).reset_index(drop=True)
# #         s2['timeseries'][int(len(s2))] = y1.tail(1)
#         for i in range(1, steps):
#             s3 = s2.shift(1)
#             y2 = pd.DataFrame(convolve(s2['timeseries'], reversed_weight_list, 'valid'))
#             print("y2 ", y2.tail(3))
#             s2 = s2.append(y2.tail(1)).reset_index(drop=True)
#             s3 = s3.append(y2.tail(1)).reset_index(drop=True)

#         predict = pd.DataFrame(s2['timeseries'].tail(steps))
#         obj = estimator(predict)
# #         aic = obj.aic()
#         rmse = None
# #         rmse = accuracy_measure(actual, predicted)
#         params = { 'degree': None, 'std': 3 }
           
#         print( 'linear decay,  AIC: +str(aic)' + '   '+ str(params))
#         return { 'Linear Decay' : [predict[-50:], rmse, aic, params]}


#     def lin_decay(self, window=5, steps=None):
#         """
#         inputs:
#         -------------------------------------------------------------------
#         timeseries = series of time series data inputs
#         window = window size. should be always odd for a gaussian window
#         pred = how many time points to predict
#         degree = used in exponential kernel, allows the user to choose the degree of the exponential curve
#         std = standard deviation of the gaussian window used in the gaussain kernel. default set to 3

#         outputs:
#         --------------------------------------------------------------------
#         y1= original series with the predicted outputs appended to it
#         predicted= predicted outputs for the time periods specified in pred
        
#         """
#         steps = self.steps
#         timeseries = self.timeseries.append(Series([0.1])).reset_index(drop=True)
#         target_lag_1 = timeseries.shift(1)
#         window = window + 1
#         sum_window = sum(x for x in range(window))
#         weight_list = []
#         for i in range(1, window):
#             weight_list =np.append(weight_list, float(i) / sum_window)
#         reversed_weight_list = list(reversed(weight_list))
#         y = convolve(target_lag_1, reversed_weight_list, 'valid')
#         y1 = DataFrame(y)

#         timeseries = timeseries.drop([len(timeseries) - 1])
#         s2 = self.timeseries.append(y1.tail(1)).reset_index(drop=True)

#         for i in range(1, steps):
#             s3 = s2.shift(1)
#             y2 = convolve(s2[0], reversed_weight_list, 'valid')
#             y3 = DataFrame(y2)
#             s2 = np.append(s2, y3.tail(1)).reset_index(drop=True)
#             s3 = np.append(s3, y3.tail(1)).reset_index(drop=True)
#             y1 = np.append(y1, y3.tail(1)).reset_index(drop=True)
#         predict = pd.DataFrame(s2[0].tail(steps))
        
#         rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))
#         obj = estimator(predict)
#         aic = obj.aic()
#         params = { 'alpha': parameters[0][0], 'beta': parameters[0][1], 'gamma': parameters[0][2], 'periodicity': m }
        
#         print( 'linear decay,  AIC:' +str(aic) + '   '+ str(params))
#         return { 'Linear Decay' : [predict, rmse, aic, params] }
#         return y1, predict


#     def exp_decay(self, degree=2, window=5, steps=None):
#         steps = self.steps
#         timeseries = self.timeseries.append(Series([0.1])).reset_index(drop=True)
#         target_lag_1 = timeseries.shift(1)
#         window = window+1
#         weight_list = []
#         sq = []
#         sq = [i**degree for i in range(window)]
#         sum_window = sum(sq)

#         for i in range(1, len(sq)):
#             weight_list.append(float(sq[i]) / sum_window)
#         reversed_weight_list = list(reversed(weight_list))
#         y = convolve(target_lag_1, reversed_weight_list, 'valid')
#         y1 = DataFrame(y)

#         timeseries = timeseries.drop([len(timeseries) - 1])
#         s2 = timeseries.append(y1.tail(1)).reset_index(drop=True)

#         for i in range(1, steps):
#             y2 = convolve(s2[0], reversed_weight_list, 'valid')
#             y3 = DataFrame(y2)
#             s2 = s2.append(y3.tail(1)).reset_index(drop=True)
#             y1 = y1.append(y3.tail(1)).reset_index(drop=True)
#         predicted = s2[0].tail(steps)
#         return y1, predicted


#     def gaussian_decay(self, window=5, steps=None, std=3):
        
#         steps = self.steps
#         timeseries = self.timeseries.append(Series([0.1])).reset_index(drop=True)
#         target_lag_1 = timeseries.shift(1)

#         gauss_window = signal.gaussian(window, std=std)

#         semi_gaussian = range(int(len(gauss_window) / 2 + 0.5))
#         weight_list = []
#         for i in semi_gaussian:

#             weight_list.append(gauss_window[i])
#         sum_weights = sum(weight_list)
#         norm_weights = []
#         for i in range(len(weight_list)):
#             norm_weights.append(weight_list[i] / sum_weights)
#         reversed_weight_list = list(reversed(norm_weights))

#         y = convolve(target_lag_1, reversed_weight_list, 'valid')
#         y1 = DataFrame(y)

#         timeseries = timeseries.drop([len(timeseries)-1])
#         s2 = timeseries.append(y1.tail(1)).reset_index(drop=True)

#         for i in range(1, steps):
#             s3 = s2.shift(1)
#             y2 = convolve(s2[0], reversed_weight_list, 'valid')
#             y3 = DataFrame(y2)
#             s2 = s2.append(y3.tail(1)).reset_index(drop=True)
#             s3 = s3.append(y3.tail(1)).reset_index(drop=True)
#             y1 = y1.append(y3.tail(1)).reset_index(drop=True)
#         predicted = s2[0].tail(steps)
#         return y1, predicted
