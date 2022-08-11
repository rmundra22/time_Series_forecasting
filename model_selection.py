import measures
import sys
import pandas as pd
import numpy as np

from seasonality_trend import SeasonalityDetector


class prediction:
    """
    Class for selection between best models of different algorithms  
    Uses Root Mean Square Error(RMSE) for predicted data of Cross Validation Data
  
    Input
    --------------------------------------------
    + Trainining timeseries, Cross validation time series for the corresponding algorithm
    + Name, model of the corresponding algorithms 
    
    Returns
    --------------------------------------------        
    An object that stores the variables of best rmse model from the models being called with the predict fn()
    1).Algo - algorithm name (string)
    2).bestmetricvalue - error metic value (float)
    3).model - model(model)
    4).predicted  - Predicted values corresponding to testseries (DF)
    """
    
    # Initializing the values
    def __init__(self, metric, operation, steps, periodicity):
        self.Algo = None        
        self.bestmetricvalue = 0
        self.model = None
        self.predicted = None
        self.metric = metric.lower()
        self.operation = operation.lower()
        self.steps = steps
        self.periodicity = periodicity
        self.max_or_min()
        self.weights = {}
        self.sum_weights = 0
        self.ensembledseries = np.ones(steps)
        self.trend_seasonal_dict = {}
        
    def max_or_min(self):
        if self.operation == 'max':
            self.bestmetricvalue = -int(sys.maxsize)
        else:
            self.bestmetricvalue = int(sys.maxsize)
        
    def accuracy_measure(self,actual,prediction):
        if self.metric == 'rmse':
            return measures.rmse(actual,prediction)
        elif self.metric == 'mape':
            return measures.mape(actual, prediction)
        elif self.metric == 'mfe':
            return measures.mfe(actual, prediction)
        elif self.metric == 'mae':
            return measures.mae(actual, prediction)
        elif self.metric == 'mfpe':
            return measures.mfpe(actual, prediction)
        elif self.metric == 'sse':
            return measures.sse(actual, prediction)
        elif self.metric == 'smse':
            return measures.smse(actual, prediction)
        elif self.metric == 'rmspe':
            return measures.rmspe(actual, prediction)
        elif self.metric =='rmsle':
            return measures.rmsle(actual, prediction)
        elif self.metric == 'unlr':
            ERR_models, ERR, ERR_1, ERR_2 = measures.unlr(actual, prediction)
            return ERR_models, ERR, ERR_1, ERR_2

    # Undifferences the predicted data if differenced values are produced from fit model
    def undifference(self,trainseries,predictedseries, order):
        
        if order == 0:
            return predictedseries
        else:
            difflast = []
            series = np.array(trainseries).flatten()
            for i in range(order):
                if i==0:
                    difflast.append(series[-1])
                else:
                    series = np.diff(series)
                    difflast.append(series[-1])
            difflast = np.array(difflast).flatten()
            predictedarray = np.array(predictedseries).flatten()
            for k in range(order):
                for p in range(predictedarray.size):
                    if p==0:
                        predictedarray[p] += difflast[order-1-k]
                    else:
                        predictedarray[p] += predictedarray[p-1]
            predictedarray = predictedarray.flatten()
            predictedseries = pd.DataFrame(predictedarray)
            predictedseries.columns = ['timeseries']
            return predictedseries
    

    def component_add(self, trainseries, predictedseries, degree = 2):
        """
        COMPONENT ADDITION TO PREDICTED SERIES
        degree(int): degree of time(t) against y for polynomial regression fitting

        """
        # Predicted Series after addition of forecasted seasonality component
        obj = SeasonalityDetector(trainseries, self.periodicity)
        obj.trend_seasonal_comp()
        cf_cycle = obj.trend_seasonal_dict['cf'][1]
        cycle = cf_cycle[:self.periodicity]

        # Forecasted trend component; z: list of parameters for time variables, t, t^2,......, t^n
        y = np.array(trainseries-cf_cycle).flatten()
        t = np.linspace(1, len(y), len(y))
        z = np.array(np.polyfit(t, y, degree))
        z = z.reshape(len(z), 1)

        # Adding back forecasted trend component
        l = len(y) + len(predictedseries) - 1
        t_pred = np.linspace( len(y)+1, l, len(predictedseries) )
        bkt = np.array(t_pred**0)
        for i in range(1,len(z)):
            tmp = np.array(t_pred**i)
            bkt = np.column_stack((tmp, bkt))
        model = np.dot(bkt,z)
        print("This is trend components ", model)
        predictedseries = pd.DataFrame(np.array(predictedseries).flatten() + model.flatten())
        predictedseries.columns = ['timeseries']
#         print("predicted series after trend component addition", predictedseries)

        # Adding back forecasted seasonality component
        for i in range(len(model)):
            position = len(trainseries) + i + 1
            predictedseries.values[i] = predictedseries.values[i] + cycle.values[position%self.periodicity]
        predictedseries = pd.DataFrame(predictedseries)
        print("predicted series after trend and seasonal component addition", predictedseries)
        predictedseries.columns = ['timeseries']

        return predictedseries


    # Selects the best model of predicted data
    def predict(self, name, model,data_list, Wgt):
        print("model name ", name)
        predictedseries = None
        if (str(name) not in ['wma', 'naive', 'drift', 'Linear_Holt_Winter', 'Additive_Holt_Winter', 'Multiplicative_Holt_Winter']):
            predictedseries = model[0].predict(h = len(data_list[1]))
            predictedseries = self.undifference(data_list[0], predictedseries, model[1])
#             print("\n Predicted series before component addition \n", predictedseries)
            predictedseries = self.component_add(data_list[0], predictedseries)
            predictedseries = pd.DataFrame(predictedseries)
        else:
            predictedseries = pd.DataFrame(model[0])
#         print("\n Predicted Series finally after all additions \n", predictedseries)
        self.weights.update({name: Wgt})
        self.sum_weights = int(self.sum_weights) + int(Wgt)
        self.ensembledseries = np.column_stack(( self.ensembledseries, np.array(predictedseries) ))
        print("predictedseries before ensembling ",predictedseries)
        tmp = self.accuracy_measure(data_list[1], predictedseries)
#         print("\n", name, " accuracy for best model ", tmp)
        if self.operation == 'max':
            if tmp > self.bestmetricvalue:
                self.bestmetricvalue =  tmp
                self.Algo = name
                self.model = model
                self.predicted = predictedseries
        else:
            if tmp < self.bestmetricvalue:
                self.bestmetricvalue =  tmp
                self.Algo = name
                self.model = model
                self.predicted = predictedseries
        return {name: [predictedseries, tmp, model]}
    
            
    def ensemble(self, testdata, ensemble_metric):
        print("\n ENSEMBLED SERIES _model_selection\n", self.ensembledseries)
        self.metric = ensemble_metric
        Y = testdata
        Y_hat_ensemble = self.ensembledseries/self.sum_weights
        Y_hat_ensemble = np.delete(Y_hat_ensemble, np.s_[0], axis=1) 
        ERR_models, ERR, ERR_1, ERR_2 = self.accuracy_measure(Y, Y_hat_ensemble)
        ensembled_predictedseries = self.ensembledseries.sum(axis=1)
        return {'ensemble':[Y_hat_ensemble, ERR_models, ERR, ERR_1, ERR_2, ensembled_predictedseries, self.weights]} 