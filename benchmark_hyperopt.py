import pandas as pd
import pyflux as pf
import numpy as np
from sys import maxsize
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL
from math import exp, isnan

'''
Parameter grid used by timeseries algorithms
arimaspace: arima
garchspace: garch,egarch,egarchm,segarch,segarchm,lmegarch
gasspace: gas
llspace: gasllev,gasllt,nllt,nllev
llnormalspace: nllev,nllt
'''
# "PML","Laplace","M-H","BBVI" methods can also be used 
# pf.Poisson(), pf.t(),pf.Skewt(),pf.Laplace(),pf.Exponential() families can also be used

arimaspace = {'p': hp.choice('p', range(1,5)),
              'q': hp.choice('q', range(1,5)),
              'd': hp.choice('d', range(0,4)),
              'method':hp.choice('method', ["MLE"])
             }


garchspace = {'p': hp.choice('p', range(1,5)),
              'q': hp.choice('q', range(1,5)),
              'method':hp.choice('method', ["MLE"])
             }

gasspace = {'ar': hp.choice('ar', range(1,5)),
            'sc': hp.choice('sc', range(1,5)),
            'integ': hp.choice('integ', range(0,4)),
            'family':hp.choice('family',[pf.Normal()]),
            'method':hp.choice('method', ["MLE"]) 
           }

llspace = {'integ': hp.choice('integ', range(0,4)),
           'family':hp.choice('family',[pf.Normal()]),
           'method':hp.choice('method', ["MLE"])
          }

llnormalspace = {'integ': hp.choice('integ', range(0,4)),
                 'method':hp.choice('method', ["MLE"])
                }


class benchmark_hyperopt:
    """
    Class for hyperparameter tuning of benchmarking algorithms using hyperopt 
    Fit model Akaike Information Criterion(AIC) is considered in parameter tuning
    
    Available benchmarking algorithms
    -------------------------------------------
    ARIMA - pyflux.arima
    ARCH  - pyflux.garch, pyflux.egarch, pyflux.egarchm, pyflux.segarch. pyflux.segarchm, 
            pyflux.lmegarch
    GAS   - pyflux.gas, pyflux.gasllev, pyflux.gasllt
    GAUSSIAN - pyflux.llev, pyflux.llt, pyflux.nllev, pyflux.nllt
    
    Input
    --------------------------------------------
    Corresponding training timeseries for each algorithm
    
    Returns
    --------------------------------------------        
    An dictionary {algorithm name: [best aic model, integration order, aic value]} for the algorithm(function) called
    1.algorithm name - string
    2.model - pyflux model
    3.integration order - int
    4.aic value - float
    """

    def __init__(self, steps, maxevals, arima_grid= arimaspace, garch_grid = garchspace, gas_grid = gasspace, ll_grid = llspace, llnormal_grid = llnormalspace):
        # parameter grids are declared in the class 
        self.arima_grid = arima_grid
        self.garch_grid = garch_grid
        self.gas_grid = gas_grid
        self.ll_grid = ll_grid
        self.llnormal_grid = llnormal_grid
        # initializing class varaibles
        self.steps = steps
        self.maxevals = maxevals
        self.aic = int(maxsize)
        self.modelbest = None
        self.order = 0
       
    def arima(self, timeseries):
        """
        The function implements hyper-parameter tuning of PyFlux ARIMA models (for given training series)
        
        Returns
        --------------------------------------------
        An dictionary {algorithm name: [best aic model, integration order, aic value, model parameters_dict]} for the best ARIMA model
        
        """
        
        self.aic = int(maxsize)
        self.modelbest = None
        
        '''
        defining fn for hyperopt optimization
        --------------------------------
        params: combination - list of parameters from the grid
        loss: expression to be minimized
        status: Set to STATUS_OK if loss is successfully computed, otherwise STATUS_FAIL 
        '''
        
        def func(params):
            model = pf.ARIMA(data=pd.DataFrame(timeseries), ar=params['p'], 
                             ma=params['q'], integ=params['d'])
            x=model.fit(params['method'])
            if isnan(x.aic) or model.predict(self.steps).isna().sum().sum()>0:
                return {'status':STATUS_FAIL}
            else:
                if x.aic < self.aic:
                    self.aic = x.aic
                    self.modelbest = model
                    self.order = params['d']
                return {'loss':x.aic, 'status':STATUS_OK}
            
        '''
        fmin returns dictionary of parameters for best model
        --------------------------------
        fn:function returning loss, 
        space: grid of hyperparameters, 
        algo:algo used while grid seach
        max_evals: number of parameter combinations(repeated) evaluated before returning the bestmodel
        best: parameters of best model
        '''
            
        best = fmin(fn= func,space= self.arima_grid ,algo=tpe.suggest,max_evals= self.maxevals)
        print( 'arima,  AIC:' +str(self.aic) + '   '+ str(best))
        return { 'arima' : [self.modelbest, self.order, self.aic, best]}
        
 
    def garch(self, timeseries):
        """
        The function implements hyper-parameter tuning of PyFlux GARCH models (for given training series)
        For detailed check arima function
        
        Returns
        --------------------------------------------
        An dictionary {algorithm name: [best aic model, integration order, aic value]} for the best GARCH model
        
        """
        self.aic = int(maxsize)
        self.modelbest=None
        def func(params):
            model = pf.GARCH(data = pd.DataFrame(timeseries), 
                                 p=params['p'], q=params['q'])
            x=model.fit(params['method'])
            if isnan(x.aic) or model.predict(self.steps).isna().sum().sum()>0:
                return {'status':STATUS_FAIL}
            else:
                if x.aic < self.aic:
                    self.aic = x.aic
                    self.modelbest = model
                    self.order = 0
                return {'loss':x.aic, 'status':STATUS_OK}
        best = fmin(fn= func,space= self.garch_grid ,algo=tpe.suggest,max_evals= self.maxevals)
        print('garch,  AIC:' +str(self.aic) + '   '+ str(best))
        return { 'garch' : [self.modelbest,self.order, self.aic, best]}
        

    def egarch(self, timeseries):
        """
        The function implements hyper-parameter tuning of PyFlux EGARCH models (for given training series)
        For detailed check arima function
        
        Returns
        --------------------------------------------
        An dictionary {algorithm name: [best aic model, integration order, aic value]} for the best EGARCH model
        
        """
        self.aic = int(maxsize)
        self.modelbest=None
        def func(params):
            model = pf.EGARCH(data = pd.DataFrame(timeseries), 
                                 p=params['p'], q=params['q'])
            x=model.fit(params['method'])
            if isnan(x.aic) or model.predict(self.steps).isna().sum().sum()>0:
                return {'status':STATUS_FAIL}
            else:
                if x.aic < self.aic:
                    self.aic = x.aic
                    self.modelbest = model
                    self.order = 0
                return {'loss':x.aic, 'status':STATUS_OK}
        best = fmin(fn= func,space= self.garch_grid ,algo=tpe.suggest,max_evals= self.maxevals)
        print('egarch,  AIC:' +str(self.aic) + '   '+ str(best))
        return {'egarch' : [self.modelbest,self.order, self.aic, best]}
    

    def egarchm(self, timeseries):
        """
        The function implements hyper-parameter tuning of PyFlux EGARCHM models (for given training series). 
        For detailed check arima function
        
        Returns
        --------------------------------------------
        An dictionary {algorithm name: [best aic model, integration order, aic value, model parameters]} for the best EGARCHM model
        
        """

        self.aic = int(maxsize)
        self.modelbest=None
        def func(params):
            model = pf.EGARCHM(data = pd.DataFrame(timeseries), 
                                 p=params['p'], q=params['q'])
            x=model.fit(params['method'])
            if isnan(x.aic) or model.predict(self.steps).isna().sum().sum()>0:
                return {'status': STATUS_FAIL}
            else:
                if x.aic < self.aic:
                    self.aic = x.aic
                    self.modelbest = model
                    self.order = 0
                return {'loss':x.aic, 'status':STATUS_OK}
        best = fmin(fn= func,space= self.garch_grid ,algo=tpe.suggest,max_evals= self.maxevals)
        print('egarchm,  AIC:' +str(self.aic) + '   '+ str(best))
        return {'egarchm' : [self.modelbest,self.order, self.aic, best]}
    
        
    def lmegarch(self, timeseries):
        """
        The function implements hyper-parameter tuning of PyFlux LMEGARCH models (for given training series). 
        For detailed check arima function
        
        Returns
        --------------------------------------------
        An dictionary {algorithm name: [best aic model, integration order, aic value, model parameters]} for the best LMEGARCH model
        
        """

        self.aic = int(maxsize)
        self.modelbest=None
        def func(params):
            model = pf.LMEGARCH(data = pd.DataFrame(timeseries), 
                                 p=params['p'], q=params['q'])
            x=model.fit(params['method'])
            if isnan(x.aic) or model.predict(self.steps).isna().sum().sum()>0:
                return {'status': STATUS_FAIL}
            else:
                if x.aic < self.aic:
                    self.aic = x.aic
                    self.modelbest = model
                    self.order = 0
                return {'loss':x.aic, 'status':STATUS_OK}
        best = fmin(fn= func,space= self.garch_grid ,algo=tpe.suggest,max_evals= self.maxevals)
        print('lmegarch,  AIC:' +str(self.aic) + '   '+ str(best))
        return {'lmegarch' : [self.modelbest,self.order, self.aic, best]}
    

    def segarch(self, timeseries):
        """
        The function implements hyper-parameter tuning of PyFlux SEGARCH models (for given training series). 
        For detailed check arima function
        
        Returns
        --------------------------------------------
        An dictionary {algorithm name: [best aic model, integration order, aic value, model parameters]} for the best SEGARCH model
        
        """

        self.aic = int(maxsize)
        self.modelbest=None
        def func(params):
            model = pf.SEGARCH(data = pd.DataFrame(timeseries), 
                                 p=params['p'], q=params['q'])
            x=model.fit(params['method'])
            if isnan(x.aic) or model.predict(self.steps).isna().sum().sum()>0:
                return {'status': STATUS_FAIL}
            else:
                if x.aic < self.aic :
                    self.aic = x.aic
                    self.modelbest = model
                    self.order = 0
                return {'loss':x.aic, 'status':STATUS_OK}
        best = fmin(fn= func,space= self.garch_grid ,algo=tpe.suggest,max_evals= self.maxevals)
        print('segarch,  AIC:' +str(self.aic) + '   '+ str(best))
        return {'segarch' : [self.modelbest,self.order, self.aic, best]}
    
        
    def segarchm(self, timeseries):
        """
        The function implements hyper-parameter tuning of PyFlux SEGARCHM models (for given training series). 
        For detailed check arima function
        
        Returns
        --------------------------------------------
        An dictionary {algorithm name: [best aic model, integration order, aic value, model parameters]} for the best SEGARCHM model
        
        """

        self.aic = int(maxsize)
        self.modelbest=None
        def func(params):
            model = pf.SEGARCHM(data = pd.DataFrame(timeseries), 
                                 p=params['p'], q=params['q'])
            x=model.fit(params['method'])
            if isnan(x.aic) or model.predict(self.steps).isna().sum().sum()>0:
                return {'status': STATUS_FAIL}
            else:
                if x.aic < self.aic:
                    self.aic = x.aic
                    self.modelbest = model
                    self.order = 0
                return {'loss':x.aic, 'status':STATUS_OK}
        best = fmin(fn= func,space= self.garch_grid ,algo=tpe.suggest,max_evals= self.maxevals)
        print('segarchm,  AIC:' +str(self.aic) + '   '+ str(best))
        return {'segarchm' : [self.modelbest,self.order, self.aic, best]}
        

    def gas(self, timeseries):
        """
        The function implements hyper-parameter tuning of PyFlux GAS models (for given training series). 
        For detailed check arima function
        
        Returns
        --------------------------------------------
        An dictionary {algorithm name: [best aic model, integration order, aic value, model parameters]} for the best GAS model
        
        """

        self.aic = int(maxsize)
        self.modelbest=None
        self.count = 0
        def func(params):
            model = pf.GAS(data=pd.DataFrame(timeseries),ar=params['ar'],
                       sc=params['sc'],integ=params['integ'],family = params['family'])
            x=model.fit(params['method'])
            if isnan(x.aic) or model.predict(self.steps).isna().sum().sum()>0:
                return {'status': STATUS_FAIL}
            else:
                if x.aic < self.aic:
                    self.aic = x.aic
                    self.modelbest = model
                    self.order = params['integ']
                return {'loss':x.aic, 'status':STATUS_OK}
        best = fmin(fn= func,space= self.gas_grid ,algo=tpe.suggest,max_evals= self.maxevals)
        print('gas,  AIC:' +str(self.aic) + '   '+ str(best))
        return {'gas' : [self.modelbest,self.order, self.aic, best]}

    
    def gasllev(self, timeseries):
        """
        The function implements hyper-parameter tuning of PyFlux GASLLEV models (for given training series). 
        For detailed check arima function
        
        Returns
        --------------------------------------------
        An dictionary {algorithm name: [best aic model, integration order, aic value, model parameters]} for the best GASLLEV model
        
        """

        self.aic = int(maxsize)
        self.modelbest=None
        def func(params):
            model = pf.GASLLEV(data=pd.DataFrame(timeseries),
                               integ=params['integ'],family = params['family'])
            x=model.fit(params['method'])
            if isnan(x.aic) or model.predict(self.steps).isna().sum().sum()>0:
                return {'status': STATUS_FAIL}
            else:
                if x.aic < self.aic:
                    self.aic = x.aic
                    self.modelbest = model
                    self.order = params['integ']
                return {'loss':x.aic, 'status':STATUS_OK}
        best = fmin(fn= func,space= self.ll_grid ,algo=tpe.suggest,max_evals= self.maxevals)
        print('gasllev,  AIC:' +str(self.aic) + '   '+ str(best))
        return {'gasllev' : [self.modelbest,self.order, self.aic, best]}
        
   
    def gasllt(self, timeseries):
        """
        The function implements hyper-parameter tuning of PyFlux GASLLT models (for given training series). 
        For detailed check arima function
        
        Returns
        --------------------------------------------
        An dictionary {algorithm name: [best aic model, integration order, aic value, model parameters]} for the best GASLLT model
        
        """

        self.aic = int(maxsize)
        self.modelbest=None
        def func(params):
            model = pf.GASLLT(data=pd.DataFrame(timeseries),
                               integ=params['integ'],family= params['family'])
            x=model.fit(params['method'])
            if isnan(x.aic) or model.predict(self.steps).isna().sum().sum()>0:
                return {'status': STATUS_FAIL}
            else:
                if x.aic < self.aic:
                    self.aic = x.aic
                    self.modelbest = model
                    self.order = params['integ']
                return {'loss':x.aic, 'status':STATUS_OK}
        best = fmin(fn= func,space= self.ll_grid ,algo=tpe.suggest,max_evals= self.maxevals)
        print('gasllt,  AIC:' +str(self.aic) + '   '+ str(best))
        return {'gasllt' : [self.modelbest,self.order, self.aic, best]}
    
        
    def llev(self, timeseries):
        """
        The function implements hyper-parameter tuning of PyFlux LLEV models (for given training series). 
        For detailed check arima function
        
        Returns
        --------------------------------------------
        An dictionary {algorithm name: [best aic model, integration order, aic value, model parameters]} for the best LLEV model
        
        """

        self.aic = int(maxsize)
        self.modelbest=None
        def func(params):
            model = pf.LLEV(data=pd.DataFrame(timeseries),integ=params['integ'])
            x=model.fit(params['method'])
            if isnan(x.aic) or model.predict(self.steps).isna().sum().sum()>0:
                return {'status': STATUS_FAIL}
            else:
                if x.aic < self.aic:
                    self.aic = x.aic
                    self.modelbest = model
                    self.order = params['integ']
                return {'loss':x.aic, 'status':STATUS_OK}
        best = fmin(fn= func,space= self.llnormal_grid ,algo=tpe.suggest,max_evals= self.maxevals)
        print('llev,  AIC:' +str(self.aic) + '   '+ str(best))
        return {'llev' : [self.modelbest, self.order , self.aic, best]}
    
    
    def llt(self, timeseries):
        """
        The function implements hyper-parameter tuning of PyFlux LLT models (for given training series). 
        For detailed check arima function
        
        Returns
        --------------------------------------------
        An dictionary {algorithm name: [best aic model, integration order, aic value, model parameters]} for the best LLT model
        
        """

        self.aic = int(maxsize)
        self.modelbest=None
        def func(params):
            model = pf.LLT(data=pd.DataFrame(timeseries),integ=params['integ'])
            x=model.fit(params['method'])
            if isnan(x.aic) or model.predict(self.steps).isna().sum().sum()>0:
                return {'status': STATUS_FAIL}
            else:
                if x.aic < self.aic:
                    self.aic = x.aic
                    self.modelbest = model
                    self.order = params['integ']
                return {'loss':x.aic, 'status':STATUS_OK}
        best = fmin(fn= func,space= self.llnormal_grid ,algo=tpe.suggest,max_evals= self.maxevals)
        print('llt,  AIC:' +str(self.aic) + '   '+ str(best))
        return {'llt' : [self.modelbest,self.order, self.aic, best]}
    
    
    def nllev(self, timeseries):
        """
        The function implements hyper-parameter tuning of PyFlux NLLEV models (for given training series). 
        For detailed check arima function
        
        Returns
        --------------------------------------------
        An dictionary {algorithm name: [best aic model, integration order, aic value, model parameters]} for the best NLLEV model
        
        """

        self.aic = int(maxsize)
        self.modelbest=None
        def func(params):
            model = pf.NLLEV(data=pd.DataFrame(timeseries),
                               integ=params['integ'],family = params['family'])
            x=model.fit(params['method'])
            if isnan(x.aic) or model.predict(self.steps).isna().sum().sum()>0:
                return {'status': STATUS_FAIL}
            else:
                if x.aic < self.aic:
                    self.aic = x.aic
                    self.modelbest = model
                    self.order = params['integ']
                return {'loss':x.aic, 'status':STATUS_OK}
        best = fmin(fn= func,space= self.ll_grid ,algo=tpe.suggest,max_evals= self.maxevals)
        print('nllev,  AIC:' +str(self.aic) + '   '+ str(best))
        return {'nllev' : [self.modelbest,self.order, self.aic, best]}
    
    
    def nllt(self, timeseries):
        """
        The function implements hyper-parameter tuning of PyFlux NLLT models (for given training series). 
        For detailed check arima function
        
        Returns
        --------------------------------------------
        An dictionary {algorithm name: [best aic model, integration order, aic value, model parameters]} for the best NLLT model
        
        """

        self.aic = int(maxsize)
        self.modelbest=None
        def func(params):
            model = pf.NLLT(data=pd.DataFrame(timeseries),
                               integ=params['integ'],family = params['family'])
            x=model.fit(params['method'])
            if isnan(x.aic) or model.predict(self.steps).isna().sum().sum()>0:
                return {'status': STATUS_FAIL}
            else:
                if x.aic < self.aic:
                    self.aic = x.aic
                    self.modelbest = model
                    self.order = params['integ']
                return {'loss':x.aic, 'status':STATUS_OK}
        best = fmin(fn= func,space= self.ll_grid ,algo=tpe.suggest,max_evals= self.maxevals)
        print('nllt,  AIC:' +str(self.aic) + '   '+ str(best))
        return {'nllt' : [self.modelbest, self.order, self.aic, best]}