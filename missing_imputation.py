import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

rng = np.random.RandomState(0)

class DataImputer:
    """
    Class to impute the missing value in the time series data
    + Mean
    + Hot Deck
    + AR
    + Kalman Filter
    -----------------------------
    Input
    1D array, timeseries data in Pandas Series format

    """
    def __init__(self, timeseries, window = 4):
        #self.timeseries.data = timeseries
        self.timeseries = timeseries
        self.window = window
        self.score = None

    def meanInpute(self):#, freq):
        """
        Return means for each period in x. freq is an int that gives the
        number of periods per cycle. E.g., 12 for monthly. NaNs are ignored
        in the mean.
        """
        # np.array([pd_mean(x[i::freq], axis=0) for i in range(freq)])
        self.timeseries = self.timeseries.fillna(self.timeseries.mean())


    def lastobsImpute(self):
        self.timeseries = self.timeseries.fillna(method = 'ffill')


    def arImpute(self):
        self.timeseries


    def emkalmanImpute(self):
        X = np.ma.array(self.timeseries)
        kf = KalmanFilter(em_vars=['transition_covariance'])
        for l in range(len(X)):
            if np.isnan(float(X[l])):
                X[l] = np.ma.masked
        out = kf.em(X).smooth(X)[0]
        for l in range(len(self.timeseries)):
            if np.isnan(timeseries[l]):
                self.timeseries[l] = out[l][0]
                

    def RandomForrest_Regressor(self, missing_rate):

        # Fucntion use RandomForrest models to input missing values for both target variable as well as regressors
        # Timeseries dataset should be in the form of panel data i.e. both target-timeseries with regressors
        
        X_full, y_full = self.timeseries.data, self.timeseries
        n_samples = X_full.shape[0]
        n_features = X_full.shape[1]

        # Estimate the score on the entire dataset, with no missing values
        estimator = RandomForestRegressor(random_state=0, n_estimators=100)
        self.score['tol_dataset'] = cross_val_score(estimator, X_full, y_full).mean()


        # Scoring after adding the missing values in x% of the lines
        if(missing_rate == None): 
            missing_rate = 0.75
        n_missing_samples = int(np.floor(n_samples * missing_rate))
        missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples, dtype=np.bool),
                                     np.ones(n_missing_samples, dtype=np.bool)))
        rng.shuffle(missing_samples)
        missing_features = rng.randint(0, n_features, n_missing_samples)

        # Estimate the score without the lines containing missing values
        X_filtered = X_full[~missing_samples, :]
        y_filtered = y_full[~missing_samples]
        estimator = RandomForestRegressor(random_state=0, n_estimators=100)
        self.score['dataset_with_missing_values'] = cross_val_score(estimator, X_filtered, y_filtered).mean()
        

        # Estimate the score after imputation of the missing values
        X_missing = X_full.copy()
        X_missing[np.where(missing_samples)[0], missing_features] = 0
        y_missing = y_full.copy()
        # Fit all the transforms one after the other to data, then fit the transformed data using the final estimator
        # estimator : estimator object implementing ‘fit’ i.e. the object to use to fit the data.
        estimator = Pipeline([("imputer", SimpleImputer(missing_values=0, strategy="mean", axis=0)),
                              ("forest", RandomForestRegressor(random_state=0, n_estimators=100))])
        self.score['imputing_missinf_values'] = cross_val_score(estimator, X_missing, y_missing).mean()

        # Things to add :-
            # removal of method that is adding missing values to dataset
            # detection of missing values and replacing them with zero
            # imputting the missing values using randomforrest regressor
            # Check for other strategies too Eg: median or mode in range

        return 