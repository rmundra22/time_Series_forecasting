from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import ml_metrics as met
from sklearn import metrics

def rmse(actual, prediction):
    """
    Calculates the Root Mean Square Error between the Actual and Prediction
    """
    actual, prediction = np.array(actual).flatten(), np.array(prediction).flatten()
    return sqrt(mean_squared_error(actual, prediction))

def mape(actual, prediction):
    """
    Calculates the Mean Absolute Percentage Error between the Actual and Prediction
    """
    actual, prediction = np.array(actual).flatten(), np.array(prediction).flatten()
    return np.mean(np.abs((actual - prediction) / actual)) * 100

def mfe(actual, prediction):
    """
    Calculates the Mean Forecasted Error between the Actual and Predction
    """
    actual, prediction = np.array(actual).flatten(), np.array(prediction).flatten()
    return np.mean(actual - prediction)

def mae(actual, prediction):
    """
    Calculates the Mean Absolute Error between the Actual and Prediction
    """
    actual, prediction = np.array(actual), np.array(prediction)
    return metrics.mean_absolute_error(actual, prediction)

def mfpe(actual, prediction):
    """
    Calculates the Mean Forecasted Percentage Error between the Actual and Predction
    """
    actual, prediction = np.array(actual), np.array(prediction)
    return np.mean((actual - prediction) / actual) * 100


def sse(actual, prediction):
    """
    Calculates the Sum of the Squared Error between the Actual and Prediction
    """
    actual, prediction = np.array(actual).flatten(), np.array(prediction).flatten()
    return np.sum(np.square(actual-prediction))

def smse(actual, prediction):
    """
    Calculates the Signed Mean Squared Error between the Actual and Prediction
    """
    actual, prediction = np.array(actual), np.array(prediction)
    diff = actual -  prediction
    return np.mean( (diff * np.square(diff))/np.abs(diff) )

def nwrmsle(actual, prediction, weights):
    """
    Normalized Weighted Root Mean Squared Logarithmic Error (NWRMSLE), calculated as follows:
        
    NWRMSLE = \sqrt{ \frac{\sum_{i=1}^n w_i \left( \ln(\hat{y}_i + 1) - \ln(y_i +1)  \right)^2  }{\sum_{i=1}^n w_i}}
    
    where for row i, y^i is the predicted unit of an item and yi is the actual unit;
    n is the total number of rows in the test set.
    w^i is the weight of each item
    
    - This metric is suitable when predicting values across a large range of orders of magnitudes.
    - It avoids penalizing large differences in prediction when both the predicted and the true number are large: 
    - Predicting 5 when the true value is 50 is penalized more than predicting 500 when the true value is 545.
    
    Source: https://www.kaggle.com/c/favorita-grocery-sales-forecasting#evaluation
    """
    actual, prediction = np.array(actual), np.array(prediction)
    diff = np.log1p(actual) - np.log1p(prediction)
    diff = np.square(diff)
    return sqrt( (weights*diff)/np.sum(weights) )

def rmspe(actual, prediction):
    """
    Calculates the Root Mean Square Percentage Error between the Actual and Prediction
    Source: https://www.kaggle.com/c/rossmann-store-sales#evaluation
    """
    actual, prediction = np.array(actual), np.array(prediction)
    diff = actual -  prediction
    return sqrt(np.mean(np.square(diff/actual)))

def rmsle(actual, prediction):
    """
    Calculates the Root Mean Square Logarithmic Error between the Actual and Prediction
    Source: https://www.kaggle.com/c/santander-value-prediction-challenge#evaluation
    """
    actual, prediction = np.array(actual), np.array(prediction)
    diff = np.log1p(actual) - np.log1p(prediction)
    return sqrt(np.mean(np.square(diff)))

def map_k(actual, prediction, k):
    """
    Calculate the Mean Average Precision of K (MAP@K) 
    Source: https://www.kaggle.com/wendykan/map-k-demo
    """
    actual, prediction = np.array(actual), np.array(prediction)
    ret = [met.apk([val],prediction[i:i+k],k) for i, val in enumerate(actual)]
    return ret
    
def unlr(Y, Y_hat):
    n = Y_hat.shape[0]
    p = Y_hat.shape[1]
    ERR_num2 = 0
    ERR_num = np.dot( np.transpose(abs( np.array(Y) - np.array(Y_hat) )), np.ones(n).reshape(n,1) )/p
    ERR_den = np.dot( np.transpose(Y), np.ones(n).reshape(n,1))
    ERR_models = np.ones((p,1))
    print("ERR_num", ERR_num)
    print("ERR_den", ERR_den)
    ERR_models = ERR_models - np.divide(ERR_num, ERR_den).reshape(p,1)
    ERR = 1 - np.sum(ERR_num)/(ERR_den[0]*p)

    try:
        n_1 = 3
        ERR_num_1 = np.dot( np.transpose(abs( np.array(Y[6:9]) - np.array(Y_hat[6:9]) )), np.ones(n_1).reshape(n_1,1) )/p
        ERR_num_2 = np.dot( np.transpose(abs( np.array(Y[9:12]) - np.array(Y_hat[9:12]) )), np.ones(n_1).reshape(n_1,1) )/p
        ERR_den_1 = np.dot( np.transpose(Y[6:9]), np.ones(n_1).reshape(n_1,1))
        ERR_den_2 = np.dot( np.transpose(Y[9:12]), np.ones(n_1).reshape(n_1,1))
        ERR_1 = 1 - np.sum(ERR_num_1)/(ERR_den_1[0]*p)
        ERR_2 = 1 - np.sum(ERR_num_2)/(ERR_den_2[0]*p)
    except:
        pass
    return ERR_models, ERR, ERR_1, ERR_2
