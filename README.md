# time_Series_forecasting

This project mainly focuses on handling the crippling challenges associated with producing reliable and high quality forecasts. The variation in forecasting problems and finding the analysts with required expertise in time series modeling is hard and time consuming process. Even with employed experts sometimes these analysis becomes very hectic and time consuming. To address these challenges, I describe a practical approach to forecast “at scale” that combines configurable models with analyst’s performance analysis. I propose an ensembling approach of 15+ traditional timeseries models with interpretable parameters that can be automatically tuned or can be intuitively adjusted by the experts. I also discuss performance analyses to compare, evaluate and select best between forecasting procedures. This Python based tool helps analyst to use their expertise most effectively by helping them with fast, reliable and practical forecasting of business time series (univariate/multivariate).


# Objective

To address the client product’s sales data using conventional time series analysis and machine learning techniques. Figure out an approach to uncover consumer behavioral patterns across states and products both historically as well as in real time by analyzing multi-product patterns.
 
Demand forecasting will continue to be one of the most important elements of the retailers’ planning process so main question that needs to be answered is to make clear projection of how much a product should sell at a specific location at a specific time and planning for demand according to local consumer preferences.

  ### ● To build a Python based tool that generates fast, robust, practical and n-step forward forecasts with historical Industry-scale timeseries data.
  ### ● Tool should offers functionalities for optimizing model fitting by allowing hyper-parameter tuning and best model selection techniques.
  ### ● Tool procedures should be robust to abnormalities in the data like missing data, outliers (anomalies), non-linear trends, seasonality effects etc.
  

# Methodological Approach (Pipeline)

Separate python modules with specific functionalities were deployed utilizing the techniques discussed in below mentioned section:

### ● Preprocessing of data - Anomaly Detection
To detect Outliers using Density based Spatial Clustering Algorithm (DBSCAN) and Local Outlier Probabilities (LoOP) . To use mean value or implements the Kalman Filter, Kalman Smoother, and EM Algorithm for a Linear Gaussian model to carry out Missing values Imputation.

### ● Signal Frequency / Periodicity Detector
To detect the timeseries (signal) frequency using techniques like Fourier Transformation, Auto correlation function, Harmonic Product Spectrum and Counting zero crossings. The calculated frequency is used as input for decomposing timeseries into its components

### ● Trend and Seasonality Decomposition
To detect and plot trend and seasonal components of timeseries. To implement Decomposition techniques using Baxter-King bandpass filter, Christiano Fitzgerald asymmetric random walk filter, Hodrick Prescott filter and Convolution filter. This is used when your series exhibits a cyclic pattern, with or without a trend.

### ● Hyperparameter tuning and parameter fitting
To optimally tune 21 traditional time series models like ARIMA, GARCH, GAS, Holt Winters, curve smoothing techniques, averaging methods etc. These models are trained and tested using differently explored datasets. Model’s hyper parameters and parameters are processed using Random Search Algorithm, Tree of Parzen Estimators (TPE), Limited-memory Broyden – Fletcher – Goldfarb – Shanno (L-BFGS) algorithm and Akaike Information Criterion (AIC).

### ● Adding back seasonality and trend components
To take all the models with tuned hyper-parameters/ parameters and forecast sales across test dataset. To revert back the differenced series if present and then add back seasonality and trend components to the predicted series of each model. Trend component is added back using Polynomial Regression Fitting and seasonality is added back considering test data periodicity.

### ● Best Model Selection
To detect the best fitted model among the 21 models using a list of 11 scoring metrics like RMSE, MAPE, MFE, MAE, MFPE, SSE, NWRMSLE, RMSLE etc. To aggregate the predictions results from the all the models (weak learner) and transform it into a single model (strong learner) using ensembling techniques.

### ● Testing fitted model using Client’s Evaluation Metric
To carry out performance analysis of the obtained model Client’s Evaluation Metric is calculated. It is an accuracy measure of the predicted forecast over n-periods. Performance Analysis also involves metric calculation for TDP’s [t+5 : t+7] and TDP’s [t+8 : t+10] periods which are 2 and 3 months ahead forecasts.

# Evaluation Metric

## Integral Normalized Absolute Mean Error

These metric measurements are useful to determine the reliability of a prediction, and enable the expert-in-loop to decide whether or not to trust the forecast results obtained from the tuned model. The Cluster-Basepack level Integral Normalized Absolute Mean Error (INMAE) index is computed with a cross-validation procedure, which involves the time series to be split into two parts:

● Training set : Approximately 83% of the total TDP’s is used to train the model.
● Train set: It is compared with the forecast of the test dataset.

INMAE represents the expected error of the computed forecast vs true forecast on future values averaged by true future values across all Cluster-Basepack combinations. It is a score that helps experts decide how finely our model is able to generalize. High score signify a highly accurate and generalized model that is able to predict the n-period ahead sales value at Cluster-Basepack level.

        Accuracy Measure = 

where, Y is the actual test dataset values.
Y_(hat,i) is the predicted test dataset values.
P represents no. of Cluster-Basepack combinations.
I represents number of test data points
