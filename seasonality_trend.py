import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

class SeasonalityDetector:
    """
    Class for Seasonal decomposition using moving averages
    
    Available filters
    --------------------------------------------
    + statsmodels.tsa.filters.bk_filter.bkfilter
    + statsmodels.tsa.filters.cf_filter.xffilter
    + statsmodels.tsa.filters.hp_filter.hpfilter
    + statsmodels.tsa.filters.convolution_filter
    
    Parameters:
    --------------------------------------------
    + timeseries (array-like) – Time series. If 2d, individual series are in columns.
    + model (str {"additive", "multiplicative"}) – Type of seasonal component.
    + filter (array-like) – The filter coefficients for filtering out the seasonal component. 
    
    Returns:
    results – A object with seasonal, trend, and residuals components.

    Return type:
    - objClass for specification of wether to apply an additive or multiplicative model is required. 
    - Residuals in the decomposed model may still contains some traceable components
    
    Visualizating components using plots likes:
    --------------------------------------------
    + Plots of rolling average : Trend Visualization
    + Plots for components of time series without filter : Additive Model
    + Plots for components of time series without filter : Multiplicative Model
    + Plots for components of time series using filters
    
    """
    
    def __init__(self, timeseries, periodicity):
        self.timeseries = timeseries
        self.periodicity = periodicity
        self.trend_seasonal_dict = {}
        
    def bk_Filter(self):
        """
        Baxter-King bandpass filter
        ---------------------------
        + low (float) – Minimum period for oscillations
        Note :- BK suggest that the Burns-Mitchell U.S. business cycle has 6 for quarterly data and 1.5 for annual data.
        + high (float) – Maximum period for oscillations 
        Note :- BK suggest that the U.S. business cycle has 32 for quarterly data and 8 for annual data.
        + K (int) – Lead-lag length of the filter 
        Note :- BK propose a truncation length of 12 for quarterly data and 3 for annual data.
    
        """
        low_q = 6
        high_q = 32
        K_q = 12
        low = low_q*3/self.periodicity
        high = high_q*3/self.periodicity
        K = int(K_q*3/self.periodicity)
        bk_cycles = sm.tsa.filters.bkfilter(self.timeseries, low, high, K)
        
        return bk_cycles
    
    def cf_Filter(self):
        """
        Christiano Fitzgerald asymmetric, random walk filter
        ----------------------------------------------------
        + low (float) – Minimum period of oscillations. 
        Note :- Features below low periodicity are filtered out. Default is 6 for quarterly data, giving a 1.5 year periodicity.
        + high (float) – Maximum period of oscillations. 
        Note :- Features above high periodicity are filtered out. Default is 32 for quarterly data, giving an 8 year periodicity.
        + drift (bool) – Whether or not to remove a trend from the data. 
        Note :- The trend is estimated as np.arange(nobs)*(X[-1] - X[0])/(len(X)-1)
    
        """
        low_q = 6
        high_q = 32
        low = low_q*3/self.periodicity
        high = high_q*3/self.periodicity
        cf_cycles, cf_trend = sm.tsa.filters.cffilter(self.timeseries, low, high, drift=True)
        return cf_cycles, cf_trend
    
    def hp_Filter(self):
        """
        Hodrick-Prescott filter
        -----------------------
        lamb (float) – The Hodrick-Prescott smoothing parameter. 
        + A value of 1600 is suggested for quarterly data. 
        + Ravn and Uhlig suggest using a value of 6.25 (1600/4**4) for annual data and 129600 (1600*3**4) for monthly data.
    
        """
        lamb_q = 1600
        lamb = lamb_q*np.power(3/self.periodicity, 4)
        hp_cycles, hp_trend = sm.tsa.filters.hpfilter(self.timeseries, lamb)
        return hp_cycles, hp_trend
    
    def convolutional_Filter(self, filt):
        """
        Linear filtering via convolution. Centered and backward displaced moving weighted average
        -----------------------------------------------------------------------------------------
        + filt (array_like) – Linear filter coefficients in reverse time-order. 
        Note :- Should have the same number of dimensions as x though if 1d and x is 2d will be coerced to 2d.
        + nsides (int, optional) – If 2, a centered moving average is computed using the filter coefficients. 
        Note :- If 1, the filter coefficients are for past values only. Both methods use scipy.signal.convolve.
    
        """
        conv_cycles, conv_trend = sm.tsa.filters.convolution_filter(self.timeseries, filt, nsides=2)
        return conv_cycles, conv_trend
        
    
    def plot_rolling_average(self):
        '''
        Plot rolling mean and rolling standard deviation for a given time series and window
        Parameters
        ----------
        y : pandas.Series
        window : length of averaging
        '''
        # calculate moving averages
        rolling_mean = self.timeseries.rolling(window = self.periodicity).mean()
        rolling_std = self.timeseries.rolling(window = self.periodicity).std()

        # plot statistics
        plt.plot(self.timeseries, label='Original')
        plt.plot(rolling_mean, color='crimson', label='Moving average mean')
        plt.plot(rolling_std, color='darkslateblue', label='Moving average standard deviation')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation', fontsize=20)
        plt.xlabel('Year')
        plt.ylabel('Y')
        plt.savefig('./img/moving_average.png')
        plt.show(block=False)
        return rolling_mean, rolling_std
     
        
    def plot_trend_seasonal_comp(self, t, s):
        plt.plot(self.trend_seasonal_dict[t][0])
        plt.plot(self.trend_seasonal_dict[s][1])
        plt.show();
        
        
    def trend_seasonal_comp(self):
        print("You are in seasonal_trend module calculating all components of timeseries")
        hp_cycle, hp_trend = self.hp_Filter()
        bk_cycle = self.bk_Filter()
        cf_cycle, cf_trend  = self.cf_Filter()
        add_decomp = seasonal_decompose(self.timeseries, model = "additive", period = self.periodicity)
        mul_decomp = seasonal_decompose(self.timeseries, model = "multiplicative", period = self.periodicity)
        self.trend_seasonal_dict.update({'add_decomp': [add_decomp.trend, add_decomp.seasonal]})
        self.trend_seasonal_dict.update({'mul_decomp': [mul_decomp.trend, mul_decomp.seasonal]})
        self.trend_seasonal_dict.update({'cf': [cf_trend, cf_cycle]})
        self.trend_seasonal_dict.update({'hp': [hp_trend, hp_cycle]})
        self.trend_seasonal_dict.update({'bk': [None, bk_cycle]})
        
