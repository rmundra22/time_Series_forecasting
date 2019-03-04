from __future__ import division
from numpy.fft import rfft
import pandas as pd
import numpy as np
from numpy import argmax, mean, diff, log
from matplotlib.mlab import find
from scipy.signal import blackmanharris, fftconvolve
import sys
from sklearn import preprocessing

class frequency_estimator:
    
    def __init__(self, timeseries):
        self.timeseries = timeseries

        
    def parabolic(self, f, x):
        """
        Quadratic Interpolation for estimating the true position of an
        inter-sample maximum when nearby samples are known.

        Inputs
        -----------------------------------------------------
        f: a vector 
        x: index for that vector

        Returns (vx, vy)
        -----------------------------------------------------
        the coordinates of the vertex of a parabola that goes
        through point x and its two neighbors.

        """
        # Requires real division.  Insert float() somewhere to force it?
        xv = 1/2 * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
        yv = f[x] - 1/4 * (f[x-1] - f[x+1]) * (xv - x)
        return (xv, yv)
    

    def freq_from_crossings(self):
        """
        Estimate 1/frequency by counting zero crossings
        """
        sig = preprocessing.scale(self.timeseries)
        # Find all indices right before a rising-edge zero crossing
        indices = find((sig[1:] >= 0) & (sig[:-1] < 0))

        # Naive (Measures 1000.185 Hz for 1000 Hz, for instance)
        # crossings = indices

        # More accurate, using linear interpolation to find intersample
        # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
        crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices]
        
        # Some other interpolation based on neighboring points might be better.
        # Spline, cubic, whatever
        crossings = pd.DataFrame(crossings)
        crossings = np.array(crossings).flatten()
        return round(mean(diff(crossings)),0)
    
    
    def freq_from_fft(self):
        """
        Estimate 1/frequency from peak of FFT
        """
        sig = preprocessing.scale(self.timeseries)
        # Compute Fourier transform of windowed signal
        windowed = sig * blackmanharris(len(sig))
        f = rfft(windowed)

        # Find the peak and interpolate to get a more accurate peak
        i = argmax(abs(f))  # Just use this for less-accurate, naive version
        true_i = self.parabolic(log(abs(f)), i)[0]

        # Convert to equivalent 1/frequency
        return len(windowed)/true_i


    def freq_from_autocorr(self):
        """
        Estimate 1/frequency using autocorrelation
        """
        sig = preprocessing.scale(self.timeseries)
        # Calculate autocorrelation (same thing as convolution, but with
        # one input reversed in time), and throw away the negative lags
        corr = fftconvolve(sig, sig[::-1], mode='full')
        corr = corr[len(corr)//2:]

        # Find the first low point
        d = diff(corr.flatten())
        start = find(d > 0)[0]

        # Find the next peak after the low point (other than 0 lag).  This bit is
        # not reliable for long signals, due to the desired peak occurring between
        # samples, and other peaks appearing higher.
        # Should use a weighting function to de-emphasize the peaks at longer lags.
        peak = argmax(corr[start:]) + start
        px, py = self.parabolic(corr, peak)

        return round(px.flatten()[0],0)


    def freq_from_HPS(self):
        """
        Estimate frequency using harmonic product spectrum (HPS)
        """
        sig = preprocessing.scale(self.timeseries)
        windowed = sig * blackmanharris(len(sig))

        from pylab import subplot, plot, log, copy, show

        # harmonic product spectrum:
        c = abs(rfft(windowed))
        maxharms = 8
        subplot(maxharms, 1, 1)
        plot(log(c))
        for x in range(2, maxharms):
            a = copy(c[::x])  # Should average or maximum instead of decimating
            # max(c[::x],c[1::x],c[2::x],...)
            c = c[:len(a)]
            i = argmax(abs(c))
            true_i = self.parabolic(abs(c), i)[0]
            f = true_i / len(windowed)
            print('Pass %d: %f Hz' , (x, f))
            c *= a
            subplot(maxharms, 1, x)
            plot(log(c))
        show()
        return
    
    
    def periodicity_estimator(self):
#         period_01 = self.freq_from_fft()
#         print('%f period_01', period_01)

        period_02 = self.freq_from_crossings()
        print('%f period_02', period_02)

        period_03 = self.freq_from_autocorr()
        print('%f period_03', period_03)

#         period_04 = self.freq_from_HPS()

        if (period_02%period_03 == 0)or(period_03%period_02 == 0):
            periodicity = round(np.min([period_02, period_03]),0)
        else:
            periodicity = round(np.average([period_02, period_03]),0)

        if periodicity >=12:
            print('Periodicity is: 12')
            return 12
        else:
            print('Periodicity is: ', periodicity)
            return int(periodicity)