from __future__ import division
import math
import numpy as np
from sys import maxsize
from collections import defaultdict
from scipy.optimize import fmin

class estimator:
    """
    Calculates AIC for given timeseries according to the distribution it follows
    
    """
    
    def __init__(self, timeseries):
        self.timeseries = np.array(timeseries).flatten()
        

    def aicmle(self, distribution):
        """ 
        Calculates maximum likelihood estimates according to the distribution of given timeseries
        
        """
        params = {}
        if distribution == 'pareto':
            params['xmin'] = np.min(self.timeseries)
            params['mu'] = 1 - self.timeseries.shape[0] / (self.timeseries.shape[0] * np.log(params['xmin']) - np.sum(np.log(self.timeseries)))
            return params

        elif distribution == 'lognormal':
            params['mu'] = np.sum(np.log(self.timeseries)) / self.timeseries.shape[0]
            params['sigma'] = np.sqrt(np.sum( (np.log(self.timeseries) - params['mu'])**2) / self.timeseries.shape[0])
            return params

        elif distribution == 'normal':
            params['mu'] = np.mean(self.timeseries)
            params['sigma'] = np.sqrt(sum((self.timeseries - np.mean(self.timeseries))**2) / self.timeseries.shape[0])
            return params

        elif distribution == 'exponential':
            params['lambda'] = 1.0 / np.mean(self.timeseries)
            return params

        elif distribution == 'boundedpl':
            params['xmin'] = np.min(self.timeseries)
            params['xmax'] = np.max(self.timeseries)
            minmuEstimate = 1.1
            params['mu'] = fmin(lambda mu: -len(self.timeseries) * 
                                np.log( (mu - 1) / (np.min(self.timeseries)**(1 - mu) - np.max(self.timeseries)**(1 - mu))) 
                                + mu * np.sum(np.log(self.timeseries)), minmuEstimate, disp=0)[0]
            return params


    def aiclike(self, params, distribution):
        """ 
        Calculates natural log likelihood values 
        
        """
        if distribution == 'pareto':
            nloglval = -(self.timeseries.shape[0] * np.log(params['mu']) + self.timeseries.shape[0] * params['mu'] * np.log(params['xmin']) - (params['xmin']+1) * np.sum(np.log(self.timeseries)))
            return nloglval

        elif distribution == 'lognormal':
            nloglval = np.sum(np.log(self.timeseries * params['sigma'] * np.sqrt(2*np.pi)) + (np.log(self.timeseries) - params['mu'])**2 / (2 * params['sigma']**2))
            return nloglval

        elif distribution == 'normal':
            nloglval = np.sum(np.log( params['sigma'] * np.sqrt(2*np.pi) ) + (self.timeseries - params['mu'])**2 / (2 * params['sigma']**2))
            return nloglval

        elif distribution == 'exponential':
            nloglval = np.sum(params['lambda'] * self.timeseries - np.log(params['lambda']))
            return nloglval

        elif distribution == 'boundedpl':
            nloglval = -len(self.timeseries) * np.log( (params['mu'] - 1) / (np.min(self.timeseries)**(1 - params['mu']) - np.max(self.timeseries)**(1 - params['mu']))) + params['mu'] * np.sum(np.log(self.timeseries))
            return nloglval




    def aicpdf(self, xvals, distribution, params):
        """ 
        Generates the values for the probability distributions 
        
        """
        if distribution == 'pareto':
            pvals = (params['xmin'] * params['mu'] ** params['xmin']) / (xvals ** (params['xmin'] + 1))
            return pvals

        elif distribution == 'lognormal':
            #import pdb; pdb.set_trace()
            pvals = np.exp(-(np.log(xvals) - params['mu'])**2 / (2 * params['sigma']**2)) / (xvals * params['sigma'] * np.sqrt(2*np.pi))
            return pvals

        elif distribution == 'normal':
            pvals = np.exp(-(xvals - params['mu'])**2 / (2 * params['sigma']**2)) / (params['sigma'] * np.sqrt(2*np.pi))
            return pvals

        elif distribution == 'exponential':
            pvals = params['lambda'] * np.exp(-params['lambda'] * xvals)
            return pvals

        elif distribution == 'boundedpl':
#             pvals = (params['mu'] * (params['mu'] ** params['xmax'] - params['xmin'] ** params['xmax'])) / (xvals ** (params['mu'] + 1))
            #mu * (xmax ^ mu - xmin ^ mu) / x ^ (mu+1)
            pvals = (params['mu'] * (params['xmax'] ** params['mu'] - params['xmin'] ** params['mu'])) / (xvals ** (params['mu'] + 1))
            return pvals


    def aic(self, ssc = 0):
        """ 
        Calculates Akaike's Information Criterion for the following distribtions:
        ------------------------------------------------------------------------
        + Pareto
        + Lognormal
        + Normal
        + Exponential
        + Boundedpl
        Gamma distribution in not included

        Paramters:
        -------------------------------------------------------------------------
        * ssc : 0 or 1
          flag to force the Small Sample Correction (ssc), which will be 
          enabled regardless for less than 40 data points

        Output: dictionary with an entry for each candidate distribution
        -------------------------------------------------------------------------
          [plots: substructure containing the vectors used to generate plots]
          [mle: contains the maximum likelihood estimate parameters for each
                distribution ]
          [nll: negative log likelihood values for each candidate distribution]
          [aic: akaike's information criterion for each candidate distribution]
          [aicdiff: difference scores for the AIC estimates]
          [weight: AIC weights (1 is likely, 0 is unlikely)]

        Reference:
        --------------------------------------------------------------------------
        This is based on a matlab code written by Theo Rhodes
        
        """ 
        if np.nanmin(np.array(self.timeseries)) == 0:
            self.timeseries = self.timeseries - np.nanmin(self.timeseries) + .01

        # create histogram to determine plot values
        # note that the original uses hist centers, this uses edges. It may matter
        counts, plotvals_edges = np.histogram(self.timeseries, 50)
        plotvals = np.array([np.mean([plotvals_edges[i], plotvals_edges[i+1]]) for i in range(plotvals_edges.shape[0]-1)])

        distributions = ['normal', 'lognormal', 'exponential', 'pareto', 'boundedpl'] 
        pdfs = [dict(name=dist) for dist in distributions]
        pdfs = defaultdict(dict)
        aicvals = defaultdict(dict)

        # calculate maximum likelihood for core distributions
        # calculate log likelihood value at maximum
        # find k (number of params)
        # generate probability density function using parameters
        kvals = dict()
        for dist in distributions:
            aicvals[dist]['mle'] = self.aicmle(dist)
            aicvals[dist]['nll'] = self.aiclike(aicvals[dist]['mle'], dist)
            kvals[dist] = len(aicvals[dist]['mle'])
            pdfs[dist]['vals'] = self.aicpdf(plotvals, dist, aicvals[dist]['mle'])

        # plot histogram and mle pdf
        # note: only creats the data to make a plot, does not actually generate it
        for dist in distributions:
            scaling = np.sum(counts) / np.sum(pdfs[dist]['vals'])
            aicvals[dist]['plots'] = {}
            aicvals[dist]['plots']['xvals'] = plotvals
            aicvals[dist]['plots']['datay'] = counts
            aicvals[dist]['plots']['aicy'] = pdfs[dist]['vals'] * scaling

        # check for small sample correction
        key_max = max(kvals.keys(), key=(lambda k: kvals[k]))
        bkt = kvals[key_max]
        if ( self.timeseries.shape[0] / bkt < 40 ):
              ssc = 1

        # calculate akaike information criteria
        for dist in distributions:
            aicvals[dist]['aic'] = 2 * aicvals[dist]['nll'] + 2 * kvals[dist]
            if ssc == 1:
                aicvals[dist]['aic'] = aicvals[dist]['aic'] + 2 * kvals[dist] * (kvals[dist] + 1) / (self.timeseries.shape[0] - kvals[dist] -1)

        # calculate AIC differences and akaike weights
        aicmin = np.nanmin([aicvals[dist]['aic'] for dist in distributions])
        for dist in distributions:
            if math.isnan(float(aicvals[dist]['aic'])) != True:
                aicvals[dist]['aicdiff'] = aicvals[dist]['aic'] - aicmin
            else: 
                aicvals[dist]['aicdiff'] = 'nan'

        aicsum = 0
        for dist in distributions:
            if math.isnan(float(aicvals[dist]['aicdiff'])) != True:
                aicsum = aicsum + np.exp(-aicvals[dist]['aicdiff'] / 2)
#         print("aicsum ", aicsum,"\n\n")
        for dist in distributions:
            if math.isnan(float(aicvals[dist]['aicdiff'])) != True:
#                 print(dist, aicvals[dist]['aicdiff'])
                aicvals[dist]['weight'] = np.exp(-aicvals[dist]['aicdiff'] / 2) / aicsum
#                 print("aicvals_weights_if_part ",aicvals[dist]['weight'])
            else:
                aicvals[dist]['weight'] = 0
#                 print("aicvals_weights_else_part ",aicvals[dist]['weight'])

        max_weight_val = np.max([aicvals[dist]['weight'] for dist in distributions])
        
        # Distribution with max AIC weight
#         print("max_weight_val", max_weight_val)
        max_weight = [key for key, value in aicvals.items() if value['weight'] == max_weight_val ]
#         print("\n\n max_weight", max_weight)
        
        if len(max_weight) == 0:
            return int(maxsize)
        else:
            # MLE of the distribution with max AIC weight
            max_weight_params = aicvals[str(max_weight[0])]['mle']
            return aicvals[str(max_weight[0])]['aic']