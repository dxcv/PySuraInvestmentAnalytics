# -*- coding: utf-8 -*-
"""
Created on Tue June 1 16:30:41 2018

@author: daniel.velasquez
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

class assets(object):
    def __init__(self, series, rets_period = 'monthly'):
        self.series = series
        self.rets_period = rets_period
        self.n = series.shape[1]

    @property
    def returns(self):
        if self.rets_period == 'daily':
            rets = self.series.pct_change().dropna()
        elif self.rets_period == 'monthly':
            rets = self.series.asfreq('M', method='pad').fillna(method="ffill").pct_change().dropna()
        elif self.rets_period == 'quarterly':
            rets = self.series.asfreq('3M', method='pad').fillna(method="ffill").pct_change().dropna()
        return(rets)

    def hf_sigmas(self, remove_cero=True):
        sigmas = pd.DataFrame(np.abs(np.log(np.array(self.series.iloc[1::]) / np.array(self.series.iloc[0:-1]))),
                              index=self.series.index[1:], columns=self.series.columns)
        pos_non_cero = [any(sigmas.iloc[i]!=0) for i in np.arange(sigmas.shape[0])]
        return(sigmas[pos_non_cero])

    def var_simul(self, M = 1e4, N = 10, max_lags = 4):
        '''
        period: returns period,
        M: Number of sample trajectories,
        N: Number of steps ahead.
        '''
        model = VAR(self.returns)
        results = model.fit(max_lags, ic = "aic")
        rets_simul = np.zeros((int(M), int(N), self.n), np.float64)
        for i in range(int(M)):
            rets_simul[i] = results.simulate_var(int(N))
        return(rets_simul)
