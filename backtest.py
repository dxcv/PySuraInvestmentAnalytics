# -*- coding: utf-8 -*-
"""
Created on Tue June 1 16:30:41 2018
@author: daniel.velasquez
"""
import numpy as np
import pandas as pd

class backtest(object):
    def __init__(self, series, cf, policy):
        self.series = series
        self.n = series.shape[1]
        self.cf = cf
        self.policy = policy # Pandas DF of weights. Dates x Assets


    def port_values(self):
        eval_dates = self.cf.index
        port_val = pd.Series(np.zeros(len(eval_dates)), index=eval_dates)  # Portfolio value
        port_val.loc[eval_dates[0]] = self.cf.iloc[0]
        delta_pr = self.series.loc[eval_dates].pct_change().dropna()

        for i in range(len(eval_dates)-1):
            port_val.loc[eval_dates[i+1]] = self.cf.loc[eval_dates[i+1]] + np.sum(port_val.loc[eval_dates[i]] * self.policy.loc[eval_dates[i]].values * (1+delta_pr.loc[eval_dates[i+1]]))
        return(port_val)

    def port_returns(self):
        eval_dates = self.cf.index
        port_rets = pd.Series(np.sum(self.policy.values * self.series.loc[eval_dates].pct_change().dropna().values, axis = 1), index = eval_dates[1:])
        return(port_rets)

    def port_total_returns(self):
        '''
        Retorno total sin flujos de caja
        '''
        eval_dates = self.cf.index
        port_rets = np.cumprod(1 + pd.Series(np.sum(self.policy.values * self.series.loc[eval_dates].pct_change().dropna().values, axis = 1), index = eval_dates[1:])).values
        total_rets = pd.Series(np.concatenate((np.array([1]), port_rets)), index=eval_dates)
        return(total_rets)
