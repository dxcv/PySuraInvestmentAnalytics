# -*- coding: utf-8 -*-
"""
Created on Tue June 1 16:30:41 2018

@author: daniel.velasquez
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from .returns import assets as ac
import datetime as dt
from sklearn.linear_model import Lasso, Ridge, RidgeCV, LassoCV
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint as lc

class factors(object):
    def __init__(self, factors_series, assets_series, w_asset = None, w_factor = None, rets_period = 'monthly', alphas=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]):
        self.rets_period = rets_period
        self.factors = ac(factors_series, rets_period=rets_period)
        self.assets = ac(assets_series, rets_period=rets_period)
        self.w_asset = w_asset
        self.w_factor = w_factor
        if w_asset is not None:
            self.assets_returns = self.assets.returns[w_asset.index]
        else:
            self.w_asset = pd.Series(0, index=assets_series.columns)
            self.assets_returns = self.assets.returns
        if w_factor is not None:
            self.factors_returns = self.factors.returns[w_factor.index].loc[self.assets_returns.index]
        else:
            self.w_factor = pd.Series(0, index=factors_series.columns)
            self.factors_returns = self.factors.returns.loc[self.assets_returns.index]
        self.n_assets = len(self.w_asset)
        self.n_factors = len(self.w_factor)
        self.alphas = alphas
        self.Q = self.assets_returns.cov()
        self.Sigma = self.factors_returns.cov()

    def asset_to_factor_exp(self, alphas=None):
        if alphas is None:
            alphas = self.alphas
        regr_cv = RidgeCV(alphas=alphas)
        #regr_cv = LassoCV(alphas=np.arange(0.0001, 1, 0.005))
        A = np.zeros((len(self.w_asset), len(self.w_factor)))
        #factor_exp = np.zeros(factors_returns.shape[1])
        for i in range(len(self.w_asset)):
            asset_id = self.assets_returns.columns[i]
            asset_rets = self.assets_returns.values[:,i]
            model_i = regr_cv.fit(X=self.factors_returns.values, y=asset_rets)
            alpha_i = model_i.alpha_
            #print(alpha_i)
            ridge_reg = Ridge(alpha=model_i.alpha_)
            ridge_reg.fit(X=self.factors_returns.values, y=asset_rets)
            A[i,:] = ridge_reg.coef_
            #factor_exp += ridge_reg.coef_*port[asset_id]
        #factor_exp_norm = pd.Series(factor_exp/factor_exp.sum(), index=factors_returns.columns)
        return A

    def factors_to_assets_obj(self, w_a, w_f, gamma=0.5):
        A = self.asset_to_factor_exp()
        exp_diff = (w_a.T @ A - w_f)
        loss = (1-gamma)*(exp_diff @ exp_diff.T) + gamma * (exp_diff.T @ self.Sigma.values @ exp_diff) + gamma * (w_a.T @ self.Q.values @ w_a)
        return(loss)

    def factors_to_assets_optim(self, gamma=0.5, A=None, activate_cons=True):
        if A is None:
            A = self.asset_to_factor_exp()

        if activate_cons:
            lb = np.zeros(self.n_assets)
            ub = np.ones(self.n_assets)
            bnds = tuple([(i,j) for i,j in zip(lb, ub)])
            cons = lc(np.ones((1,len(lb))), np.ones(1), np.ones(1))
        else:
            bnds = None
            cons = ()
        w_ini = np.ones(self.n_assets)/self.n_assets

        sol = minimize(self.factors_to_assets_obj, w_ini, args=(self.w_factor.values, gamma), bounds=bnds, constraints=cons, method='SLSQP')

        w_a = pd.Series(np.round(sol.x, 5), index=self.w_asset.index)
        return w_a

    def assets_to_factors_obj(self, w_f, w_a, gamma=0.5):
        A = self.asset_to_factor_exp()
        exp_diff = (w_a.T @ A - w_f)
        loss = (1-gamma)*(exp_diff @ exp_diff.T) + gamma * (exp_diff.T @ self.Sigma.values @ exp_diff) + gamma * (w_a.T @ self.Q.values @ w_a)
        return(loss)

    def assets_to_factors_optim(self, gamma=0.5, A=None, activate_cons=True):
        if A is None:
            A = self.asset_to_factor_exp()
        if activate_cons:
            lb = np.zeros(self.n_factors)
            ub = np.ones(self.n_factors)
            bnds = tuple([(i,j) for i,j in zip(lb, ub)])
            cons = lc(np.ones((1,len(lb))), np.ones(1), np.ones(1))
        else:
            bnds = None
            cons = ()
        w_ini = np.ones(self.n_factors)/self.n_factors

        sol = minimize(self.assets_to_factors_obj, w_ini, args=(self.w_asset.values, gamma), bounds=bnds, constraints=cons, method='SLSQP')

        w_f = pd.Series(np.round(sol.x, 5), index=self.w_factor.index)
        return w_f
