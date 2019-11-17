# -*- coding: utf-8 -*-
"""
Created on Tue June 1 16:30:41 2018
@author: daniel.velasquez
"""

import numpy as np
import pandas as pd
import numpy.random as rnd
import math
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.util import varsim
from scipy.optimize import differential_evolution as de
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint as lc
from scipy.optimize import linprog
from scipy.stats import norm
import datetime as dt
from dateutil.relativedelta import relativedelta as rd
from .valuation import bond, get_df
from .valuation import cash_flows as cf
from .risks import hull_white as hw
from numba import jit
import warnings

# Working Directory
# ALM Environment
class alm_env(object):
    def __init__(self, s, asset_data, series, cc_curve, rate_type='efec', asset_names=None, lb=None, ub=None, linCons=None, consLB=None, consUB=None, rets_period='monthly', d=2, q=100, n_bases=7, type_bases='poly', ini_date=dt.date(2007, 1, 1), alpha_limits=[2, 15], util_type='exp', M=2e4, max_lags=2, opt_method='gd'):
        '''
        w0: Initial wealth.
        sigma: Assets returns standard deviation
        mean: Assets returns mean
        linCons: Matrix mxn, m constraints and n assets
        consLB, consUB: Lower and upper bound constraints. Vector dim number of assets
        rets_period: Returns period
        '''

        self.s = s
        if np.any(asset_names is None):
            self.asset_names = list(asset_data.index) # Asset data  limits the investable universe unless explicitily defined through asset_names.
            self.asset_data = asset_data#.loc[series.columns
        else:
            self.asset_names = asset_names
            self.asset_data = asset_data.loc[asset_names]

        #series_ind = [np.any(x==series.columns) for x in self.asset_names]
        self.series = series[self.asset_names]#[self.asset_names[series_ind]]

        self.rets_period = rets_period
        self.per = {'daily':365, 'monthly': 12, 'quarterly': 4, 'yearly':1}[self.rets_period]
        self.cc_curve = cc_curve
        self.rate_type = rate_type
        self.mean_rets = self.returns.mean(axis=0) * self.per
        self.cov = self.returns.cov() * self.per

        self.corr = self.returns.corr()
        self.vols = pd.Series(np.sqrt(np.diag(self.cov)), index=self.asset_names)

        self.n_assets = len(self.asset_names)
        self.lags = None

        if np.any(lb==None):
            self.lb = np.zeros(self.n_assets)
        else:
            self.lb = lb
        if np.any(ub==None):
            self.ub = np.ones(self.n_assets)
        else:
            self.ub = ub

        self.linCons = linCons
        self.consLB = consLB
        self.consUB = consUB

        self.bnds = tuple([(i,j) for i,j in zip(self.lb, self.ub)])
        #con1 = {'type': 'eq', 'fun': lambda x: x.sum() - 1.0}

        #A = np.ones((1,len(lb)))
        #allConsLB = allConsUB = np.ones(1)
        #if (np.any(linCons!=None)):
        #    A = np.vstack([A, linCons])
        #    allConsLB = np.append(allConsLB, consLB)
        #    allConsUB = np.append(allConsUB, consUB)

        #self.cons = lc(A, allConsLB, allConsUB)

        self.cons = lc(np.ones((1,len(self.lb))), np.ones(1), np.ones(1))
        if np.any(linCons!=None):
            self.cons = [self.cons, lc(linCons, consLB, consUB)]
        #self.cons = ([con1, ])
        self.d = d
        self.q = q
        self.n_bases = n_bases
        self.type_bases = type_bases
        self.ini_date = ini_date
        self.alpha_limits = alpha_limits
        self.util_type = util_type
        self.M = M
        self.max_lags = max_lags
        self.opt_method = opt_method
        self.assets_liab_corr = pd.Series()
        self.optimal_policy = None
        self.optimal_surplus_policy = None
        self.fi_assets = self.asset_data[self.asset_data["AssetClass"] == "Renta Fija"][['DurMod', 'YTM']]
    def sample_rets(self, train_size=10e3, test_size=10e3):
        sample_train = rnd.multivariate_normal(self.mean_rets, self.cov, size=train_size)
        sample_test = rnd.multivariate_normal(self.mean_rets, self.cov, size=test_size)
        return({'train': sample_train, 'test': sample_test})

    @property
    def returns(self):
        if self.rets_period == 'daily':
            rets = self.series.pct_change().dropna()
        elif self.rets_period == 'monthly':
            rets = self.series.asfreq('M', method='backfill').pct_change().dropna()
        elif self.rets_period == 'quarterly':
            rets = self.series.asfreq('3M', method='backfill').pct_change().dropna()
        elif self.rets_period == 'yearly':
            rets = self.series.asfreq('12M', method='backfill').pct_change().dropna()
        return(rets)

    @property
    def covExt(self):
        liab_corr = [self.assets_liab_corr[x] if x in self.assets_liab_corr.index else 0 for x in self.asset_names]
        R = np.concatenate((np.append([1], liab_corr).reshape((1,self.n_assets + 1)), np.c_[liab_corr, self.corr.values]), axis=0)
        D = np.diag(np.append(np.array(self.liab_vol), np.sqrt(np.diag(self.cov))))
        covExt_names = np.concatenate((['Liab'], self.cov.index.values))
        Sigma = pd.DataFrame(D @ R @ D, index=covExt_names, columns=covExt_names)
        return(Sigma)

    def gen_cf(self, n_years, ytm):
        cf = ytm * np.ones(n_years + 1)
        cf[0] = 0
        cf[-1] = cf[-1] + 100
        cf_series = pd.Series(cf, index = range(n_years+1))
        return(cf_series)

    def fwd_curve(self, times, t1, cc_curve=None):
        if cc_curve is None:
            cc_curve = self.cc_curve.copy()
        t2 = times + t1
        cc_rate_t1 = np.interp(t1, xp=list(cc_curve.index), fp=list(cc_curve.iloc[:, 0]))
        cc_rates_t2 = np.interp(t2, xp=list(cc_curve.index), fp=list(cc_curve.iloc[:, 0]))

        times_temp = times.copy()
        times_temp[times_temp==0] = 1
        cc_curve_fwd = pd.DataFrame(((1 + cc_rates_t2 * t2) / (1 + cc_rate_t1 * t1) - 1)/times_temp, index = times, columns = ['Rate'])
        return (cc_curve_fwd)

    def cf_obj(self, n_years, ytm, dur_obj, cc_curve=None, rate_type=None):
        '''
        '''
        asset_cf = self.gen_cf(int(n_years), ytm)
        if rate_type is None:
            rate_type = self.rate_type
        if cc_curve is None:
            cc_curve = self.cc_curve.copy()
        cf_inst = cf(asset_cf)
        obj_val = (np.abs(cf_inst.value_df(cc_curve, type=rate_type)["mod_dur"]) - dur_obj)**2
        return(obj_val)

    def varsim(self, coefs, intercept, sig_u, steps=100, initvalues=None, seed=None):
        """
        Simulate VAR(p) process, given coefficients and assuming Gaussian noise
        Parameters
        ----------
        coefs : ndarray
            Coefficients for the VAR lags of endog.
        intercept : None or ndarray 1-D (neqs,) or (steps, neqs)
            This can be either the intercept for each equation or an offset.
            If None, then the VAR process has a zero intercept.
            If intercept is 1-D, then the same (endog specific) intercept is added
            to all observations.
            If intercept is 2-D, then it is treated as an offset and is added as
            an observation specific intercept to the autoregression. In this case,
            the intercept/offset should have same number of rows as steps, and the
            same number of columns as endogenous variables (neqs).
        sig_u : ndarray
            Covariance matrix of the residuals or innovations.
            If sig_u is None, then an identity matrix is used.
        steps : None or int
            number of observations to simulate, this includes the initial
            observations to start the autoregressive process.
            If offset is not None, then exog of the model are used if they were
            provided in the model
        seed : None or integer
            If seed is not None, then it will be used with for the random
            variables generated by numpy.random.
        Returns
        -------
        endog_simulated : nd_array
            Endog of the simulated VAR process
        """
        rs = np.random.RandomState(seed=seed)
        rmvnorm = rs.multivariate_normal
        p, k, k = coefs.shape
        if sig_u is None:
            sig_u = np.eye(k)
        ugen = rmvnorm(np.zeros(len(sig_u)), sig_u, steps)
        result = np.zeros((steps, k))
        if intercept is not None:
            # intercept can be 2-D like an offset variable
            if np.ndim(intercept) > 1:
                if not len(intercept) == len(ugen):
                    raise ValueError('2-D intercept needs to have length `steps`')
            # add intercept/offset also to intial values
            result += intercept
            result[0:p] = self.returns.values[-p:]
            result[p:] += ugen[p:]
        else:
            result[p:] = ugen[p:]

        # add in AR terms
        for t in range(p, steps):
            ygen = result[t]
            for j in range(p):
                ygen += np.dot(coefs[j], result[t-j-1])
        return result

    def simulate_var(self, var_results, steps=None, offset=None, seed=None):
        """
        simulate the VAR(p) process for the desired number of steps
        Parameters
        ----------
        steps : None or int
            number of observations to simulate, this includes the initial
            observations to start the autoregressive process.
            If offset is not None, then exog of the model are used if they were
            provided in the model
        offset : None or ndarray (steps, neqs)
            If not None, then offset is added as an observation specific
            intercept to the autoregression. If it is None and either trend
            (including intercept) or exog were used in the VAR model, then
            the linear predictor of those components will be used as offset.
            This should have the same number of rows as steps, and the same
            number of columns as endogenous variables (neqs).
        seed : None or integer
            If seed is not None, then it will be used with for the random
            variables generated by numpy.random.
        Returns
        -------
        endog_simulated : nd_array
            Endog of the simulated VAR process
        """
        steps_ = None
        if offset is None:
            if var_results.k_exog_user > 0 or var_results.k_trend > 1:
                # if more than intercept
                # endog_lagged contains all regressors, trend, exog_user
                # and lagged endog, trimmed initial observations
                offset = var_results.endog_lagged[:,:var_results.k_exog].dot(
                                                     var_results.coefs_exog.T)
                steps_ = var_results.endog_lagged.shape[0]
            else:
                offset = var_results.intercept
        else:
            steps_ = offset.shape[0]

        # default, but over written if exog or offset are used
        if steps is None:
            if steps_ is None:
                steps = 1000
            else:
                steps = steps_
        else:
            if steps_ is not None and steps != steps_:
                raise ValueError('if exog or offset are used, then steps must'
                                 'be equal to their length or None')
        y = self.varsim(var_results.coefs, offset, var_results.sigma_u, steps=steps, seed=seed)
        return y

    def var_simul(self, M = 1e4, max_lags = 4, policy_date = None):
        '''
        period: returns period,
        M: Number of sample trajectories,
        N: Number of steps ahead.
        '''
        N = self.N

        if policy_date is None:
            model = VAR(self.returns)
        else:
            model = VAR(self.returns[self.ini_date:policy_date])
        results = model.fit(max_lags)
        self.lags = results.k_ar
        rets_simul = np.array(list(map(lambda x: self.simulate_var(results, x), [int(N) + results.k_ar]*int(M))))
        rets_simul_out = rets_simul[:,-N:,:]
        rets_simul_dec_per = np.zeros((int(M), len(self.dec_pers), self.n_assets), np.float64)
        for i in range(len(self.dec_pers)):
            dec_i = self.dec_pers[i]
            rets_simul_dec_per[:,i,:] = np.prod(1 + rets_simul_out[:, (dec_i - self.per*self.dec_per_in_years):dec_i, :], axis = 1)
        return({'per':rets_simul_out, 'dec_per':rets_simul_dec_per})

    def sm_var_simul(self, M = 1e4, max_lags = 4, policy_date = None):
        '''
        var_simul usanado stats model
        period: returns period,
        M: Number of sample trajectories,
        N: Number of steps ahead.
        '''
        N = self.N
        if policy_date is None:
            model = VAR(self.returns)
        else:
            model = VAR(self.returns[self.ini_date:policy_date])
        results = model.fit(max_lags, ic = "aic")
        self.lags = results.k_ar
        rets_simul = np.array(list(map(results.simulate_var, [int(N) + results.k_ar]*int(M))))
        rets_simul_out = rets_simul[:,-N:,:]
        rets_simul_dec_per = np.zeros((int(M), len(self.dec_pers), self.n_assets), np.float64)
        for i in range(len(self.dec_pers)):
            dec_i = self.dec_pers[i]
            rets_simul_dec_per[:,i,:] = np.prod(1 + rets_simul_out[:, (dec_i - self.per):dec_i, :], axis = 1)
        return({'per':rets_simul_out, 'dec_per':rets_simul_dec_per})


    def make_util(self, alpha, a=0, b=1, type='power'):
        '''
        Utility functional.

        Parameters:
        type ('power', 'exp', 'mv'): power, exponential or mean-variance utility function.
        Returns:
        Utility function.
        '''

        if type == 'power':
            def ut(s):
                return ((np.power(s, 1 - alpha) - 1) / (1 - alpha))
        elif type == 'exp':
            def ut(s):
                return (a - b * np.exp(-alpha * s))
        elif  type == 'mv':
            def ut(s):
                return (a - b * np.exp(-alpha * s))
        return (ut)

    def utility_fun(self, s, a1=0, b1=1, alpha_lim=[1.5, 3.5], type='power'):
        '''
        Utility functional

        Parameters:
        type ('power', 'exp', 'mv'): power, exponential or mean-variance utility function.
        Returns:
        Set of Utility function for each alpha.
        '''
        k = len(s)
        alphas = np.interp(s, xp=[np.min(s), np.max(s)], fp=alpha_lim)
        a = np.zeros(k)
        b = np.ones(k)
        a[0] = a1
        b[0] = b1

        if type == 'exp' and k > 1:
            for i in np.arange(k - 1):
                a[i + 1] = a[i] - b[i] * (1 - alphas[i] / alphas[i + 1]) * np.exp(-alphas[i] * s[i])
                b[i + 1] = b[i] * (alphas[i] / alphas[i + 1]) * np.exp((alphas[i + 1] - alphas[i]) * s[i])
        func_list = np.array([self.make_util(alphai, ai, bi, type) for alphai, ai, bi in zip(alphas, a, b)])

        return (func_list)

    def mc_val_fun(self, x, at, lt, ct, util_func, sample_rets, sign=1):
        '''
        x: weights
        wt: Wealth level
        st: Cash flow income
        util_func: Utility print_function
        sample_rets: Sample returns
        '''
        port_rets = sample_rets @ x
        #st_next = (at + ct) * port_rets / lt
        max_s = self.s.max()
        st_next = np.array(list(map(lambda x: min(x, max_s), (at + ct) * port_rets / lt)))
        val = sign * np.mean(util_func(st_next))
        return(val)

    def mc_val_fun_v2(self, x, at, lt, ct, util_func, sample_rets, sign=1):
        '''
        x: weights
        wt: Wealth level
        st: Cash flow income
        util_func: Utility print_function
        sample_rets: Sample returns
        '''
        port_rets = sample_rets @ x
        #st_next = (at + ct) * port_rets / lt
        max_s = self.s.max()
        st_next = np.array(list(map(lambda x: min(x, max_s), (at + ct) * port_rets / lt)))
        val = sign * np.mean(util_func(st_next))
        return(val)

    def mc_val_fun_de(self, x, at, lt, ct, util_func, sample_rets, sign=1):
        '''
        x: weights
        wt: Wealth level
        st: Cash flow income
        util_func: Utility print_function
        sample_rets: Sample returns
        '''
        if np.sum(x)!=1:
            x = x/np.sum(x)
        if np.any(x < self.lb) | np.any(x > self.ub) | np.any(self.linCons @ x < self.consLB) | np.any(self.linCons @ x > self.consUB):
            val = 100
        else:
            port_rets = sample_rets @ x
            #st_next = (at + ct) * port_rets / lt
            max_s = self.s.max()
            st_next = np.array(list(map(lambda x: min(x, max_s), (at + ct) * port_rets / lt)))
            val = sign * np.mean(util_func(st_next))
        return(val)

    def mc_approx_val_fun(self, x, at, lt, ct, sample_rets, coeff, n_bases, type_bases, sign=1):
        '''
        w: weights
        at: Assets
        ct: Cash flow income
        util_func: Utility print_function
        sample_rets: Sample returns
        '''
        port_rets = sample_rets @ x
        #st_next = (at + ct) * port_rets / lt
        max_s = self.s.max()
        st_next = np.array(list(map(lambda x: min(x, max_s), (at + ct) * port_rets / lt)))
        val = sign * np.mean(self.approx_utility_fun(st_next, coeff, n_bases, type_bases))
        # val = sign * np.mean([self.approx_utility_fun((at + ct) * k / lt, coeff, n_bases, type_bases) for k in (1 + sample_rets) @ w])
        return(val)

    def mc_approx_val_fun_de(self, x, at, lt, ct, sample_rets, coeff, n_bases, type_bases, lb, ub, sign=1):
        '''
        w: weights
        at: Assets
        ct: Cash flow income
        util_func: Utility print_function
        sample_rets: Sample returns
        '''
        if np.sum(x)!=1:
            x=x/np.sum(x)
        if any(x > ub) | any(x < lb):
            val = 100
        else:
            port_rets = sample_rets @ x
            #st_next = (at + ct) * port_rets / lt
            max_s = self.s.max()
            st_next = np.array(list(map(lambda x: min(x, max_s), (at + ct) * port_rets / lt)))
            val = sign * np.mean(self.approx_utility_fun(st_next, coeff, n_bases, type_bases))
        return(val)

    def phix(self, x, n_bases, type_bases):
        if type_bases == "poly":
            poly = PolynomialFeatures(n_bases)
            poly.fit_transform(x.reshape((len(x), 1)))
            return(poly.fit_transform(x.reshape((len(x), 1))))
        elif type_bases == "gauss":
            return(np.insert(np.exp(-(x - np.linspace(0, 1, num=n_bases))**2 / 0.2), 0, 1))

    def post_params(self, training, basis_function, q, d, n_bases=9, type_bases='gauss'):
        feat = self.phix(training['x'], n_bases, type_bases)
        D = np.transpose(feat) @ feat + np.identity(feat.shape[1]) * d / q
        w = np.linalg.inv(D) @ (np.transpose(feat) @ training['t'])
        return({'w': w, 'Q': q * D})

    def approx_utility_fun(self, s, coeff, n_bases, type_bases):
        return(self.phix(s, n_bases, type_bases) @ coeff)


    def plot_series(self, asset_name):
        plt.plot(self.series[asset_name])
        plt.title(asset_name)
        plt.xlabel('Periodos')

    def plot_util_fun(self, alpha):
        util_list = self.utility_fun(s=self.s, a1=0, b1=1, alpha_lim=[alpha, alpha], type=self.util_type)
        ut = np.piecewise(self.s, [self.s >= b for b in self.s], util_list)
        plt.plot(self.s, ut, label='lambda:' + str(alpha))
        util_list = self.utility_fun(s=self.s, a1=0, b1=1, alpha_lim=self.alpha_limits, type=self.util_type)
        ut = np.piecewise(self.s, [self.s >= b for b in self.s], util_list)
        plt.plot(self.s, ut, label='lambda:'+ str(self.alpha_limits))
        plt.title('Función de Utilidad')
        plt.xlabel('Tasa de fondeo')
        plt.legend()

    def plot_util_func_array(util_func_array):
        for k in range(util_func_array.shape[1]):
            plt.plot(self.s, util_func_array[:,k], label=k)
        plt.title('Utility Function')
        plt.xlabel('Funding Ratio')
        plt.legend()

    def plot_optimal_policy(self):
        if self.optimal_policy is None:
            Warning("No se ha estimado polícita óptima!")
        else:
            plt.barh(self.optimal_policy.index, 100*self.optimal_policy.values, color='#b5ffb9', edgecolor='white')
            plt.title("Portafolio")
            plt.xlabel("Asignación")

    def plot_curve(self, curve=None, time_in_years=True):
        if curve is None:
            curve = self.cc_curve
        if time_in_years:
            time_axis = curve.index.values
        else:
            time_axis = curve.index.values / 365
        plt.plot(time_axis, curve.iloc[0::, 0].values * 100)
        plt.title("Tasas de Interés")
        plt.xlabel("Plazo")
        plt.ylabel("Tasas")

    def port_risk_assets(self, optim_port=None, assets_liab_corr=None, quant=0.95, normal=False, plot=False,  liab_val=None, index_df=['Retorno Promedio', 'Volatilidad', 'VeR', 'VeRC', 'Vol. Surplus', 'Duración']):
        if optim_port is None:
            optim_port = self.optimal_policy
        if assets_liab_corr is None:
            assets_liab_corr = self.assets_liab_corr

        port_returns = self.returns[optim_port.index.values].values @ optim_port.values
        port_mean = self.mean_rets.values @ optim_port.values
        port_vol = np.sqrt(np.transpose(optim_port.values) @ self.cov.values @ optim_port.values)

        neg_returns = -port_returns
        if normal:
            var = norm.ppf(quant) * port_vol / np.sqrt(self.per)
            cvar = norm.pdf(norm.ppf(quant))/(1 - quant) * port_vol / np.sqrt(self.per)
        else:
            var = np.quantile(neg_returns, quant)
            cvar = neg_returns[neg_returns > var].mean()

        w_l = self.surplus(liab_val=liab_val)['w_l']
        port_ext = np.concatenate(([-w_l], optim_port.values))
        liab_corr = [assets_liab_corr[x] if x in assets_liab_corr.index else 0 for x in self.asset_names]
        R = np.concatenate((np.append([1], liab_corr).reshape((1,self.n_assets + 1)), np.c_[liab_corr, self.corr.values]), axis=0)
        D = np.diag(np.append(np.array(self.liab_vol), np.sqrt(np.diag(self.cov))))
        covExt_names = np.concatenate((['Liab'], self.cov.index.values))
        Sigma = D @ R @ D
        surplus_vol = float(np.sqrt(port_ext.reshape(1,self.n_assets+1) @ Sigma @ port_ext))

        fi_w = optim_port.loc[self.asset_names_cf].values/100
        port_dur = (fi_w @ self.assets_dur)/np.sum(fi_w)
        summ_df = pd.DataFrame(np.round(np.concatenate((np.array([port_mean, port_vol, var, cvar, surplus_vol])*100,[port_dur])), 2), index = index_df, columns=['Variable'])

        if plot:
            ph = plt.hist(x=neg_returns * 100, bins='auto', color='grey', alpha=0.7, rwidth=0.85)
            plt.axvline(x=var * 100,color='red', label=f"VeR({int(quant*100)}%): {np.round(var * 100,2)}%, VeRC:{np.round(cvar * 100,2)}%")
            plt.title('Distribucion de Perdidas')
            plt.legend(loc="upper left")
        return({'neg_returns':neg_returns, 'mean':port_mean, 'vol':port_vol, 'var':var, 'cvar':cvar, 'summ_df':summ_df, 'surplus_vol':surplus_vol, 'dur':port_dur})

    def liab_risk(self, cc_curve=None, delta_cc=0.01, rate_type=None):
        if self.liab_cf is None:
            raise ValueError("Liabilities cashflows not available.")
        if rate_type is None:
            rate_type = self.rate_type
        if cc_curve is None:
            cc_curve = self.cc_curve.copy()
        cfs = np.sum(self.liab_cf, axis=1)
        cf_inst = cf(cfs)
        key_rates = cf_inst.key_rates(cc_curve, delta_cc)
        return({'risk':cf_inst.value_df(cc_curve, type=rate_type), 'key_rates':key_rates})

    def surplus(self, cc_curve=None, rate_type=None, liab_val=None):
        if rate_type is None:
            rate_type = self.rate_type
        if cc_curve is None:
            cc_curve = self.cc_curve.copy()

        if liab_val is None and self.liab_cf is None:
            raise ValueError("liab_val cannot be estimated.")
        elif liab_val is None and self.liab_cf is not None:
            cfs = np.sum(self.liab_cf, axis=1)
            liab_val = cf(cfs).value_df(cc_curve, type=rate_type)['price']
        surp = self.capital - liab_val
        w_a = self.capital/(self.capital + liab_val)
        w_l = liab_val/(self.capital + liab_val)
        return({'liab_val':liab_val, 'val':surp, 'rate':self.capital/liab_val, 'w_a':w_a, 'w_l':w_l})

    def summary_dfs(self, group_names=None, surplus_index=['Activos', 'Pasivos', 'Surplus', 'Tasa de Fondeo'], \
    liab_index=['Valor Presente', 'Dur. Mac.', 'Dur. Mod.', 'Convex.', 'YTM'], assets_colnames=None,  liab_val=None):
        '''
        Summary data frames
        '''
        assets_lim_df = None
        group_lim_df = None
        liab_df = None
        ## Surplus
        surplus_df = pd.DataFrame(np.round([self.capital, *self.surplus(liab_val=liab_val).values()], 2)[0:4], \
        index=surplus_index, columns=['Variable'])

        ## Exposure
        if self.lb is not None and self.ub is not None:
            assets_lim_df = pd.DataFrame(np.vstack(([100*self.lb], [100*self.ub])).T, index=self.asset_names, \
            columns=['LB', 'UB'])
        if self.consLB is not None and self.consUB is not None:
            group_lim_df = pd.DataFrame(np.vstack((100*self.consLB, 100*self.consUB)).T, index=group_names, \
            columns=['LB', 'UB'])
        ## Assets
        assets_df = self.asset_data.loc[self.asset_names][['AssetClass', 'Currency', 'DurMod']]
        del assets_df.index.name
        if assets_colnames is not None:
            assets_df.columns = assets_colnames
        ## Liab
        if self.liab_cf is not None:
            liab_df = pd.DataFrame(np.round([*self.liab_risk()['risk'].values()], 3), index=liab_index, columns=['Obligaciones'])

        return({'surplus_df':surplus_df, 'assets_df':assets_df, 'liab_df':liab_df, 'assets_lim_df':assets_lim_df, 'group_lim_df':group_lim_df})

# ALM Environment for agent with fixed income Assets
class alm_env_wm(alm_env):
    def __init__(self, cf, s, asset_data, series, cc_curve, rate_type, liab_rate, liab_vol, asset_names, lb, ub, linCons=None, consLB=None, consUB=None, rets_period='monthly', dec_per_in_years=1, d=2, q=100, n_bases=7, type_bases='poly', ini_date=dt.date(2007, 1, 1), alpha_limits=[2, 15], util_type='exp', M=2e4, max_lags=2, opt_method='gd'):
        super().__init__(s, asset_data, series, cc_curve, rate_type, asset_names, lb, ub, linCons, consLB, consUB, rets_period, d, q, n_bases, type_bases, ini_date, alpha_limits, util_type, M, max_lags, opt_method)

        self.series = series[self.asset_names]
        self.cf = cf
        self.liab_cf = pd.DataFrame(cf['Liab'])
        self.time_hor = self.cf.index[-1]
        self.dec_per_in_years = dec_per_in_years
        self.liab_gr_rate = (1+liab_rate)**self.dec_per_in_years
        cc_rates = np.interp(self.cf.index, xp=list(self.cc_curve.index), fp=list(self.cc_curve.iloc[:, 0]))
        disc_factors = np.array(1/(1+cc_rates*self.cf.index))
        self.capital = np.sum(self.cf['CF'].values * disc_factors)

        self.N = self.per * self.time_hor
        self.dec_pers = np.arange(self.per*dec_per_in_years, self.N + 1, self.per*dec_per_in_years)
        pers_in_y = (self.dec_pers - self.per * self.dec_per_in_years)/self.per
        self.dec_pers_in_years = pers_in_y.astype(int)
        self.rets_simul = None
        self.liab_vol = liab_vol
        self.asset_names_cf = self.fi_assets.index
        self.assets_dur = self.fi_assets.loc[self.asset_names_cf]['DurMod'].values

    def plot_cf(self, plot_assets=True):
        if plot_assets:
            plt.bar(self.cf.index, -self.cf["Liab"].values, color='#f9bc86', edgecolor='white')
            plt.bar(self.cf.index, self.cf["CF"].values, color='#b5ffb9', edgecolor='white')
            plt.title("Flujos de Caja")
        else:
            plt.bar(self.cf.index, self.cf["Liab"].values, color='#f9bc86', edgecolor='white')
            plt.title("Flujos de Caja Obligaciones")
        plt.xlabel("Tiempo (años)")

    def policy(self, policy_date, dur_interv=None, run_simul=False, simul_model=None, cc_curve=None, dec_per_in_years=None, alpha_lim=None, util_type=None):
        if cc_curve is None:
            cc_curve = self.cc_curve.copy()
        if alpha_lim is not None:
            self.alpha_limits = alpha_lim
        if util_type is not None:
            self.util_type = util_type
        ls = len(self.s)
        cc_rates, df = get_df(cc_curve, times=self.cf.index, type=self.rate_type)
        a0 = np.sum(df.loc[self.cf.index].values * self.cf["CF"].values)
        l0 = np.sum(df.loc[self.cf.index].values * self.cf["Liab"].values)
        s0 = a0/l0

        series_train = self.series[self.ini_date:policy_date]
        ut = np.zeros(ls)
        weights = np.zeros((ls, self.n_assets, len(self.dec_pers)))
        util_func_array = np.zeros((ls, len(self.dec_pers)))
        training = None
        util_list = self.utility_fun(s=self.s, a1=0, b1=1, alpha_lim=self.alpha_limits, type=self.util_type)

        if self.rets_simul is None or run_simul:
            if(simul_model=='sm'):
                self.rets_simul = self.sm_var_simul(M = self.M, max_lags = self.max_lags, policy_date=policy_date)
            else:
                self.rets_simul = self.var_simul(M = self.M, max_lags = self.max_lags, policy_date=policy_date)
            print('finished simulation!')

        if dec_per_in_years is None:
            rets_simul_dec_per = self.rets_simul['dec_per']
            rets_simul_train = rets_simul_dec_per[0:int(self.M/2),:,:]
            rets_simul_test = rets_simul_dec_per[int(self.M/2):int(self.M),:,:]
        else:
            self.dec_per_in_years = dec_per_in_years
            self.dec_pers = np.arange(self.per*dec_per_in_years, self.N + 1, self.per*dec_per_in_years)
            pers_in_y = (self.dec_pers - self.per * dec_per_in_years)/self.per
            self.dec_pers_in_years = pers_in_y.astype(int)
            rets_simul_dec_per = np.zeros((int(self.M), len(self.dec_pers), self.n_assets), np.float64)
            for i in range(len(self.dec_pers)):
                dec_i = self.dec_pers[i]
                rets_simul_dec_per[:,i,:] = np.prod(1 + self.rets_simul['per'][:, (dec_i - self.per*self.dec_per_in_years):dec_i, :], axis = 1)
            rets_simul_train = rets_simul_dec_per[0:int(self.M/2),:,:]
            rets_simul_test = rets_simul_dec_per[int(self.M/2):int(self.M),:,:]
        ##
        cons = self.cons.copy()
        if dur_interv is not None:
            liab_dur = np.abs(self.liab_risk()['risk']['mac_dur'])
            dur_vec = np.zeros(self.n_assets)
            dur_vec[[self.asset_names.index(x) for x in self.asset_names_cf.values]] = self.assets_dur
            dur_lb = liab_dur - dur_interv
            dur_ub = liab_dur + dur_interv

            dur_cons = {'type': 'ineq', 'fun' : lambda w: np.array([(dur_vec @ w) / np.sum(w[dur_vec!=0]) - dur_lb,-(dur_vec @ w) / np.sum(w[dur_vec!=0]) + dur_ub]),
            'jac' : lambda w: np.array([(np.sum(w[dur_vec!=0]) * dur_vec - (dur_vec @ w) * w * (dur_vec!=0)) / np.sum(w[dur_vec!=0])**2,
                                       (-np.sum(w[dur_vec!=0]) * dur_vec + (dur_vec @ w) * w * (dur_vec!=0)) / np.sum(w[dur_vec!=0])**2])}
            cons.append(dur_cons)

        w_ini = self.lb + (1-np.sum(self.lb))*(self.ub - self.lb)/np.sum(self.ub - self.lb)
        k = 0
        for j in reversed(self.dec_pers_in_years):
            k += 1
            lt = l0 / df.loc[j]
            ct = 0
            if j == self.dec_pers_in_years[-1]:
                for i in range(ls):
                    si = self.s[i]
                    ai = si * lt
                    if self.opt_method=='gd':
                        sol = minimize(self.mc_val_fun, w_ini, args=(ai, lt*self.liab_gr_rate, ct, util_list[i], rets_simul_train[:,-k,:], -1), bounds=self.bnds, constraints=cons, method='SLSQP')
                        ut[i] = self.mc_val_fun(sol.x, ai, lt*self.liab_gr_rate, ct, util_list[i], rets_simul_test[:,-k,:])
                    elif self.opt_method=='de':
                        sol = de(self.mc_val_fun_de, bounds=self.bnds, popsize=50, strategy='best1bin', args=(ai, lt*self.liab_gr_rate, ct, util_list[i], rets_simul_train[:,-k,:], self.lb, self.ub, -1))
                        ut[i] = self.mc_val_fun_de(sol.x, ai, lt*self.liab_gr_rate, ct, util_list[i], rets_simul_test[:,-k,:], self.lb, self.ub)
                    weights[i, :, -k] = sol.x/np.sum(sol.x)
                training = {'x': self.s, 't': ut}
            else:
                if j == 0:
                    params = self.post_params(training, self.phix, self.q, self.d, n_bases=self.n_bases, type_bases=self.type_bases)
                    #rets_simul['dec_per'][:,j,:]
                    if self.opt_method=='gd':
                        sol = minimize(self.mc_approx_val_fun, w_ini, args=(a0, l0*self.liab_gr_rate, 0, rets_simul_train[:,-k,:], params['w'], self.n_bases, self.type_bases, -1), bounds=self.bnds, constraints=cons, method='SLSQP')
                    elif self.opt_method=='de':
                        sol = de(self.mc_approx_val_fun_de, bounds=self.bnds, popsize=50, strategy='best1bin', args=(a0, l0*self.liab_gr_rate, 0,  rets_simul_train[:,-k,:], params['w'], self.n_bases, self.type_bases, self.lb, self.ub, -1))
                    weights[:, :, -k] = sol.x/np.sum(sol.x)
                    optimal_policy = pd.Series(sol.x/np.sum(sol.x), index=self.asset_names)
                    ut = self.mc_approx_val_fun(sol.x/np.sum(sol.x), a0, l0*self.liab_gr_rate, ct, rets_simul_dec_per[:,-k,:], params['w'], self.n_bases, self.type_bases)
                else:
                    ut = np.zeros(ls)
                    params = self.post_params(training, self.phix, self.q, self.d, n_bases=self.n_bases, type_bases=self.type_bases)
                    for i in range(ls):
                        si = self.s[i]
                        ai = si * lt
                        if self.opt_method=='gd':
                            sol = minimize(self.mc_approx_val_fun, w_ini, args=(ai, lt*self.liab_gr_rate, 0, rets_simul_train[:,-k,:], params['w'], self.n_bases, self.type_bases, -1), bounds=self.bnds, constraints=cons, method='SLSQP')
                        elif self.opt_method=='de':
                            sol = de(self.mc_approx_val_fun_de, bounds=self.bnds, popsize=20, strategy='best1bin', args=(ai, lt*self.liab_gr_rate, 0,  rets_simul_train[:,-k,:], params['w'], self.n_bases, self.type_bases, self.lb, self.ub, -1))
                        ut[i] = self.mc_approx_val_fun(sol.x/np.sum(sol.x), ai, lt*self.liab_gr_rate, ct,
                        rets_simul_test[:,-k,:], params['w'], self.n_bases, self.type_bases)
                        weights[i, :, -k] = sol.x/np.sum(sol.x)
                    training = {'x': self.s, 't': ut}
            util_func_array[:,-k] = ut
            print('time: ' + str(j))
        return({'s0':s0, 'l0':l0, 'policy':optimal_policy, 'simul_decisions':weights, 'ut':ut, 'rets_simul':rets_simul_dec_per, 'util_func_array':util_func_array})

    def multiple_policy(self, eval_dates):
        ls = len(self.s)
        cf_index = np.array(self.cf["CF"].index)
        cf_index[cf_index==0]=1
        cc_rates = np.interp(cf_index, xp=list(cc_curve.index), fp=list(cc_curve.iloc[:, 0]))
        df = pd.Series(1/np.power(list(1 + cc_rates), list(cf_index)), index=self.cf.index)
        dec_dates = eval_dates[0:-1]
        prices_dec_dates = self.series.loc[dec_dates]
        policy = pd.DataFrame(np.zeros((len(dec_dates), len(self.asset_names))), index=dec_dates, columns=self.asset_names)
        util_list = self.utility_fun(s=self.s, a1=0, b1=1, alpha_lim=self.alpha_limits, type=self.util_type)

        l_dec_dates = len(dec_dates)
        port_value = pd.Series(np.zeros(len(eval_dates)), index=eval_dates)  # Portfolio value
        port_returns = pd.Series(np.zeros(len(eval_dates)), index=eval_dates)
        port_value.loc[dec_dates[0]] = self.cf["CF"].loc[0]
        w_ini = self.lb + (1-np.sum(self.lb))*(self.ub - self.lb)/np.sum(self.ub - self.lb)
        time_hor = self.time_hor
        for di in range(l_dec_dates):
            dec_i = dec_dates[di]
            a0 = np.sum(df.loc[self.cf.index[self.cf.index>di] - di].values * self.cf["CF"].values[self.cf.index>di]) + port_value.loc[dec_i]
            l0 = np.sum(df.loc[self.cf.index[self.cf.index>=di] - di].values * self.cf["Liab"].values[self.cf.index>=di])
            s0 = a0/l0

            series_train = self.series[dt.date(2007, 1, 1):dec_i]
            ut = np.zeros(ls)
            weights = np.zeros((ls, self.n_assets, len(self.dec_pers)))
            training = None
            rets_simul = self.var_simul(M = self.M, max_lags = self.max_lags, policy_date=dec_i)
            rets_simul_train = rets_simul['dec_per'][0:int(self.M/2),:,:]
            rets_simul_test = rets_simul['dec_per'][int(self.M/2):int(self.M),:,:]

            time_hor = time_hor - di
            N = self.per * time_hor
            dec_pers = np.arange(self.per* self.dec_per_in_years, N + 1, self.per* self.dec_per_in_years)
            pers_in_y = (dec_pers - self.per * self.dec_per_in_years)/(self.per * self.dec_per_in_years)
            dec_pers_in_years = pers_in_y.astype(int)


            for j in reversed(dec_pers_in_years):
                lt = l0 / df.loc[j]
                ct = 0
                if j == (time_hor - 1):
                    for i in range(ls):
                        si = self.s[i]
                        ai = si * lt
                        if self.opt_method=='gd':
                            sol = minimize(self.mc_val_fun, w_ini, args=(ai, lt, 0, util_list[i], rets_simul_train[:,j,:], -1), bounds=self.bnds, constraints=self.cons)
                        elif self.opt_method=='de':
                            sol = de(self.mc_val_fun_de, bounds=self.bnds, popsize=50, strategy='best1bin', args=(ai, lt, ct, util_list[i], rets_simul_train[:,j,:], self.lb, self.ub, -1))
                        ut[i] = self.mc_val_fun_de(sol.x, ai, lt, ct, util_list[i], rets_simul_test[:,j,:], self.lb, self.ub)
                        weights[i, :, j] = sol.x/np.sum(sol.x)
                    training = {'x': self.s, 't': ut}
                if j == 0:
                    params = self.post_params(training, self.phix, self.q, self.d, n_bases=self.n_bases, type_bases=self.type_bases)
                    if self.opt_method=='gd':
                        sol = sol = minimize(self.mc_approx_val_fun, w_ini, args=(a0, l0, 0, rets_simul_train[:,j,:], params['w'], self.n_bases, self.type_bases, -1), bounds=self.bnds, constraints=self.cons)
                    elif self.opt_method=='de':
                        sol = de(self.mc_approx_val_fun_de, bounds=self.bnds, popsize=100, strategy='best1bin', args=(a0, l0, 0,  rets_simul_train[:,j,:], params['w'], self.n_bases, self.type_bases, self.lb, self.ub, -1))
                    ut = self.mc_approx_val_fun(sol.x/np.sum(sol.x), a0, l0, 0, rets_simul_test[:,j,:], params['w'], self.n_bases, self.type_bases)
                    weights[:, :, j] = sol.x/np.sum(sol.x)
                    policy.iloc[di,:] = sol.x/np.sum(sol.x)
                else:
                    ut = np.zeros(ls)
                    params = self.post_params(training, self.phix, self.q, self.d, n_bases=self.n_bases, type_bases=self.type_bases)
                    for i in range(ls):
                        si = self.s[i]
                        ai = si * lt
                        if self.opt_method=='gd':
                            sol = minimize(self.mc_approx_val_fun, w_ini, args=(ai, lt, 0, rets_simul_train[:,j,:], params['w'], self.n_bases, self.type_bases, -1), bounds=self.bnds, constraints=self.cons)
                        elif self.opt_method=='de':
                            sol = de(self.mc_approx_val_fun_de, bounds=self.bnds, popsize=20, strategy='best1bin', args=(ai, lt, 0,  rets_simul_train[:,j,:], params['w'], self.n_bases, self.type_bases, self.lb, self.ub, -1))
                        ut[i] = self.mc_approx_val_fun(sol.x/np.sum(sol.x), ai, lt, 0, rets_simul_test[:,j,:], params['w'], self.n_bases, self.type_bases)
                        weights[i, :, j] = sol.x/np.sum(sol.x)
                    training = {'x': self.s, 't': ut}
                print('time: ' + str(j))

            prices_t0 = self.series.loc[dec_i].values
            prices_t1 = self.series.loc[eval_dates[di+1]].values
            port_value.loc[eval_dates[di+1]] = np.sum(port_value.loc[dec_i] * policy.iloc[di,:].values * prices_t1/prices_t0) + self.cf["CF"].loc[di+1]

        port_returns = pd.Series(np.sum(policy.values * self.series.loc[eval_dates].pct_change().dropna().values, axis = 1), index = eval_dates[1:])
        return({'policy':policy, 'port_returns':port_returns, 'port_value':port_value})

class alm_env_fi(alm_env):
    def __init__(self, capital, assets_cf, liab_cf, s, asset_data, series, cc_curve, rate_type, liab_rate, liab_vol, asset_names, lb, ub, linCons=None, consLB=None, consUB=None, rets_period='monthly', dec_per_in_years=1, d=2, q=100, n_bases=7, type_bases='poly', ini_date=dt.date(2007, 1, 1), alpha_limits=[2, 15], util_type='exp', M=2e4, max_lags=2, opt_method='gd'):
        super().__init__(s, asset_data, series, cc_curve, rate_type, asset_names, lb, ub, linCons, consLB, consUB, rets_period, d, q, n_bases, type_bases, ini_date, alpha_limits, util_type, M, max_lags, opt_method)
        self.capital = capital
        self._assets_cf = assets_cf

        if capital is None:
            self.capital = -float(liab_cf.loc[0])
        else:
            self.capital = capital

        self.liab_cf = liab_cf
        if self.liab_cf is not None:
            self.liab_cf.loc[0] = 0

        self.dec_per_in_years = dec_per_in_years
        self.liab_rate = liab_rate # Annualized liability rate of return
        self.liab_vol = liab_vol
        ytm_from_cc = True
        if ytm_from_cc:
            times = np.arange(1, int(self.fi_assets['YTM'].max())+1, 1)
            cc_rates = np.interp(times, xp=list(self.cc_curve.index), fp=list(self.cc_curve.iloc[:, 0]))
            disc_factors = np.array(1/(1+cc_rates*times))
            ytm = (1 - disc_factors)/np.cumsum(disc_factors)

            fi_ytm = np.zeros(len(self.fi_assets['DurMod']))
            for i in range(len(self.fi_assets['DurMod'])):
                if self.fi_assets['DurMod'][i] < 1:
                    fi_ytm[i] = np.interp(self.fi_assets['DurMod'][i], xp=list(cc_curve.index), fp=list(cc_curve.iloc[:, 0]))*100
                else:
                    fi_ytm[i] = np.interp(self.fi_assets['DurMod'][i], xp=list(times), fp=list(ytm))*100
            self.fi_assets['YTM'] = fi_ytm


        self.calc_cf_ind = True
        self.asset_names_cf = self.assets_cf.columns
        self._assets_dur = self.fi_assets.loc[self.asset_names_cf]['DurMod'].values

    @property
    def assets_cf(self):
        '''
        It not available, simulate bonds cash flows using Duration and assuming par price.
        '''
        if self.calc_cf_ind:
            if self._assets_cf.shape[0] == 0:
                fi_assets_simul_cf = self.fi_assets
            else:
                fi_assets_simul_cf = self.fi_assets.iloc[[np.any(x not in self._assets_cf.columns) for x in self.fi_assets.index]]

            if fi_assets_simul_cf.shape[0] > 0:
                for i in range(fi_assets_simul_cf.shape[0]):
                    sol = de(self.cf_obj, bounds=((1, 30),), popsize=50, strategy='best1bin', args=(fi_assets_simul_cf.iloc[i]['YTM'], fi_assets_simul_cf.iloc[i]['DurMod']))
                    cf_series = self.gen_cf(int(sol.x), fi_assets_simul_cf.iloc[i]['YTM'])
                    self._assets_cf = self._assets_cf.merge(cf_series.to_frame(fi_assets_simul_cf.index[i]), left_index=True, right_index=True, how='outer')
            self.calc_cf_ind = False
        return (self._assets_cf)

    @property
    def assets_dur(self):
        return(self._assets_dur)

    @assets_dur.setter
    def assets_dur(self, use_cf):
        if use_cf:
            self._assets_dur = [np.abs(np.round(self.port_risk(optim_port)['fi_assets_risk'][x]['mod_dur'], 3)) for x in self.asset_names_cf]
        else:
            self._assets_dur = self.fi_assets.loc[self.asset_names_cf]['DurMod'].values

    def assets_cash_cf(self, assets_prices, optim_port):
        assets_cash_cf = self.assets_cf.copy()
        assets_cash = float(np.abs(self.capital)) * optim_port.loc[self.asset_names_cf] / (assets_prices / 100)
        for ai in self.asset_names_cf:
            assets_cash_cf[ai] = self.assets_cf[ai] * assets_cash[ai] / 100
        return(assets_cash_cf)

    def cf_profile(self, assets_prices, optim_port, reinvest=False, short_term_rate=0.03, reinvest_asset=None):
        '''
        Reinvestment Model:
        * We assumen constant cc curve, which means that bond prices are constant.
        * Deficits are funded for 1 periodo at the short term rate.
        * Superavits are invested at the reinvest_asset defined or to the short term rate.
        * If reinvest_asset maturity is beyond the max_per, income is reinvested at the short term rate.
        '''
        cfs = self.assets_cash_cf(assets_prices, optim_port)
        liab = -self.liab_cf.iloc[:,0]
        liab_max_per = liab.index.max()
        assets = pd.DataFrame(np.sum(cfs, axis=1).values, index=cfs.index).iloc[:,0]
        fund_cost = 0
        cum_fund_cost = 0
        liab_total = liab.copy()
        if reinvest:
            max_per = int(assets.index.append(liab.index).max())
            if reinvest_asset is not None:
                reinvest_asset_cf = self.assets_cf[reinvest_asset].dropna()
            if len(short_term_rate)==1:
                short_term_rate = short_term_rate * np.ones(max_per)
            fund_cost = pd.Series(0, index=range(max_per+1))
            liab_total = pd.Series(0, index=range(max_per+1))
            liab_total.loc[liab.index] = liab.values
            for i in np.arange(0, max_per):
                if i > liab_max_per:
                    liab_i = 0
                else:
                    liab_i = liab[i]
                cf_net_i = assets[i] + liab_i
                if cf_net_i > 0:
                    assets[i] -= cf_net_i
                    if reinvest_asset is None or (i+reinvest_asset_cf.index.max())>(max_per - 1):
                        assets[i+1] += cf_net_i * (1 + short_term_rate[i])
                    else:
                        nom = float(cf_net_i / (assets_prices[reinvest_asset] / 100))
                        assets[i+reinvest_asset_cf.index] += pd.Series(nom * reinvest_asset_cf.values / 100, index=i+reinvest_asset_cf.index)
                if cf_net_i < 0:
                    assets[i] += -cf_net_i # Se financia el descalce.
                    liab_total[i+1] += cf_net_i * (1 + short_term_rate[i])
                    fund_cost[i+1] = np.abs(cf_net_i * short_term_rate[i])
            cum_fund_cost = np.cumsum(fund_cost)
        return({'assets':assets, 'liab':liab_total, 'fund_cost':fund_cost, 'cum_fund_cost':cum_fund_cost})

    def plot_cf(self, assets_prices=None, optim_port=None, plot_assets=False):
        liab_cfs = self.liab_cf.iloc[:,0]
        if plot_assets:
            if optim_port is None:
                optim_port = self.optimal_policy
            if  assets_prices is None or optim_port is None:
                raise ValueError('asset_prices and optim_port cannot be None.')
            cfs = self.assets_cash_cf(assets_prices, optim_port)
            plt.bar(cfs.index, np.sum(cfs, axis=1).values, color='#b5ffb9', edgecolor='white')
            plt.bar(liab_cfs.index, -liab_cfs.values, color='#f9bc86', edgecolor='#f9bc86')
            plt.title("Perfil de Activos y Pasivos")
            plt.xlabel("Tiempo (años)")
        else:
            plt.bar(liab_cfs.index, liab_cfs.values, color='#f9bc86', edgecolor='#f9bc86')
            plt.title("Perfil de Pasivos")
            plt.xlabel("Tiempo (años)")


    def bond_risk(self, asset_name, cc_curve=None, curr_date=dt.date.today(), conv='actual/360', rate_type=None):
        if rate_type is None:
            rate_type = self.rate_type
        if cc_curve is None:
            cc_curve = self.cc_curve.copy()
        asset_cf = self.assets_cf[asset_name].dropna()[1::]
        cpn = asset_cf.iloc[0] / 100
        matur_date = curr_date + rd(years=+asset_cf.index[-1])
        freq = int(1/np.diff(asset_cf.index)[1])
        asset_inst = bond(cpn, matur_date, freq, conv)
        return(asset_inst.value_df(curr_date, cc_curve, type=rate_type))

    def cf_risk(self, asset_name, cc_curve=None, rate_type=None):
        if rate_type is None:
            rate_type = self.rate_type
        asset_cf = self.assets_cf[asset_name].dropna()
        if cc_curve is None:
            cc_curve = self.cc_curve.copy()
        cf_inst = cf(asset_cf)
        return(cf_inst.value_df(cc_curve, type=rate_type))

    def cf_matrix_risk(self, cc_curve=None, rate_type=None):
        if rate_type is None:
            rate_type = self.rate_type
        assets_risk = {x:self.cf_risk(x, cc_curve, rate_type) for x in self.asset_names_cf}
        return(assets_risk)

    def group_by(self, optim_port, factor):
        factor_weight = pd.concat([optim_port.rename('Weight'), self.asset_data[factor]], axis=1, sort=False).groupby(factor).sum()
        return(factor_weight)

    def port_risk(self, optim_port=None, assets_par_price=None, port_asset_names=None, cc_curve=None, delta_cc=0.01, rate_type=None):
        '''
        optim_port: Pandas Series of portfolio weights,
        assets_par_price: Assets with par price,
        port_asset_names: Assets used to group portfolio cash flows to estimate risk measures, including key rates. If any asset is left
        out, the market value of the portfolio will not match the present value of cash flows of all assets. E.g. if Real Estate is not
        included its cf will not be considere to estimate key rates under the assumption its cf are not affected by the cc curve.
        '''
        if optim_port is None and self.optimal_policy is None:
            raise ValueError('portfolio not available!')

        if optim_port is None:
            optim_port = self.optimal_policy
        if port_asset_names is None:
            port_asset_names = self.asset_names_cf
        if rate_type is None:
            rate_type = self.rate_type
        if cc_curve is None:
            cc_curve = self.cc_curve.copy()


        assets_risk = self.cf_matrix_risk(cc_curve)
        assets_prices = pd.Series([assets_risk[x]['price'] if x in assets_risk else 100 for x in self.asset_names], index=self.asset_names)
        if assets_par_price is not None:
            assets_prices[assets_par_price] = 100

        nom_per_asset = float(np.abs(self.capital)) * optim_port / (assets_prices / 100)
        port_val = np.sum(nom_per_asset * assets_prices / 100)

        assets_cash = self.assets_cash_cf(assets_prices, optim_port)
        asset_cf = np.sum(assets_cash[port_asset_names], axis=1)
        cf_inst = cf(asset_cf)
        port_risk = cf_inst.value_df(cc_curve, type=rate_type)
        key_rates = cf_inst.key_rates(cc_curve, delta_cc)

        return({'nom_per_asset':nom_per_asset, 'port_value':port_val, 'fi_assets_risk':assets_risk, 'assets_prices':assets_prices, 'port_risk':port_risk, 'key_rates':key_rates})


    def dur_obj(self, weights, dur_type='mod_dur', assets_par_price=None, port_asset_names=None, cc_curve=None):
        '''
        policy: Portfolio weights
        '''
        policy = pd.Series(weights, self.asset_names)
        summ_risk = self.port_risk(policy, assets_par_price, port_asset_names, cc_curve)
        obj_val = (summ_risk['port_risk'][dur_type]-self.liab_risk()['risk'][dur_type])**2
        return(obj_val)

    def policy_match_dur(self, dur_type='mod_dur', assets_par_price=None, port_asset_names=None, cc_curve=None):
        w_ini = self.lb + (1-np.sum(self.lb))*(self.ub - self.lb)/np.sum(self.ub - self.lb)
        sol = minimize(self.dur_obj, w_ini, args=(dur_type, assets_par_price, port_asset_names, cc_curve), bounds=self.bnds, constraints=self.cons)
        self.optimal_policy = pd.Series(sol.x, index=self.asset_names)

        return(self.optimal_policy)

    def policy_match_cf(self, assets_par_price=None, cc_curve=None, round_digits=3):
        assets_risk = self.cf_matrix_risk(cc_curve)
        assets_prices = pd.Series([assets_risk[x]['price'] for x in self.asset_names], index=self.asset_names)
        if assets_par_price is not None:
            assets_prices[assets_par_price] = 100
        assets_cf = self.assets_cf[assets_prices.index].fillna(0).values
        sol = linprog(c=assets_prices.values/100, A_ub= -assets_cf/100, b_ub=-self.liab_cf.values[:,0], method='interior-point')
        capital = sol.fun
        self.optimal_policy = pd.Series(np.round(sol.x/capital, round_digits), index=assets_prices.index)
        return({'capital':capital, 'policy':self.optimal_policy})

    def fc_obj_de(self, x, lb, ub, assets_prices, short_term_rate=[0.03], reinvest_asset=None):
        if np.sum(x)!=1:
            x = lb + (1-np.sum(lb)) * (x - lb)/np.sum(x - lb)
        if any(x > ub) | any(x < lb):
            val = 10000

        policy = pd.Series(x, self.asset_names)
        obj_val = self.cf_profile(assets_prices, policy, reinvest=True, short_term_rate=short_term_rate, reinvest_asset=reinvest_asset)['cum_fund_cost'].values[-1]
        return(obj_val)

    def policy_min_fc(self, assets_par_price=None, short_term_rate=[0.03], reinvest_asset=None, cc_curve=None):
        assets_risk = self.cf_matrix_risk(cc_curve)
        assets_prices = pd.Series([assets_risk[x]['price'] for x in self.asset_names], index=self.asset_names)
        if assets_par_price is not None:
            assets_prices[assets_par_price] = 100
        sol = de(self.fc_obj_de, bounds=self.bnds, popsize=100, strategy='best1bin', args=(self.lb, self.ub, assets_prices, short_term_rate, reinvest_asset))
        optimal_policy = pd.Series(self.lb + (1-np.sum(self.lb)) * (sol.x - self.lb)/np.sum(sol.x - self.lb), index=self.asset_names)
        return(optimal_policy)

    def suplus_obj(self, x, w_l, Sigma):
        w = np.append([-w_l], x)
        val = w.T @ Sigma @ w
        return(val)

    def surplus_obj_de(self, x, Sigma, risk_obj, sign=-1):
        if np.sum(x)!=1:
            x = x/np.sum(x)
        if np.any(x < self.lb) | np.any(x > self.ub) | np.any(self.linCons @ x < self.consLB) | np.any(self.linCons @ x > self.consUB | (x.T @ Sigma @ x) > risk_obj):
            val = 100
        else:
            val = sign * (x @ self.mean_rets)
        return(val)

    def policy_surplus_optim(self, ret_obj, dur_interv=1, assets_liab_corr=pd.Series(), method="GD", mean_rets_input=None, liab_val=None, liab_dur=None, gd_method='SLSQP'):
        '''
        dur_interv: Duration interval. Target duration w.r.t. liabilities duration,
        ret_obj: Minimun return objective,
        assets_liab_corr: Pandas series of correlation of each asset and liabilities. If asset not found, it assumes cero correlation.
        method: ['SLSQP', 'trust-constr']
        '''
        ######
        try:
            l_ret = len(ret_obj)
        except:
            l_ret = 0
        mean_rets = self.mean_rets.copy()
        if mean_rets_input is not None:
            mean_rets[mean_rets_input.index] = mean_rets_input.values
        self.assets_liab_corr = assets_liab_corr
        liab_corr = [assets_liab_corr[x] if x in assets_liab_corr.index else 0 for x in self.asset_names]
        R = np.concatenate((np.append([1], liab_corr).reshape((1,self.n_assets + 1)), np.c_[liab_corr, self.corr.values]), axis=0)
        D = np.diag(np.append(np.array(self.liab_vol), np.sqrt(np.diag(self.cov))))

        Sigma = D @ R @ D

        w_l = self.surplus(liab_val=liab_val)['w_l']
        w_ini = self.lb + (1-np.sum(self.lb))*(self.ub - self.lb)/np.sum(self.ub - self.lb)

        if liab_dur is None:
            liab_dur = np.abs(self.liab_risk()['risk']['mac_dur'])
        dur_vec = np.zeros(self.n_assets)
        dur_vec[[self.asset_names.index(x) for x in self.asset_names_cf.values]] = self.assets_dur
        dur_lb = liab_dur - dur_interv
        dur_ub = liab_dur + dur_interv

        dur_cons = {'type': 'ineq', 'fun' : lambda w: np.array([(dur_vec @ w) / np.sum(w[dur_vec!=0]) - dur_lb,-(dur_vec @ w) / np.sum(w[dur_vec!=0]) + dur_ub]),
        'jac' : lambda w: np.array([(np.sum(w[dur_vec!=0]) * dur_vec - (dur_vec @ w) * w * (dur_vec!=0)) / np.sum(w[dur_vec!=0])**2,
                                   (-np.sum(w[dur_vec!=0]) * dur_vec + (dur_vec @ w) * w * (dur_vec!=0)) / np.sum(w[dur_vec!=0])**2])}

        budget_cons = lc(np.ones((1,len(self.lb))), np.ones(1), np.ones(1))
        if self.linCons is not None:
            linCons = np.row_stack([self.linCons.copy(), mean_rets.values])
            consUB = np.append(self.consUB.copy(), np.array([np.Inf]))
        else:
            linCons = mean_rets.values
            consUB = np.array([np.Inf])

        if l_ret == 0:
            if self.linCons is not None:
                consLB = np.append(self.consLB.copy(), np.array([ret_obj]))
            else:
                consLB = np.array([ret_obj])
            cons = [budget_cons, lc(linCons, consLB, consUB), dur_cons]
            sol = minimize(self.suplus_obj, w_ini, args=(w_l, Sigma), bounds=self.bnds, constraints=cons, method=gd_method)
            optim_port = pd.Series(np.round(100*sol.x, 3), index=self.asset_names)
            port_mean = np.transpose(optim_port.values) @ mean_rets.values
            port_excess_mean = port_mean - self.liab_rate*100
            port_vol = np.round(np.float(np.diag(np.sqrt(np.transpose(optim_port.values).reshape((1, self.n_assets)) @ self.cov.values @ optim_port.values))[0]), 3)
            optim_port_ext = np.append(np.array(-w_l*100), optim_port.values)
            surplus_vol = np.sqrt(optim_port_ext @ Sigma @ optim_port_ext)
            port_dur = np.transpose(optim_port.loc[self.asset_names_cf].values/100) @ self.assets_dur
        else:
            sol_exists = np.zeros(l_ret, dtype=bool)
            w_mat = np.zeros((self.n_assets, l_ret))
            for i in range(l_ret):
                if self.linCons is not None:
                    consLB = np.append(self.consLB.copy(), np.array([ret_obj[i]]))
                else:
                    consLB = np.array([ret_obj[i]])
                budget_cons = lc(np.ones((1,len(self.lb))), np.ones(1), np.ones(1))
                cons = [budget_cons, lc(linCons, consLB, consUB), dur_cons]
                sol = minimize(self.suplus_obj, w_ini, args=(w_l, Sigma), bounds=self.bnds, constraints=cons, method=gd_method)
                sol_exists[i] = sol.success
                w_mat[:, i] = np.round(100*sol.x, 3)
                if sol.success and i>0 and np.all(w_mat[:,i]==w_mat[:,i-1]):
                    sol_exists[i] = False
            if not np.any(sol_exists):
                raise ValueError("Solution not found. Try changing constraints or method.")
            port_mean = w_mat[:,sol_exists].T @ mean_rets.values
            optim_port = pd.DataFrame(w_mat[:,sol_exists], index=self.asset_names, columns=np.round(port_mean,2))
            port_excess_mean = port_mean - self.liab_rate*100
            port_vol = np.diag(np.sqrt(np.transpose(optim_port.values) @ self.cov.values @ optim_port.values))
            optim_port_ext = np.concatenate(((-100*w_l*np.ones(sol_exists.sum())).reshape((1, sol_exists.sum())), optim_port.values), axis = 0)
            surplus_vol = np.diag(np.sqrt(np.transpose(optim_port_ext) @ Sigma @ optim_port_ext))

            fi_mat = np.transpose(optim_port.loc[self.asset_names_cf].values/100)
            port_dur = (fi_mat @ self.assets_dur.astype(float))/np.sum(fi_mat, axis=1)
            warnings.warn(optim_port)
            self.optimal_surplus_policy = {'optim_port':optim_port, 'optim_port_ext':optim_port_ext, 'port_mean':port_mean, 'port_excess_mean':port_excess_mean, 'port_vol':port_vol, 'surplus_vol':surplus_vol, 'port_dur':port_dur, 'Sigma':Sigma}
        return self.optimal_surplus_policy

    def plot_surplus_frontier(self, optim_port_surplus=None):
        if optim_port_surplus is None:
            optim_port_surplus = self.optimal_surplus_policy
        plt.plot(optim_port_surplus['surplus_vol'], optim_port_surplus['port_mean'], '-')
        plt.title("Frontera Eficiente")
        plt.xlabel("Volatilidad de Surplus")

    def plot_surplus_dur_ret(self, optim_port_surplus=None):
        if optim_port_surplus is None:
            optim_port_surplus = self.optimal_surplus_policy
        plt.plot(optim_port_surplus['port_dur'], optim_port_surplus['port_mean'], '--')
        plt.title("Rentabilidad y Duración Portafolios optimizados")
        plt.ylabel("Rentabilidad")
        plt.xlabel("Duracion")

    def plot_surplus_ac(self, optim_port_surplus=None):
        if optim_port_surplus is None:
            optim_port_surplus = self.optimal_surplus_policy
        ac = self.asset_data.loc[optim_port_surplus['optim_port'].index]['AssetClass']
        ac_surplus = optim_port_surplus['optim_port'].copy()
        ac_surplus.index = ac.values
        ac_surplus_df = ac_surplus.groupby(ac_surplus.index).sum()
        n_cols = ac_surplus_df.shape[1]
        ind = ac_surplus_df.columns
        width = 0.35       # the width of the bars: can also be len(x) sequence
        colors = ['blue', 'yellow', 'gray']
        bottom = np.vstack((np.zeros(n_cols), ac_surplus_df.cumsum(axis=0).iloc[:-1].values))

        for i in range(ac_surplus_df.shape[0]):
            x = ac_surplus_df.iloc[i,:].values
            plt.bar(ind, x, label =  ac_surplus_df.index[i], bottom=bottom[i,:], color=cm.BuPu(i/10, 1))
        plt.legend()
        plt.title("Asset Classes")
        plt.xlabel('Retorno Esperado')

    # Yield Curve Simulation
    def hw_affine_df_simul(self, simul_times, a=1.3, sigma=0.1, cc_curve=None, cf_times=None, M=1000, cc_type="nom"):
        if cf_times is None:
            cf_times = np.round(np.array(self.assets_cf.index), 3)
        if cc_curve is None:
            cc_curve = self.cc_curve.copy()
        if cc_type == 'nom':
            cc_curve.iloc[1::,0] = np.log(1 + cc_curve.iloc[1::,0] * cc_curve.index[1::])/cc_curve.index[1::]

        hw_model = hw(a=a, sigma=sigma, cc_curve=cc_curve)
        affine = hw_model.hw_affine(cf_times, simul_times)

        A = affine['A']
        B = affine['B']
        fwd_curve = affine["fwd_curve"]
        fwd_times = np.round(affine["fwd_times"],3)

        rt = hw_model.hw_simul_rt(fwd_curve, M=M)
        df_array = np.zeros((len(fwd_times),len(cf_times),M))
        rates_array = np.zeros((len(fwd_times),len(cf_times),M))
        for i in range(M):
            rt_sim = rt[:, i]
            affine_df = hw_model.hw_affine_df(A, B, rt_sim, fwd_times)
            df_array[:,:,i] = affine_df['df_mat']
            rates_array[:,:,i] = affine_df['rates_mat']
        return({'df_array':df_array, 'rates_array':rates_array, 'rt_simul':rt, 'fwd_times':fwd_times, 'fwd_curve':fwd_curve})


    def cf_val_simul(self, simul_times, horiz = 1, a=1.3, sigma=0.01, cc_curve=None, M=1000):
        '''
        Simul liab.
        '''
        cf_times = np.unique(np.sort(np.concatenate((self.assets_cf.index, self.liab_cf.index))))
        for i in range(horiz):
            cf_times = np.append(cf_times, np.append((cf_times[np.where(cf_times.astype(int) == cf_times[-1] - 1)] + 1)[1::], cf_times[-1]+1))

        fwd_curve = self.fwd_curve(cf_times, horiz, cc_curve=cc_curve)
        liab_pr_fwd = self.liab_risk(cc_curve=fwd_curve)['risk']['price']
        assets_risk = self.cf_matrix_risk(fwd_curve)
        assets_pr_fwd = pd.Series([assets_risk[x]['price'] for x in self.asset_names], index=self.asset_names)

        ir_model = self.hw_affine_df_simul(simul_times=simul_times, a=a, sigma=sigma, cc_curve=cc_curve, cf_times=cf_times, M=M)

        try:
            pos = int(np.where(horiz == simul_times)[0])
        except:
            print('Horizon not in simulation times.')

        liab_cf_ind = self.liab_cf.index > 0
        liab_times_array = np.round(np.array(self.liab_cf[liab_cf_ind].index),3)
        liab_cf_array = self.liab_cf.iloc[liab_cf_ind ,0].values

        liab_pos = np.searchsorted(np.round(ir_model['fwd_times'][pos,:],3), liab_times_array) - 1
        liab_simul = np.array([np.sum(ir_model['df_array'][pos, liab_pos, i] * liab_cf_array) for i in range(ir_model['df_array'].shape[2])])

        fwd_curve = self.fwd_curve(cf_times, horiz, cc_curve=cc_curve)
        liab_pr_fwd = self.liab_risk(cc_curve=fwd_curve)['risk']['price']
        liab_rets = (liab_simul - liab_pr_fwd)/liab_pr_fwd

        assets_simul = np.zeros([M, self.assets_cf.shape[1]])
        assets_rets = np.zeros([M, self.assets_cf.shape[1]])
        for i in range(self.assets_cf.shape[1]):
            asset_cf = self.assets_cf.iloc[:,i].dropna()
            asset_cf_ind = asset_cf.index > 0
            asset_times_array = np.array(asset_cf[asset_cf_ind].index)
            asset_cf_array = asset_cf[asset_cf_ind].values
            asset_pos = np.searchsorted(np.round(ir_model['fwd_times'][pos,:],3), asset_times_array) - 1
            asset_simul = [np.sum(ir_model['df_array'][pos, asset_pos, i] * asset_cf_array) for i in range(ir_model['df_array'].shape[2])]
            assets_simul[:,i] = asset_simul
            asset_pr = assets_pr_fwd.loc[self.asset_names_cf[i]]
            assets_rets[:, i] = (asset_simul - asset_pr)/asset_pr

        corr_names = np.append(['Liab'], self.asset_names_cf)
        corr_df = pd.DataFrame(np.corrcoef(np.column_stack((liab_rets, assets_rets)).T), columns = corr_names)


        #if assets_zero_corr is not None:
        #        assets_pr_fwd[assets_par_price] = 100

        return({'liab':liab_simul, 'assets':assets_simul, 'liab_rets':liab_rets, 'assets_rets':assets_rets, 'corr_df':corr_df})
