# -*- coding: utf-8 -*-
"""
Created on Tue June 1 16:30:41 2018

@author: daniel.velasquez
"""
import numpy as np
import pandas as pd


class hull_white(object):
    def __init__(self, a, sigma, cc_curve):
        self.a = a
        self.sigma = sigma
        self.cc_curve_hw = cc_curve
        self.base = 360

    def hw_affine(self, cf_times, simul_times):
        '''
        cf_times: cash flows times
        cc_curve: Cero coupon curve
        simul_times: Simulation times
        '''
        r0 = float(self.cc_curve_hw.iloc[0])
        simul_rates = np.interp(simul_times, xp=list(self.cc_curve_hw.index),
                                fp=list(self.cc_curve_hw.iloc[:, 0]))
        ls = len(simul_times)
        lcf = len(cf_times)
        cf_rates = np.interp(cf_times, xp=list(self.cc_curve_hw.index),
                             fp=list(self.cc_curve_hw.iloc[:, 0]))

        p0t = np.exp(- simul_rates * simul_times)
        p0T = np.exp(- cf_rates * cf_times)

        p0t_mat = p0t.reshape((ls, 1)) @ np.ones((1, lcf))
        p0T_mat = np.ones((ls, 1))  @ p0T.reshape((1, lcf))
        a_max = np.max(cf_times)
        cf_diff_times = np.array([np.clip(cf_times - x, a_min=0, a_max=a_max) for x in simul_times])
        B = (1 - np.exp(- self.a * cf_diff_times)) / self.a
        simul_times_mat = simul_times.reshape((ls, 1)) @ np.ones((1, lcf))
        ft_mat = simul_rates.reshape((ls, 1)) @ np.ones((1, lcf))
        ptT = p0T_mat / p0t_mat
        ptT[ptT > 1] = 0
        A = ptT * np.exp(B * ft_mat - ((self.sigma ** 2) / (4 * self.a)) * (1 - np.exp(-2 * self.a * simul_times_mat)) * (B ** 2))

        fwd_curve = pd.DataFrame(np.concatenate([np.array([r0]), simul_rates]),
                                 index=np.concatenate([np.array([1 / self.base]), simul_times]),
                                 columns=['Rate'])
        return ({'A': A, 'B': B, 'fwd_curve': fwd_curve, 'fwd_times': cf_diff_times})

    def hw_simul_rt(self, fwd_curve, M=1000):
        ft = np.array(fwd_curve.iloc[1:, 0])
        simul_times = np.array(fwd_curve.index[1:])
        alpha = np.concatenate([np.array([0]), ft + (self.sigma**2 / (2 * self.a**2)) * (1 - np.exp(-self.a * simul_times))**2])

        ls = len(simul_times)
        rt = np.zeros(((ls + 1), M))
        rt[0, :] = float(fwd_curve.iloc[0])
        diff_simul_times = np.concatenate([[simul_times[0]], np.diff(simul_times)])

        for i in range(ls):
            mu = (rt[i, :] - alpha[i]) * np.exp(-self.a * diff_simul_times[i]) + alpha[i + 1]
            var = (self.sigma ** 2 / (2 * self.a)) * (1 - np.exp(-2 * self.a * diff_simul_times[i]))
            rt[i + 1, ] = np.random.normal(mu, np.sqrt(var), M)

        return(rt[1:, :])

    def hw_affine_df(self, A, B, rt_sim, fwd_times):
        lcf = A.shape[1]
        lt = len(rt_sim)
        df_mat = A * np.exp(-B * (rt_sim.reshape((lt, 1)) @ np.ones((1, lcf))))
        df_mat1 = df_mat.copy()
        df_mat1[df_mat1==0] = 1

        rates_mat = np.zeros(fwd_times.shape)
        fwd_times1 = fwd_times.copy()
        fwd_times1[fwd_times1==0] = 1
        rates_mat[fwd_times != 0] = (- np.log(df_mat1) / fwd_times1)[fwd_times != 0]
        return({'df_mat': df_mat, 'rates_mat': rates_mat})

    # For calibration:
    def hw_B(self, t0, tn):
        b = (1 - np.exp(-self.a * (tn - t0))) / self.a
        return(b)

    def hw_logA(self, t0, tn, rate_type='nominal'):
        rt0 = np.interp(t0, xp=list(self.cc_curve_hw.index), fp=list(self.cc_curve_hw.iloc[:, 0]))
        rtn = np.interp(tn, xp=list(self.cc_curve_hw.index), fp=list(self.cc_curve_hw.iloc[:, 0]))
        if rate_type[0] == 'n':
            if t0 != 0:
                rt0 = np.log(1 + rt0 * t0) / t0
                rtn = np.log(1 + rtn * tn) / tn
        if rate_type[0] == 'e':
            rt0 = np.log(1 + rt0)
            rtn = np.log(1 + rtn)
        pt0 = np.exp(-rt0 * t0)
        ptn = np.exp(-rtn * tn)
        b = self.hw_B(t0=t0, tn=tn)

        a_new = np.log(ptn / pt0) - b * (-rt0) - (self.sigma ** 2 / (4 * (self.a ** 3))) * ((np.exp(-self.a * tn) - np.exp(-self.a * t0)) ** 2) * (np.exp(2 * self.a * t0) - 1)
        return(a_new)
