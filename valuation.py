
"""
Created on Tue June 1 16:30:41 2018
@author: daniel.velasquez
"""
import numpy as np
import datetime as dt
import calendar
from dateutil.relativedelta import relativedelta
import pandas as pd

def get_df(cc_curve, times, type="efec"):
    cc_rates = np.interp(times, xp=list(cc_curve.index), fp=list(cc_curve.iloc[:, 0]))
    if type=="efec":
        df = 1 / (1 + cc_rates) ** times
    elif type=="nom":
        df = 1 / (1 + cc_rates * times)
    else:
        df = np.exp(-cc_rates * times)
    df_df = pd.Series(df, index=times)
    return cc_rates, df_df


class dayscounters(object):
    def __init__(self, conv):
        self.conv = conv

    @property
    def days_count(self):
        if self.conv == 'thirty/360':
            def day_count(start_date, end_date):
                """Returns number of days between start_date and end_date, using Thirty/360 convention"""
                d1 = min(30, start_date.day)
                d2 = min(d1, end_date.day) if d1 == 30 else end_date.day

                return 360 * (end_date.year - start_date.year) + 30 * (end_date.month - start_date.month) + d2 - d1
        else:
            def day_count(start_date, end_date):
                """Returns number of days between start_date and end_date"""
                return (end_date - start_date).days

        return(day_count)

    @property
    def year_fraction(self):
            if self.conv == 'thirty/360' or self.conv == 'actual/360':
                def year_frac(start_date, end_date):
                    """Returns fraction in years between start_date and end_date"""
                    return self.days_count(start_date, end_date) / 360.0
            elif self.conv == 'actual/365':
                def year_frac(start_date, end_date):
                    """Returns fraction in years between start_date and end_date, using Actual/365 convention"""
                    return self.days_count(start_date, end_date) / 365.0
            else:
                def year_frac(start_date, end_date):
                    """Returns fraction in years between start_date and end_date, using Actual/Actual convention"""
                    if start_date == end_date:
                        return 0.0

                    start_date = dt.datetime.combine(start_date, dt.datetime.min.time())
                    end_date = dt.datetime.combine(end_date, dt.datetime.min.time())

                    start_year = start_date.year
                    end_year = end_date.year
                    year_1_diff = 366 if calendar.isleap(start_year) else 365
                    year_2_diff = 366 if calendar.isleap(end_year) else 365

                    total_sum = end_year - start_year - 1
                    diff_first = dt.datetime(start_year + 1, 1, 1) - start_date
                    total_sum += diff_first.days / year_1_diff
                    diff_second = end_date - dt.datetime(end_year, 1, 1)
                    total_sum += diff_second.days / year_2_diff
                    return total_sum
            return(year_frac)


class bond(dayscounters):
    def __init__(self, cpn, matur_date, freq, conv='actual/360'):
        self.cpn = cpn
        self.matur_date = matur_date
        self.freq = freq
        self.conv = conv

    def value_ytm(self, val_date, ytm):
        ncpn = int(((self.matur_date - val_date).days / 365) * self.freq + 2)
        per_months = 12 / self.freq
        cpn_dates = [self.matur_date + relativedelta(months=- i * per_months) for i in np.arange(0, ncpn)[:: -1]]

        fut_cpn_dates = [d for d in cpn_dates if d > val_date]
        # prev_cpn_date = [d for d in cpn_dates if d <= val_date][0]

        # matur_time = self.year_fraction(val_date, matur_date)
        cpn_times = np.array([self.year_fraction(val_date, d) for d in fut_cpn_dates])

        lfcpn = len(fut_cpn_dates)
        delta = np.array([self.year_fraction(d0, d1) for d0, d1 in zip(cpn_dates[-(lfcpn + 1): -1], cpn_dates[-lfcpn:])])
        cf = self.cpn * delta
        cf[-1] = cf[-1] + 1
        df = 1 / (1 + ytm / self.freq) ** (self.freq * cpn_times)
        price = np.sum(cf * df)
        tdf = df * cpn_times

        mac_dur = -np.sum(cf * tdf) / price
        mod_dur = mac_dur / (1 + ytm / self.freq)
        convex = np.sum((cpn_times / self.freq + cpn_times ** 2) * cf * df) / (price * (1 + ytm / self.freq)**2)
        return({'price': price, 'mac_dur': mac_dur, 'mod_dur': mod_dur, 'convex': convex})

    def value_df(self, val_date, cc_curve, df=None, type="efec"):
        ncpn = int(((self.matur_date - val_date).days / 365) * self.freq + 2)
        per_months = 12 / self.freq
        cpn_dates = [self.matur_date + relativedelta(months=- i * per_months) for i in np.arange(0, ncpn)[:: -1]]

        fut_cpn_dates = [d for d in cpn_dates if d > val_date]
        # prev_cpn_date = [d for d in cpn_dates if d <= val_date][0]

        # matur_time = self.year_fraction(val_date, matur_date)
        cpn_times = np.array([self.year_fraction(val_date, d) for d in fut_cpn_dates])

        lfcpn = len(fut_cpn_dates)
        delta = np.array([self.year_fraction(d0, d1) for d0, d1 in zip(cpn_dates[-(lfcpn + 1): -1], cpn_dates[-lfcpn:])])
        cf = self.cpn * delta
        cf[-1] = cf[-1] + 1
        if df is None:
            cc_rates = np.interp(cpn_times, xp=list(cc_curve.index), fp=list(cc_curve.iloc[:, 0]))
            if type=="efec":
                df = 1 / (1 + cc_rates) ** cpn_times
            elif type=="nom":
                df = 1 / (1 + cc_rates * cpn_times)
            else:
                df = np.exp(-cc_rates * cpn_times)
        price = np.sum(cf * df)
        tdf = df * cpn_times
        mac_dur = -np.sum(cf * tdf) / price

        ytm = (np.prod(1 / df) ** (1 / np.sum(self.freq * cpn_times)) - 1) * self.freq
        mod_dur = mac_dur / (1 + ytm / self.freq)
        convex = np.sum((cpn_times / self.freq + cpn_times ** 2) * cf * df) / (price * (1 + ytm / self.freq)**2)
        return({'price': price, 'mac_dur': mac_dur, 'mod_dur': mod_dur, 'convex': convex, 'ytm': ytm, 'fut_cpn_dates':fut_cpn_dates})


class cash_flows(object):
    def __init__(self, cfs):
        '''
        cfs: Cash flows. Time in years.
        '''
        self.cfs = cfs.dropna()
        self.cf_values = np.array(self.cfs.values[self.cfs.index>0])
        self.times = np.array(self.cfs.index[self.cfs.index>0])
        try:
            self.freq = int(1/np.diff(self.cfs.index)[1])
        except:
            self.freq = 1

    def value_ytm(self, ytm):
        df = 1 / (1 + ytm / self.freq) ** (self.freq * self.times)
        price = np.sum(self.cf_values * df)
        tdf = df * self.times

        mac_dur = -np.sum(self.cf_values * tdf) / price
        mod_dur = mac_dur / (1 + ytm / self.freq)
        convex = np.sum((self.times / self.freq + self.times ** 2) * self.cf_values * df) / (price * (1 + ytm / self.freq)**2)
        return({'price': price, 'mac_dur': mac_dur, 'mod_dur': mod_dur, 'convex': convex})

    def price_df(self, cc_curve, type="efec"):
        cc_rates = np.interp(self.times, xp=list(cc_curve.index), fp=list(cc_curve.iloc[:, 0]))
        if type=="efec":
            df = 1 / (1 + cc_rates) ** self.times
        elif type=="nom":
            df = 1 / (1 + cc_rates * self.times)
        else:
            df = np.exp(-cc_rates * self.times)
        price = np.sum(self.cf_values * df)
        return(price)

    def value_df(self, cc_curve, type="efec"):
        cc_rates = np.interp(self.times, xp=list(cc_curve.index), fp=list(cc_curve.iloc[:, 0]))
        if type=="efec":
            df = 1 / (1 + cc_rates) ** self.times
        elif type=="nom":
            df = 1 / (1 + cc_rates * self.times)
        else:
            df = np.exp(-cc_rates * self.times)

        price = np.sum(self.cf_values * df)
        tdf = df * self.times

        mac_dur = -np.sum(self.cf_values * tdf) / price
        ytm = (np.prod(1 / df) ** (1 / np.sum(self.freq * self.times)) - 1) * self.freq
        mod_dur = mac_dur / (1 + ytm / self.freq)
        convex = np.sum((self.times / self.freq + self.times ** 2) * self.cf_values * df) / (price * (1 + ytm / self.freq)**2)
        return({'price': price, 'mac_dur': mac_dur, 'mod_dur': mod_dur, 'convex': convex, 'ytm': ytm})

    def key_rates(self, cc_curve, delta_cc=0.01):
        cc_rates = np.interp(self.times, xp=list(cc_curve.index), fp=list(cc_curve.iloc[:, 0]))
        df = 1 / (1 + cc_rates * self.times)
        price = np.sum(self.cf_values * df)
        n_rates = len(cc_rates)
        kr = np.zeros(n_rates)
        for i,cc_i in zip(range(n_rates), cc_rates):
            df[i] = 1 / (1 + (cc_i + delta_cc) * self.times[i])
            price_down = np.sum(self.cf_values * df)
            df[i] = 1 / (1 + (cc_i - delta_cc) * self.times[i])
            price_up = np.sum(self.cf_values * df)
            kr[i] = 100 * 0.5 * (price_up - price_down) / price
        return(pd.Series(kr, index=self.times))
