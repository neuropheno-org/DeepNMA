#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils for extracting features from a time series

Created on Mon May 18 12:43:32 2020

@author: adonay
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, filtfilt
from scipy.stats import iqr, entropy
from mne.filter import filter_data
from scipy import linalg


def get_TS_features(TS, time, prefix, suffix, deriv_ord):
    """ Extract time series (TS) features, returns
    time series featrures in a dictornary.
    
    Parameters
    -------------
    TS : 1D array
        Times series to extract features
    time : in seconds
        Time for each sample.
    prefix : str
        To be added at beginning of keys name of dict features
    suffix: str
        To be added at the end of keys name of dict features
    deriv_ord: int
        The max derivative order from which to compute features. Up to 4th order.
    
    Returns
    --------
    
    Dict with features
    """
    
    def percentile_90(x):
        return np.percentile(x,90)
    
    def percentile_10(x):
        return np.percentile(x,10)    
    
    def sig_range(x):
        return max(x) - min(x)
    
    def sing_entropy(x):
        hist1 = np.histogram(x, density=True)
        # ent = -(hist1[0]*np.log(np.abs(hist1[0]))).sum()
        return entropy(hist1[0], base=2)
    
    def abs_avg (x):
        return np.average(np.abs(x))
    
    der_names = ["pos", 'vel', 'acc', 'jerk', 'jounce']
    
    measures = [(np.average, "avg"),
                (abs_avg, "t_avg"),
                (np.max, "max"),
                (np.min, "min"),
                (np.std, "std"),
                (np.median, "med"),
                (percentile_90, "pth90"),
                (percentile_10, "pth10"), 
                (sig_range, "range"),
                (iqr, "iqr"),
                (sing_entropy, "entropy")]
    
             
    features = dict()
    
    for order in np.arange(deriv_ord + 1):
        ord_name = der_names[order]
        
        if order == 0:
            ts_deriv = [TS]
        else:    
            deriv = np.gradient(ts_deriv[order - 1], time)
            ts_deriv.append(deriv)
        
        for measure, name in measures:            
            feat = measure(ts_deriv[order])
            
            features[f"{prefix}_{ord_name}_{name}_{suffix}"] = feat
        
        
        
    # vel = np.gradient(TS, time)
    # acc = np.gradient(vel, time)
    # jerk = np.gradient(acc, time)
    # jounce = np.gradient(jerk, time)
    
    # ts = dict()
    # ts['ts_vel_' + name_ext] = np.average(vel)
    # # ts['ts_0vel_' + hand] = sum((vel[:-1]*vel[1:])<0)/ time[-1]-time[0]
    
    # ts['ts_acc_' + hand] = np.average(acc)
    # ts['ts_jerk_' + hand] = np.average(jerk)
    # ts['ts_jounce_' + hand] = np.average(jounce)
    # ts['ts_vel_abs_' + hand] = np.average(np.abs(vel))
    # ts['ts_acc_abs_' + hand] = np.average(np.abs(acc))
    # ts['ts_jerk_abs_' + hand] = np.average(np.abs(jerk))
    # ts['ts_jounce_abs_' + hand] = np.average(np.abs(jounce))
    
    # ts['ts_amp_max_' + hand] = np.max(TS)
    # ts['ts_amp_min_' + hand] = np.min(TS)
    # ts['ts_amp_abs_avg_' + hand] = np.average(np.abs(TS)) 
    
    # ts['ts_vel_std_' + hand] = np.std(vel)
    # ts['ts_vel_range_' + hand] = max(vel) - min(vel)
    # ts['ts_vel_med_' + hand] = np.median(vel)
    # ts['ts_vel_10th_' + hand] = np.percentile(vel,10)
    # ts['ts_vel_90th_' + hand] = np.percentile(vel,90)
    # ts['ts_vel_iqr_' + hand] = iqr(vel)
    
    
    # add entropy, loop over features
    
    
    # ts['ts_acc_std_' + hand] = np.std(acc)
    # ts['ts_acc_range_' + hand] = max(acc) - min(acc)
    # ts['ts_acc_med_' + hand] = np.median(acc)
    # ts['ts_acc_10th_' + hand] = np.percentile(acc,10)
    # ts['ts_acc_90th_' + hand] = np.percentile(acc,90)
    # ts['ts_acc_iqr_' + hand] = iqr(acc)

    
    return features


def get_periods_feat(TS, time, hand, do_plot=False, ax=None, height=.1, prominence=.05, filt_param=[60, 2, 6]):
    """ Get peaks from a TS and extract features from Peaks from 
    {right or left} hand, returns peak featrures in a dictornary.
    
    Parameters
    -------------
    TS : 1D array
        Times series to extract features
    time : in seconds
        Time for each sample
    hand: string {'r', 'l'}
        String to be added in the dic of features
        
    Returns
    --------
    Dict with features
    """
    
    
    pk_pos, _ =  find_maxmin_peaks(TS, 2,  height=height, prominence_f=prominence, filt_param=filt_param)
    _, pk_neg =  find_maxmin_peaks(TS, 2,  height=1, prominence_f=prominence,  filt_param=filt_param)
    pks_all = np.sort(np.hstack((pk_pos, pk_neg)))
    pk_pos_n = len(pk_pos)
    
    ix_nxt_neg, ix_n = get_next_through(pk_pos, pk_neg)
    
    pkth_times =  time[pk_neg[ix_nxt_neg]] - time[pk_pos[:ix_n]] 
    assert(np.all(pkth_times>0)) # if neg through bebore peak
    pkth_time_diff = np.diff(pkth_times)

    
    pkth_amps = TS[pk_pos[:ix_n]] - TS[pk_neg[ix_nxt_neg]]
    pkth_amps_diff = np.diff(pkth_amps)
    pkth_amp_std = np.std(pkth_amps_diff)
    
    
    pk_vel = np.gradient(TS[pks_all])#, time[pks_all])
    pk_acc = np.gradient(pk_vel)#, time[pks_all])
    pk_jerk = np.gradient(pk_acc)#, time[pks_all])
    
    pks = dict()
    pks['pk_vel_' + hand] = np.average(pk_vel)
    pks['pk_acc_' + hand] = np.average(pk_acc)
    pks['pk_jerk_' + hand] = np.average(pk_jerk)
    pks['pk_avg_period_' + hand] = (time[pk_pos[-1]] - time[pk_pos[0]])/pk_pos_n
    pks['pk_vel_abs_' + hand] = np.average(np.abs(pk_vel))
    pks['pk_acc_abs_' + hand] = np.average(np.abs(pk_acc))
    pks['pk_jerk_abs_' + hand] = np.average(np.abs(pk_jerk))
    
    pks['pk_pos_var_amp_' + hand] = np.var(TS[pk_pos])
    pks['pk_neg_var_amp_' + hand] = np.var(TS[pk_neg])
    pks['pk_pos_var_time_' + hand] = np.var(time[pk_pos])
    # pks['pk_neg_var_time_' + hand] = np.var(time[pk_neg])  

    pks['pk_pos_avg_amp_' + hand] = np.average(TS[pk_pos])
    pks['pk_neg_avg_amp_' + hand] = np.average(TS[pk_neg])
    pks['pk_pos_avg_time_' + hand] = np.average(time[pk_pos])
    # pks['pk_neg_avg_time_' + hand] = np.average(time[pk_neg]) 

    pks['pkth_time_var_' + hand] = np.var(pkth_times)
    pks['pkth_timediff_var_' + hand] = np.var(pkth_time_diff)
    pks['pkth_amp_std_' + hand] = pkth_amp_std    
    pks['pkth_amp_std_' + hand] = np.std(pkth_amps_diff)
    
    # median, var, abs diff pk, pk-1
    # take slope (y ~ x0 + time) or compare first vs half session take diff 
    # change over time feat category
    
    if do_plot:
        if ax is None:
            plt.figure()
            plt.plot(time, TS, 'b')
            plt.plot(time[pk_pos], TS[pk_pos], 'ro')
            plt.plot(time[pk_neg], TS[pk_neg], 'go')
            plt.plot(time[pk_neg[ix_nxt_neg]], TS[pk_neg[ix_nxt_neg]], 'ko')
        else:
            ax.plot(time, TS, 'b')
            ax.plot(time[pk_pos], TS[pk_pos], 'ro')
            ax.plot(time[pk_neg], TS[pk_neg], 'go')
            ax.plot(time[pk_neg[ix_nxt_neg]], TS[pk_neg[ix_nxt_neg]], 'ko')
        
    return pks
    


def find_maxmin_peaks(TS, filter_len=2, height=1, prominence_f=.1, filt_param=[60, 2, 6]):
    "Gets positive and negative peaks from a TS"
    # TS = filtfilt(np.ones(filter_len)/filter_len, 1, TS)
    # TS = filter_data(TS, filt_param[0], filt_param[1], filt_param[2], pad='reflect', fir_design="firwin2", verbose=0)
    # TS = TS[0,:] if TS.shape[0]== 1 else TS
    prominence = [prominence_f*(np.max(TS) - np.min(TS)), np.max(TS) - np.min(TS)]
    if height is not None:
        height = height + TS.mean()
    
    pos_peaks, _ = find_peaks(TS, prominence=prominence, height=height)
    
    if height is not None:
        neg_peaks, _ = find_peaks(-TS, prominence=prominence, height=-height)
    else:
        neg_peaks, _ = find_peaks(-TS, prominence=prominence, height=height)
    
    return pos_peaks, neg_peaks

def get_next_through(pk_pos, pk_neg ):
    """ Find the next though given pk_pos and pk_neg indexes of peaks, returns
    the neg pk after the pos pk"""
    
    ix_nxt_neg = [np.where(ppos<pk_neg)[0] for ppos in pk_pos] 
    ix_nxt_neg = [i[0] for i in ix_nxt_neg if i.size]
    ix_n = len(ix_nxt_neg)
    
    return ix_nxt_neg, ix_n

def pca_temporal(data, wind_size=30, overlap=1):
    "Apply PCA over sliding window"
    n_sampls, half_wind = data.shape[1], np.int(wind_size/2)
    winds = np.arange(wind_size, n_sampls+1)
    data_pc = []
    for wind in winds:
        U, s, V = linalg.svd(data[:, wind-wind_size:wind], full_matrices=False)
        scale = linalg.norm(s) / np.sqrt(data.shape[1])
        wind_pc = scale * V
        if wind == wind_size:
            data_pc.append(wind_pc[:, :half_wind+1])
        elif wind == n_sampls:
            data_pc.append(wind_pc[:, half_wind:])
        else:
            data_pc.append(wind_pc[:, half_wind:half_wind+1])
    data_pc = np.hstack(data_pc)
    return data_pc

