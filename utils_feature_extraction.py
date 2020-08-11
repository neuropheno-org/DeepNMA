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


def get_TS_features(TS, time, hand):
    """ Extract time series (TS) features from {right or left} hand, returns
    time series featrures in a dictornary.
    
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
    vel = np.gradient(TS)
    acc = np.gradient(vel, time)
    jerk = np.gradient(acc, time)
    jounce = np.gradient(jerk, time)
    
    ts = dict()
    ts['ts_vel_' + hand] = np.average(vel)
    ts['ts_acc_' + hand] = np.average(acc)
    ts['ts_jerk_' + hand] = np.average(jerk)
    ts['ts_jounce_' + hand] = np.average(jounce)
    ts['ts_vel_abs_' + hand] = np.average(np.abs(vel))
    ts['ts_acc_abs_' + hand] = np.average(np.abs(acc))
    ts['ts_jerk_abs_' + hand] = np.average(np.abs(jerk))
    ts['ts_jounce_abs_' + hand] = np.average(np.abs(jounce))
    
    ts['ts_amp_max_' + hand] = np.max(TS)
    ts['ts_amp_min_' + hand] = np.min(TS)
    ts['ts_amp_abs_avg_' + hand] = np.average(np.abs(TS)) 
    
    return ts


def get_periods_feat(TS, time, hand, do_plot=False):
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
    pk_pos, _ =  find_maxmin_peaks(TS, 2, .1)
    _, pk_neg =  find_maxmin_peaks(TS, 2, 1)
    pks_all = np.sort(np.hstack((pk_pos, pk_neg)))
    pk_pos_n = len(pk_pos)
    
    ix_nxt_neg, ix_n = get_next_through(pk_pos, pk_neg)
    
    pkth_times =  time[pk_neg[ix_nxt_neg]] - time[pk_pos[:ix_n]] 
    assert(np.all(pkth_times>0)) # if neg though bebore peak
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
    pks['pk_neg_var_time_' + hand] = np.var(time[pk_neg])  

    pks['pk_pos_avg_amp_' + hand] = np.average(TS[pk_pos])
    pks['pk_neg_avg_amp_' + hand] = np.average(TS[pk_neg])
    pks['pk_pos_avg_time_' + hand] = np.average(time[pk_pos])
    pks['pk_neg_avg_time_' + hand] = np.average(time[pk_neg]) 

    pks['pkth_time_var_' + hand] = np.var(pkth_times)
    pks['pkth_timediff_var_' + hand] = np.var(pkth_time_diff)
    pks['pkth_amp_std_' + hand] = pkth_amp_std    
    pks['pkth_amp_std_' + hand] = np.std(pkth_amps_diff)
    
    if do_plot:
        plt.figure()
        plt.plot(time, TS, 'b')
        plt.plot(time[pk_pos], TS[pk_pos], 'ro')
        plt.plot(time[pk_neg], TS[pk_neg], 'go')
        plt.plot(time[pk_neg[ix_nxt_neg]], TS[pk_neg[ix_nxt_neg]], 'ko')
        
    return pks
    


def find_maxmin_peaks(TS, filter_len=2, height=1, threshold=1):
    "Gets positive and negative peaks from a TS"
    TS = filtfilt(np.ones(filter_len)/filter_len, 1, TS)
    prominence = [0.1*(max(TS) - min(TS)), max(TS) - min(TS)]
    if height is not None:
        height = height + TS.mean()
    pos_peaks, _ = find_peaks(TS, prominence=prominence, height=height)
    neg_peaks, _ = find_peaks(-TS, prominence=prominence, height=-height)
    return pos_peaks, neg_peaks

def get_next_through(pk_pos, pk_neg ):
    """ Find the next though given pk_pos and pk_neg indexes of peaks, returns
    the neg pk after the pos pk"""
    
    ix_nxt_neg = [np.where(ppos<pk_neg)[0] for ppos in pk_pos] 
    ix_nxt_neg = [i[0] for i in ix_nxt_neg if i.size]
    ix_n = len(ix_nxt_neg)
    
    return ix_nxt_neg, ix_n
