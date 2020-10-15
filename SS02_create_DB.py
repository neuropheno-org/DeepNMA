#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 09:23:45 2020

@author: adonay
"""


import os.path as op
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import utils_io as uio
import utils_signal_processing as sig_proc
import utils_visualization as viz
import utils_retraining as retrn

import os.path as op
import pickle
import numpy as np
import pandas as pd
from scipy import interpolate
from mne.filter import filter_data, resample
import utils_io as io
import utils_feature_extraction as feat_ext


def saved_TS_data(paths, TS_predictions=None):
    "Save TS_predictions if not None or load them/create dict"

    if TS_predictions is not None:
        with open(paths['out_TS'], 'wb') as f:
            pickle.dump(TS_predictions, f)
        return

    if op.exists(paths['out_TS']):
        with open(paths['out_TS'], 'rb') as f:
            TS_predictions = pickle.load(f)
    else:
        TS_predictions = {}
    return TS_predictions

def times_later_2_fing(times, n_right, n_left=None):
    "Convert L/R beg end to n fingers *[beg end]"
    
    n_left = n_right if n_left == None else n_left

    time_fings = np.vstack([
        np.reshape(np.repeat(times[i[0]:i[1]], s), (2,-1)).T
         for i, s in zip([[0,2],[2,4]], [n_right, n_left])])
    return time_fings
    
# Path definitions
root_dir = '/home/adonay/Desktop/projects/Ataxia'
model_name = '_resnet152_FingerTappingJan29shuffle1_730000'
paths = uio.get_paths(model_name, root_dir)

# load data
df_beh = pd.read_csv(paths['beh'], index_col=0)
TS_preds = saved_TS_data(paths)

# Remove bad
TS_preds = {k:v for k, v in TS_preds.items() if v['pred_qual'] == 'good'}

sfreq_common = 60
BP_filr = [1, 10]


stats = []
for subj in TS_preds.keys():
    stats_s = []
    TS = TS_preds[subj]["TS"]
    tstmp = TS_preds[subj]["timestamp"]
    # make times for each finger 
    times = times_later_2_fing(TS_preds[subj]["times"], TS.shape[0]/2)
    
    ts_fil =[]
    ts_tstmp = []
    ts_freqs = []
    for fing_ts, fing_time in zip(TS, times):
        
        fing_ts = fing_ts[:,fing_time[0]:fing_time[1]]
        fing_tsmp = tstmp[fing_time[0]:fing_time[1]].flatten()
        mask = np.zeros([fing_tsmp.shape[0]]) ==1
        mask = np.isnan(fing_ts[0])
        
        ts2, time2 = sig_proc.interp_tracking(fing_ts, fing_tsmp, mask, lin_space=True)
        sfreq = 1/ np.average(np.diff(time2.T))
        ts_freqs.append(sfreq)
        ts3, time3, ratio = sig_proc.resample_ts(ts2, time2, sfreq, sfreq_common)
        
        ts_fil.append(filter_data(ts3, sfreq, 1, 10, pad='reflect', verbose=0))
        ts_tstmp.append(time3)
        
        # ts_fil = filter_data(ts3, sfreq, 1, 10, pad='reflect')
        # plt.figure()
        # plt.plot(fing_tsmp, fing_ts[1,:])  
        # plt.plot(time3, ts_fil[1,:])
    
    del TS_preds[subj]['raw']
    TS_preds[subj]['TS_filt'] = ts_fil
    TS_preds[subj]['times_filt'] = ts_tstmp
    ts_freqs.append(ratio)
    TS_preds[subj]['sfreq_ori'] = [ts_freqs, ratio]
    print(ts_freqs[0], ts_freqs[3])

fname = f"TS_filt_{BP_filr[0]}_{BP_filr[1]}hz_{sfreq_common}Fs_{model_name}.pickle"
with open(paths['out'] + fname, 'wb') as f:
    pickle.dump(TS_preds, f)
    
            
            
  
