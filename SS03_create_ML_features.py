#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 07:39:40 2020

@author: adonay
"""


import os.path as op
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import utils_io as uio
import utils_signal_processing as sig_proc
import utils_feature_extraction as feat_ext
from mne.filter import filter_data
from sklearn.decomposition import PCA

def zscore(x):
    x = (x - np.nanmean(x))/ np.nanstd(x)
    return x

def minmax_scaler(x):
    xmin, xmax = np.min(x), np.max(x)
    x = (x - xmin)/ (xmax - xmin)
    return x


def filename_creator(subj, folder_name):
    
    diagn = df_beh.loc[subj, 'gen_diagnosis']
    diag_num = df_beh.loc[subj, 'gen_diagnosis_num']
    age = df_beh.loc[subj, 'age']
    
    if diagn == "Control":
        severity = 0
        ttl = f"{n} {subj}, age {df_beh.loc[subj, 'age']} ,{diagn}"
    elif diagn == "PD":
        ttl = f"{n} {subj}, age {df_beh.loc[subj, 'age']} ,{diagn}, {df_beh.loc[subj, ' UPDRS Total']}"
        
        severity = df_beh.loc[subj, 'updrs_arm_total_R']
    else:
        ttl = f"{n} {subj}, age {df_beh.loc[subj, 'age']} ,{ddiagn}, {df_beh.loc[subj, 'BARS Total']}"
        severity = df_beh.loc[subj, 'common_arm_score_L']
     
    fname = f"{folder_name}/{severity}_{diag_num}_{age}_{subj}.png" 
    return fname

# def save_plot(x, y, folder):
    
         
# Path definitions
root_dir = '/home/adonay/Desktop/projects/Ataxia'
model_name = '_resnet152_FingerTappingJan29shuffle1_650000'
paths = uio.get_paths(model_name, root_dir)


sfreq_common = 60
BP_filr = [1, 10]


# load data
df_beh = pd.read_csv(paths['beh'], index_col=0)


# fname = f"TS_filt_{BP_filr[0]}_{BP_filr[1]}hz_{sfreq_common}Fs_{model_name}.pickle"
fname = f"TS_{sfreq_common}Fs_{model_name}.pickle"
with open(paths['out'] + fname, 'rb') as f:
    TS_preds = pickle.load( f)


## Initialize dataframe for storing features and patient data
subjs = list(TS_preds.keys())
init_var = []
for h in ['r', 'l']:
    t = np.arange(200)
    v = np.sin(t)
    feat_TS = feat_ext.get_TS_features(v, t, h)
    init_var.extend(list(feat_TS.keys()))
    feat_pks = feat_ext.get_periods_feat(v,t, h, False, None, .5, .1, [1, None, None])
    init_var.extend(list(feat_pks.keys()))
df = pd.DataFrame(columns=init_var, index=subjs)

n= 20
n += 1
subj = subjs[n]
fig0, ax = plt.subplots(1)


# run analysis
for n, subj in enumerate(subjs):
    ts = TS_preds[subj]['TS_filt'] 
    times = TS_preds[subj]['times_filt']
    
    for ix, s in enumerate(['r', 'l']):
        ix_inx = 0 + (ix*3)
        ix_thb = 1 + (ix*3)
        
        min_sz = min(ts[ix_inx].shape[1], ts[ix_thb].shape[1])
        ts_inx = ts[ix_inx][:,:min_sz] 
        ts_thb = ts[ix_thb][:,:min_sz] 
        
        tapping = np.sqrt(np.sum(ts_inx - ts_thb, axis=0)**2)
        
        tapping = filter_data(tapping, sfreq_common, BP_filr[0], BP_filr[1], pad='reflect', verbose=0)
        tapping = zscore(tapping)
        time = times[ix_inx][:min_sz]
               
        
        _, freq, line = ax.psd(tapping, Fs=sfreq_common, return_line=True)
        px = line[0].get_ydata()
        mx = np.argmax(px[6:])+6
        pk_fq = freq[mx]
        filt_param = [sfreq_common, pk_fq-1,8]

        feat_pks = feat_ext.get_periods_feat(tapping, time, s, False, [],  height=.05, prominence=.1)
        feat_TS = feat_ext.get_TS_features(tapping, time, s)

        
        feat = {**feat_TS, **feat_pks}#, **feat_pks2}
        for k, v in feat.items():
            df.loc[subj, k]= v
        
        # Calculate finger correlations
        ts_inx_f = filter_data(ts_inx, sfreq_common, 1, None, pad='reflect', verbose=0)
        ts_thb_f = filter_data(ts_thb, sfreq_common, 1, None, pad='reflect', verbose=0)

        neg_corr = min(np.corrcoef(ts_inx_f, ts_thb_f)[0,2], np.corrcoef(ts_inx_f, ts_thb_f)[1,3])
        pos_corr = max(np.corrcoef(ts_inx_f, ts_thb_f)[0,2], np.corrcoef(ts_inx_f, ts_thb_f)[1,3])

        
        # df.loc[subj, "ts_fing_corr_pos" + s] = pos_corr
        df.loc[subj, "ts_fing_corr_neg" + s] = neg_corr
        
        df.loc[subj, "pk_freq_" + s] = pk_fq
        df.loc[subj, "out_times_" + s] = times[ix_inx][-1] - times[ix_inx][0]
        df.loc[subj, "out_fps_" + s] = TS_preds[subj]['sfreq_ori'][0][ix_inx]
    
    # r_feat = [c for c in df.columns if c[-1] == "r"]
    # l_feat = [c for c in df.columns if c[-1] == "l"]
     
    # len_sum = (times[0].size + times[3].size )
    # weigh_rl = [times[0].size/len_sum, times[3].size/len_sum]
     
    # for c_r, c_l in zip(r_feat, l_feat):
    #     df[c_r[:-1]+"b"] = df[c_r]*weigh_rl[0] + df[c_l]*weigh_rl[1]
        
Finger_tapping = pd.concat([df, df_beh],  axis=1, join='inner')
n_sbj = Finger_tapping.shape[0]

fname_out = paths['out'] + f'FT_feat_{n_sbj}subj_{BP_filr[0]}_{BP_filr[1]}hz_{sfreq_common}Fs_{model_name}.csv'

assert(n_sbj == len(subjs))
Finger_tapping.to_csv(fname_out)   
height,prominence = [1, .1], .05
print(f"height: {height}, prominence: {prominence}") 
# runfile('/data/github/DeepNMA/SS04_feature_ML_class.py', wdir='/data/github/DeepNMA')
# runfile('/data/github/DeepNMA/SS04_feature_ML_regres.py', wdir='/data/github/DeepNMA')
