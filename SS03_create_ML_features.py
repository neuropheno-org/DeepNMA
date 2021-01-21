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
import matplotlib 
font = {'family' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)

def zscore(x):
    x = (x - np.nanmean(x))/ np.nanstd(x)
    return x

def minmax_scaler(x):
    xmin, xmax = np.min(x), np.max(x)
    x = (x - xmin)/ (xmax - xmin)
    return x

def make_fig():
    fig = plt.figure(figsize=(15, 20), constrained_layout=True)
    gs = fig.add_gridspec(4, 4)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('TS, peaks and pk slopes')
    ax2 = fig.add_subplot(gs[1,:-2])
    ax3 = fig.add_subplot(gs[2, :-2])
    ax4 = fig.add_subplot(gs[3, 0])
    ax5 = fig.add_subplot(gs[3, 1])
    ax6 = fig.add_subplot(gs[1, 2:])
    ax7 = fig.add_subplot(gs[2, 2:])
    ax8 = fig.add_subplot(gs[3, 2:])
    return fig, [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

def filename_creator(subj, folder_name):
    
    diagn = df_beh.loc[subj, 'gen_diagnosis']
    diag_num = df_beh.loc[subj, 'gen_diagnosis_num']
    age = df_beh.loc[subj, 'age']
    
    if diagn == "Control":
        severity = 0
        ttl = f"{n} {subj}, age {df_beh.loc[subj, 'age']} ,{diagn}"
    elif diagn == "PD":
        ttl = f"{n} {subj}, age {df_beh.loc[subj, 'age']} ,{diagn}, {df_beh.loc[subj, 'updrs_arm_total_R']}"
        
        severity = df_beh.loc[subj, 'updrs_arm_total_R']
    else:
        ttl = f"{n} {subj}, age {df_beh.loc[subj, 'age']} ,{diagn}, {df_beh.loc[subj, 'common_arm_score_L']}"
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



ord_num = 3
## Initialize dataframe for storing features and patient data
subjs = list(TS_preds.keys())
init_var = []
for s in ['r', 'l']:
    t = np.arange(200)
    v = np.sin(t)
    pk_pos, pk_neg = feat_ext.get_peaks(v, height=.5, prominence=.1, time=[])
    
    feat_thth = feat_ext.get_pkpk_feat(v, t, pk_neg, "thth", s)        
    feat_pkth = feat_ext.get_pktrough_feat(v, t, pk_pos, pk_neg, "pkth", s)        
    feat_pks = feat_ext.get_peak_shape_feat(v, t, pk_pos, 'pk', s, ord_num)
    feat_ths = feat_ext.get_peak_shape_feat(v, t, pk_neg, 'th', s, ord_num)
    feat_TS = feat_ext.get_TS_features(v, t, 'ts', s, ord_num, exclude_order=[0])
        
    keys = [list(k.keys()) for k in [feat_TS, feat_pks, feat_ths, feat_pkth, feat_thth]]
    keys = [i for kk in keys for i in kk]
    
df = pd.DataFrame(columns=keys, index=subjs)

n= 15
n += 1
subj = subjs[n]
fig0, ax = plt.subplots(1)

do_plot = False
fig, axs= make_fig()




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
        
        tapping1 = filter_data(tapping, sfreq_common, BP_filr[0], BP_filr[1], pad='reflect', verbose=0)
        tapping1 = zscore(tapping1)
        
        tapping2 = filter_data(tapping, sfreq_common, None, BP_filr[1], pad='reflect', verbose=0)
        tapping2 = zscore(tapping2)
        
        time = times[ix_inx][:min_sz]
               
        _, freq, line = ax.psd(tapping1, Fs=sfreq_common, return_line=True)
        px = line[0].get_ydata()
        mx = np.argmax(px[6:])+6
        pk_fq = freq[mx]


        pk_pos, pk_neg = feat_ext.get_peaks(tapping1, height=.05, prominence=.1, time=time, do_plot=do_plot, ax=axs[0])
        
        
        feat_thth = feat_ext.get_pkpk_feat(tapping2, time, pk_neg, "thth", s,  do_plot=do_plot, axs=axs[3:5])        
        feat_pkth = feat_ext.get_pktrough_feat(tapping2, time, pk_pos, pk_neg, "pkth", s,  do_plot=do_plot, axs=axs[1:3])        
        feat_pks = feat_ext.get_peak_shape_feat(tapping2, time, pk_pos, 'pk', s, ord_num, do_plot=do_plot, ax=axs[0])
        feat_ths = feat_ext.get_peak_shape_feat(tapping2, time, pk_neg, 'th', s, ord_num, do_plot=do_plot, ax=axs[0])
        feat_TS = feat_ext.get_TS_features(tapping2, time, 'ts', s, ord_num, exclude_order=[0], do_plot=do_plot, axs=axs[5:])

        
        feat = {**feat_TS, **feat_pks, **feat_ths, **feat_thth, **feat_pkth}
        for k, v in feat.items():
            df.loc[subj, k]= v

        # Calculate finger correlations
        ts_inx_f = np.diff(filter_data(ts_inx, sfreq_common, 1, None, pad='reflect', verbose=0))
        ts_thb_f = np.diff(filter_data(ts_thb, sfreq_common, 1, None, pad='reflect', verbose=0))

        neg_corr = min(np.corrcoef(ts_inx_f, ts_thb_f)[0,2], np.corrcoef(ts_inx_f, ts_thb_f)[1,3])
        pos_corr = max(np.corrcoef(ts_inx_f, ts_thb_f)[0,2], np.corrcoef(ts_inx_f, ts_thb_f)[1,3])

        
        df.loc[subj, "ts_vel_corr_pos_" + s] = pos_corr
        df.loc[subj, "ts_vel_corr_neg_" + s] = neg_corr
        
        df.loc[subj, "pk_freq_" + s] = pk_fq
        df.loc[subj, "out_times_" + s] = times[ix_inx][-1] - times[ix_inx][0]
        df.loc[subj, "out_fps_" + s] = TS_preds[subj]['sfreq_ori'][0][ix_inx]
    
    # r_feat = [c for c in df.columns if c[-1] == "r"]
    # l_feat = [c for c in df.columns if c[-1] == "l"]
     
    # len_sum = (times[0].size + times[3].size )
    # weigh_rl = [times[0].size/len_sum, times[3].size/len_sum]
     
    # for c_r, c_l in zip(r_feat, l_feat):
    #     df[c_r[:-1]+"b"] = df[c_r]*weigh_rl[0] + df[c_l]*weigh_rl[1]
    if do_plot:
        fname = filename_creator(subj, paths['out']+"TS_feats_imgs")
        fig.savefig(fname) 
        _ = [a.cla() for a in axs]
Finger_tapping = pd.concat([df, df_beh],  axis=1, join='inner')
n_sbj = Finger_tapping.shape[0]

fname_out = paths['out'] + f'FT_feat_{n_sbj}subj_{BP_filr[0]}_{BP_filr[1]}hz_{sfreq_common}Fs_{model_name}.csv'

assert(n_sbj == len(subjs))
Finger_tapping.to_csv(fname_out)   

runfile('/data/github/DeepNMA/SS04_feature_ML_class.py', wdir='/data/github/DeepNMA')
# runfile('/data/github/DeepNMA/SS04_feature_ML_regres.py', wdir='/data/github/DeepNMA')
