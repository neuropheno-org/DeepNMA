#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:46:38 2020

@author: adonay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils_io as uio
import utils_signal_processing as sig_proc
import utils_visualization as viz
import utils_retraining as retr

def run_step(step, subj):
    "check if step done, if overwrite true or inspected"
    sj_vals = res.subj_vals(step, subj)
    if "inspected" in sj_vals and any(sj_vals["inspected"] == "no"):
        return True

    if sj_vals.shape[0] > 0 and  OVERWRT[step]:
        res.remove_vals(step, subj)

    return sj_vals.shape[0] == 0 or  OVERWRT[step]


# Path definitions
pat_dlc = '/mnt/Samsung_T5/DLC/FingerTapping-Adonay-2020-01-29/'
pat_dlc_vids = [pat_dlc + 'videos/',
                 pat_dlc + 'videos_dsmpl_labeled/',
                 pat_dlc + 'new_videos/']

lab_csv_name = "CollectedData_Adonay.csv"
DLC_path= "FingerTapping-Adonay-2020-01-29/labeled-data"
extension = "_Finger_Tapping"

model_name = '_resnet152_FingerTappingJan29shuffle2_550000'
pat_sbjs, subjs = uio.get_sbj_folders(pat_dlc_vids, name_len=16,
                                      sufix=f'*{model_name}.csv')
paths = uio.get_paths(model_name)
df_beh = pd.read_csv(paths['beh'], index_col=0)
# CSV files storing results
res = uio.results_dic(paths)


OVERWRT = {'pred_qual':False,
           'times':False,
           "outliers":False,
           "bad_pred":False,
           "good_pred":False,
           "nans_pred":False}


fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20, 10))


TS_predictions = {}
n= 1
isb, path_s, subj = n, pat_sbjs[n], subjs[n]
for isb, (path_s, subj) in enumerate(zip(pat_sbjs, subjs)):
    subj_data={}

    subj_diag = df_beh.loc[subj,'General Diagnosis']
    print(f'Doing s: {isb}, {subj}')

    # Load time stamps & finger positions
    timestmp = uio.load_timestamps(subj, paths)
    fingers, prob = uio.load_finger_pred(path_s, pred_theshold=.1)
    subj_data['raw'] = fingers
    subj_data['timestamp'] = timestmp

    # preprocess TS
    int1, int2, out = sig_proc.ts_prepro(fingers, timestmp)

    # Do quality control prediction: good or bad
    if run_step("pred_qual", subj):
        res_qc = viz.plot_check_quality(fingers, int1, int2, out, timestmp,
                                        axes, fig, subj, subj_diag)
        res.add_vals("pred_qual", [subj, res_qc])
    else:
        res_qc = res.subj_vals("pred_qual", subj)["quality"].to_list()[0]
        viz.resp_corr_fig_bkg(fig, res_qc)
    subj_data['pred_qual'] = res_qc

    if res_qc == "good":
        # Get beginning and end right & left tapping
        if run_step("times", subj):
            res_tms, res_lg = viz.plot_get_times(fingers, int2,timestmp, axes,
                                                 fig, subj, subj_diag)
            tms_vals = np.hstack((subj, res_tms, 'yes', res_lg))
            res.add_vals("times", list(tms_vals))

        times = res.get_times(subj)
        int1, int2, out = sig_proc.ts_prepro(fingers, timestmp, times)
        subj_data['times'] = times

        relab = []
        # Check and select good outliers
        if run_step("outliers", subj):
            fig2, ax = plt.subplots(1, 1, sharex=True, figsize=(20, 10))
            out_checked, relab = viz.plot_oulier_qc(out, fingers, int2, path_s,
                                                    subj, ax, fig2, relab)
            res.add_outliers(out_checked, subj)
            plt.close(fig2)
        else:
            cols_v = ['finger', 'x_y_ax', 'frames', 'pos', 'inspected']
            out_checked = res.subj_vals("outliers", subj)[cols_v].values.tolist()
        # change int2 good outliers
        int2 = sig_proc.restore_bad_outliers(out_checked, int1, int2)
        out_good = [o for o in out_checked if 'good' in o]

        # Selects nans for retraining
        if run_step("nans_pred", subj):
            nans_pred = sig_proc.get_dist_nans(fingers, int2, max_nans=5)
            gd_nan_pred, bd_relab, relab = viz.nan_inspec(nans_pred, path_s,
                                                          subj,int1, int2, relab)
            res.add_vals('nans_pred', [subj, gd_nan_pred])
        else:
            bd_relab = []
            gd_nan_pred, = res.subj_vals("nans_pred",
                                         subj)['frames'].values.tolist()
            gd_nan_pred = eval(gd_nan_pred) if gd_nan_pred else []

    elif res_qc == "bad":
        if run_step("good_pred", subj):
            ttl = "SELECT GOOD PREDICTIONS " + subj_diag
            bd_relab, good_pred = viz.plot_ts_inspection([], timestmp, int1, int2,
                                                  path_s, subj, ttl, axes, fig)
            res.add_vals('good_pred', [subj, good_pred])
    else:
        ValueError("No prediction quality information")

    # update relabels
    relab.extend(bd_relab)
    for rlb in bd_relab:
        int2[rlb[1], :, rlb[0]] = rlb[2:]

    # inspect suspicious
    relabeled= []
    if run_step("bad_pred", subj):
        relabeled, _ = viz.plot_ts_inspection(out_good, timestmp, int1, int2,
                                              path_s, subj, subj_diag, axes, fig)
        relabeled.extend(relab)
        res.add_bad_pred(relabeled, subj)
    else:
        if len(relab):
            res.add_bad_pred(relab, subj)
        cols_v = ['frames', 'finger', 'x', 'y']
        relabeled = res.subj_vals("bad_pred", subj)[cols_v].values.tolist()
        relabeled = [] if np.all(np.isnan(relabeled)) else relabeled

    for rlb in relabeled:
        if not np.all(np.isnan(rlb)):
            int2[int(rlb[1]), :, int(rlb[0])] = rlb[2:]

    subj_data['TS'] = int2
    TS_predictions[subj] = subj_data
    retr.save_labeled_data(int2, res, subj, path_s, DLC_path, extension,
                      lab_csv_name)
    axes[0].clear(); axes[1].clear(); fig.patch.set_facecolor('w')





