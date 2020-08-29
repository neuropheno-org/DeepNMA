#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:46:38 2020

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

def run_step(step, subj):
    "check if step done, if overwrite true or inspected"
    sj_vals = res.subj_vals(step, subj)
    if "inspected" in sj_vals and any(sj_vals["inspected"] == "no"):
        return True

    if sj_vals.shape[0] > 0 and  OVERWRT[step]:
        res.remove_vals(step, subj)

    return sj_vals.shape[0] == 0 or  OVERWRT[step]

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


# Path definitions
root_dir = '/mnt/Data/projects/Ataxia'

pat_dlc = '/mnt/Samsung_T5/DLC/FingerTapping-Adonay-2020-01-29/'
pat_dlc_vids = [pat_dlc + 'videos/',
                 pat_dlc + 'videos_dsmpl_labeled/',
                 pat_dlc + 'new_videos/']

lab_csv_name = "CollectedData_Adonay.csv"
pat_dlc_labs= op.join(pat_dlc, "labeled-data")

extension = "_Finger_Tapping"

model_name = '_resnet152_FingerTappingJan29shuffle2_550000'
pat_sbjs, subjs = uio.get_sbj_folders(pat_dlc_vids, name_len=16,
                                      sufix=f'*{model_name}.csv')

paths = uio.get_paths(model_name, root_dir)
df_beh = pd.read_csv(paths['beh'], index_col=0)
# CSV files storing results
res = uio.results_dic(paths)


OVERWRT = {"subj_done" :False,
           'pred_qual':False,
           'times':False,
           "outliers":False,
           "bad_pred":False,
           "good_pred":False,
           "nans_pred":False}


TS_predictions = saved_TS_data(paths)
subjs_in = [i for i, s in enumerate(subjs) if s not in TS_predictions.keys()]
pat_sbjs, subjs = [[l[p] for p in subjs_in] for l in [pat_sbjs, subjs]]


n= 0
isb, path_s, subj = n, pat_sbjs[n], subjs[n]
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20, 10))
for isb, (path_s, subj) in enumerate(zip(pat_sbjs, subjs)):
    subj_data={}

    subj_diag = df_beh.loc[subj,'General Diagnosis']
    print(f'Doing s: {isb}, {subj}/{len(subjs)}')

    # Load time stamps & finger positions
    timestmp = uio.load_timestamps(subj, paths)
    fingers, prob = uio.load_finger_pred(path_s, pred_theshold=.1)
    subj_data['raw'] = fingers
    subj_data['timestamp'] = timestmp

    # preprocess TS
    int1, int2, out = sig_proc.ts_prepro(fingers, timestmp)
    relab = []

    # Do quality control prediction: good or bad
    if run_step("pred_qual", subj):
        res_qc, relab = viz.plot_check_quality(fingers, int1, int2, out, timestmp,
                                        axes, fig, subj, subj_diag, path_s)
        print(f"{subj} : {res_qc}")
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
            print(f"R/L beg-end: {res_tms[:2]}, {res_tms[2:]}, length: " +
                  f"{res_lg.astype(int)} secs")
            res.add_vals("times", list(tms_vals))

        times = res.get_times(subj)
        int1, int2, out = sig_proc.ts_prepro(fingers, timestmp, times)
        subj_data['times'] = times


        # Check and select good outliers
        if run_step("outliers", subj):
            fig2, ax = plt.subplots(1, 1, sharex=True, figsize=(20, 10))
            out_checked, relab = viz.plot_oulier_qc(out, fingers, int2, path_s,
                                                    subj, ax, fig2, relab)
            out_num = [1 if o[4] == 'good' else 0 for o in out_checked]
            rel_num = np.unique([r[0] for r in relab]).size
            print(f"done: {len(out_checked)} outliers, good: {sum(out_num)}" + 
                  f", bad: {len(out_num)-sum(out_num)}, relabaled: {rel_num}")
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

        # update relabels
        relab.extend(bd_relab)
        for rlb in bd_relab:
            if not np.isnan(int2[int(rlb[1]), 0, int(rlb[0])]):
                int2[int(rlb[1]), :, int(rlb[0])] = rlb[2:]

        # inspect suspicious
        relabeled= []
        if run_step("bad_pred", subj):
            relabeled, _ = viz.plot_ts_inspection(
                out_good, timestmp, int1, int2, path_s, subj, subj_diag, axes, fig)
            n_relab = np.unique([r[0] for r in relabeled]).size
            print(f"{subj}: {n_relab} relabeled")
            relabeled.extend(relab)
            res.add_bad_pred(relabeled, subj)
        else:
            if len(relab):
                res.add_bad_pred(relab, subj)
            cols_v = ['frames', 'finger', 'x', 'y']
            relabeled = res.subj_vals("bad_pred", subj)[cols_v].values.tolist()
            relabeled = [] if np.all(np.isnan(relabeled)) else relabeled

        for rlb in relabeled:
            if not (np.all(np.isnan(rlb)) and 
                    np.isnan(int2[int(rlb[1]), 0, int(rlb[0])])):
                int2[int(rlb[1]), :, int(rlb[0])] = rlb[2:]

    elif res_qc == "bad":
        if run_step("good_pred", subj) & run_step("bad_pred", subj):  ##TODO fix with a proper step
            ttl = f"SELECT GOOD and BAD PREDICTIONS for retraining"
            bd_relab, good_pred = viz.plot_ts_inspection(
                [], timestmp, int1, int2, path_s, subj, subj_diag, axes, fig, ttl)
            n_relab = np.unique([r[0] for r in bd_relab]).size
            n_gd = len(good_pred)
            print(f"{subj}: {n_relab} relabeled, {n_gd} good")
            res.add_vals('good_pred', [subj, good_pred])
            res.add_bad_pred(bd_relab, subj)
    else:
        ValueError("No prediction quality information")


    subj_data['TS'] = int2
    TS_predictions[subj] = subj_data
    retrn.save_labeled_data(int2, res, subj, path_s, pat_dlc_labs, extension,
                      lab_csv_name)
    saved_TS_data(paths, TS_predictions)
    axes[0].clear(); axes[1].clear(); fig.patch.set_facecolor('w')

q



