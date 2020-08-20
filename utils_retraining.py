#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 08:57:41 2020

@author: adonay
"""
import os
import os.path as op
import numpy as np
import pandas as pd
import cv2
import glob
import utils_signal_processing as sig_proc
import utils_feature_extraction as feat_ext
import utils_io as uio
import utils_visualization as viz



def save_frame_png(frame_num, subj, path_s, folder_str):
    img_path = f"{folder_str}/img{frame_num:04}.png"
    if not op.exists(img_path):
        frame = uio.get_prediction_frame(path_s, subj, frame_num)
        cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return img_path




def check_equal_res_ts(subj, res, TS):

    res_types = ["outliers", "bad_pred"]
    for rt in res_types:

        if rt == "outliers":
            cols_v = ['finger', 'x_y_ax', 'frames', 'pos', 'inspected']
            vals = res.subj_vals(rt, subj)[cols_v].values.tolist()
            for val in vals[2:]:
                if val[4] == "good":
                    i1, i2, i3 = int(val[0]), int(val[1]), int(val[2])
                    assert round(TS[i1, i2, i3]) == val[3]

        elif rt == "bad_pred":
            cols_v = ['finger', 'frames', "x", "y"]
            vals = res.subj_vals(rt, subj)[cols_v].values.tolist()
            for val in vals:
                if not all(np.isnan(val)):
                    i1, i2 = int(val[0]), int(val[1])
                    np.testing.assert_array_equal(np.round(TS[i1, :, i2]),
                                                  np.array(val[2:]))


def get_res_frames(subj, res):

    res_types = ["outliers", "nans_pred", "bad_pred", "good_pred"]
    frames = []
    for rt in res_types:
        vals = res.subj_vals(rt, subj)['frames'].values.tolist()
        if vals == [[]] or vals == []:
            continue
        if rt == "nans_pred":
            vals, = vals
            vals = eval(vals) if isinstance(vals, str) else vals
            if vals == []:
                continue
        elif rt =="good_pred" :
            vals = eval(vals[0]) if isinstance(vals[0], str) else vals[0]
        if not any(np.isnan(vals)):
            frames.extend([int(v) for v in vals])
    return np.unique(frames)


def load_label_csv(folder_str, lab_csv_name):
    f_name = op.join(folder_str, lab_csv_name)
    if op.exists(f_name):
        lab_csv = pd.read_csv(op.join(folder_str, lab_csv_name))
    else:
        # Find a labeled data csv and copy it
        template = glob.glob(f"{op.dirname(folder_str)}/*/{lab_csv_name}")[0]
        lab_csv = pd.read_csv(template)
        lab_csv = lab_csv.iloc[:2, :]
    return lab_csv, f_name


def save_labeled_data(TS, res, subj, path_s, DLC_path, extension,
                      lab_csv_name):
    """
    DLC_path : str
    
    Parameters
    ----------
    subj : str
        Name.
    path_s : str
        Path to subject predictions csv file.
    DLC_path : str
        Folder of labeled-data.
    extension : str
        Str with subject folder extension.

    Returns
    -------
    None.

    """
    

    folder_str = op.join(DLC_path, subj + extension)
    os.makedirs(folder_str, exist_ok=True)
    
    lab_csv, f_name = load_label_csv(folder_str, lab_csv_name)
    
    frames = get_res_frames(subj, res)
        
    ix = lab_csv.shape[0]
    lab_fold  = op.basename(DLC_path)
    new_frames = []
    for frm_ix in frames:
        img_path = save_frame_png(frm_ix, subj, path_s, folder_str)
        img_path = lab_fold + img_path.split(lab_fold)[1]

        pos, = TS[:,: , frm_ix].reshape((1, -1))

        ix_s = lab_csv.index[lab_csv["scorer"] == img_path].tolist()
        assert(len(ix_s)<2),  "Label repeated, check it please"
        if not ix_s:
            ix_s = ix
            new_frames.append(frm_ix)
        else: # Check if already saved
            p_csv, = lab_csv.iloc[ix_s, 1:].to_numpy().astype(np.float64)
            p_csv = [int(p) if not np.isnan(p) else p for p in p_csv]
            p_ori = [int(p) if not np.isnan(p) else p for p in pos]
            if viz.array_equal_nan(p_csv, p_ori):
                continue
            else:
                new_frames.append(frm_ix)

        lab_csv.loc[ix_s, "scorer"] = img_path
        lab_csv.iloc[ix_s, 1:] = pos
        ix = lab_csv.shape[0]

    if len(new_frames):
        lab_csv.to_csv(f_name, index=False)
