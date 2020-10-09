#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Utils for input and ouput of data
Created on Mon May 18 12:35:24 2020

@author: adonay
"""
import os.path as op
import glob
import pickle
import pandas as pd
import numpy as np
import cv2
from datetime import datetime

def get_sbj_list(path, name_len=16, sufix='.pickle' ):
    """ Get names subjects
    path : path string to files
    sufix : ending file name e.g '*.csv' 
    name_len : int, num name characters in the filename that def. sbj ID
    ------
    returns:
        pat_sbsj: list files path
        subjs : list of names subjects"""
        
    pat_sbjs = glob.glob(path + '*' + sufix)
    
    subjs = []
    for s in pat_sbjs:
        subjs.append(op.basename(s)[:name_len])
        
    return pat_sbjs, subjs


def get_sbj_folders(path_list, name_len=16, sufix='.pickle' ):

    paths_sbjs, subjs = [], [] 
    for fold in path_list:
        out = get_sbj_list(fold, name_len, sufix)
        paths_sbjs.append(out[0])
    paths_sbjs = [i for l in paths_sbjs for i in l]
    paths_sbjs.sort(key=lambda x: op.basename(x))

    subjs = [op.basename(s)[:name_len] for s in paths_sbjs]

    return paths_sbjs, subjs


def get_paths(model_name, root_dir="."):
    path = {
    "timestamps" : root_dir +"/_All_Video_Timestamps_or_Framerates/",
    "beh" : root_dir +"/2019_12_18_All_Clinical_Data.csv",
    "out" : root_dir + "/outputs/DLC_finger_tracking/"
    }
    path["out_TS"] = op.join(path['out'], f'TS_data{model_name}.pickle')
    path['outliers'] = path["out"] + f"Outlier_detect_{model_name}.csv"
    path['times'] = path["out"] + f"TS_times_{model_name}.csv"
    path['pred_qual'] = path["out"] + f"Pred_good_bad_{model_name}.csv"
    path['bad_pred'] = path["out"] + f"Bad_pred_{model_name}.csv"
    path['good_pred'] = path["out"] + f"Good_pred_{model_name}.csv"
    path['nans_pred'] = path["out"] + f"Nans_pred_{model_name}.csv"
    return path


class results_dic():
    # times: File where beg and end TS will be stored
    # outliers: Good and bad predicted outlier positions
    # bad_pred: Bad predictions manually labeled
    # pred_qual: If a video prediction was good for analysis
    res_files = {'times':["subj", "r_beg", "r_end", "l_beg", "l_end",
                          "inspected", "r_len", "l_len", 'time_stmp'],
                 'outliers':["subj", "frames", "finger", "x_y_ax", "pos",
                          "inspected", 'time_stmp'],
                 'bad_pred':["subj", "frames", "finger", "x", "y", "inspected",
                          'time_stmp'],
                 'pred_qual':["subj", "quality", 'time_stmp'],
                 'nans_pred':["subj", "frames", 'time_stmp'],
                 'good_pred':["subj", "frames", 'time_stmp'],
        }
    def __init__(self, path):
        self.path = path
        for k, v in self.res_files.items():
             if not op.exists(path[k]):
                 d = pd.DataFrame(columns=v)
             else:
                 d = pd.read_csv(path[k], index_col=0)
             setattr(self, k, d)

    def add_vals(self, res_type, vals):
        "Add vals to result type csv, vals = [subj, values]"
        if not hasattr(self, res_type):
            raise AttributeError ("No such result type")
        rt = getattr(self, res_type)
        rt.reset_index(drop=True, inplace=True)

        # Add input timestamp
        time_stmp = datetime.now().strftime("%d-%b-%Y (%H:%M)")
        vals.append(time_stmp)

        # Get subject row index
        subj = vals[0]
        if not rt["subj"].str.contains(subj).any():
            rt_len = len(rt.index)
            ix = 0 if rt_len == 0 else rt_len + 1
        else:
            ix = np.where(rt["subj"].str.contains(subj))[0]
            assert(ix.size==1), 'Error, subject in multiple or None rows'
            ix = ix[0]

        # Add vals in subj row or on new row
        rt.loc[ix,:]= vals
        setattr(self, res_type, rt)

        # Save csv with new vals
        rt.to_csv(self.path[res_type])

    def remove_vals(self, res_type, subj):
        "Remove vals to result type csv, vals = [subj, values]"
        if not hasattr(self, res_type):
            raise AttributeError ("No such result type")
        rt = getattr(self, res_type)

        # Remove subject rows
        rt = rt[~(rt["subj"] == subj)]
        rt.reset_index(inplace=True, drop=True)
        setattr(self, res_type, rt)

        # Save csv with new vals
        rt.to_csv(self.path[res_type])


    def add_outliers(self, out_checked, subj):
        # out_checked = [finger num, ax num, sample, pred, inspect]
        time_stmp = datetime.now().strftime("%d-%b-%Y (%H:%M)")
        self.outliers.reset_index(drop=True, inplace=True)
        
        for out in out_checked:
            fing, ax, sampl, val, insp = out
            vals = [subj, sampl, fing, ax, val, insp, time_stmp]
            tmp = self.outliers

            # Check if outlier already exsists
            s_ix = None
            if any(tmp["subj"]==subj):
                tmp = tmp[tmp["subj"]==subj]
                if any(tmp["frames"]==sampl):
                    tmp = tmp[tmp["frames"]==sampl]
                    if any(tmp["finger"]==fing):
                        tmp = tmp[tmp["finger"]==fing]
                        if any(tmp["x_y_ax"]==ax):
                            tmp = tmp[tmp["x_y_ax"]==ax]
                            s_ix = tmp.index.to_list()[0]
            if s_ix is None:
                s_ix = len(self.outliers)
            # Add values
            self.outliers.loc[s_ix,:] = vals
        self.outliers.to_csv(self.path["outliers"])

    def add_bad_pred(self, pred_corrected, subj):
        # pred_corrected = [frames, finger num,x, y]
        time_stmp = datetime.now().strftime("%d-%b-%Y (%H:%M)")
        self.bad_pred.reset_index(drop=True, inplace=True)

        if len(pred_corrected) == 0: # detect this step done
            pred_corrected = [[np.nan, np.nan, np.nan, np.nan]]
        for pred in pred_corrected:
            sampl, fing, x, y = pred
            vals = [subj, sampl, fing, x, y, 'yes', time_stmp]
            tmp = self.bad_pred

            # Check if outlier already exsists
            s_ix = None
            if any(tmp["subj"]==subj):
                tmp = tmp[tmp["subj"]==subj]
                if any(tmp["frames"]==sampl):
                    tmp = tmp[tmp["frames"]==sampl]
                    if any(tmp["finger"]==fing):
                        tmp = tmp[tmp["finger"]==fing]
                        s_ix = tmp.index.to_list()[0]
            if s_ix is None:
                s_ix = len(self.bad_pred)
            # Add values
            self.bad_pred.loc[s_ix,:] = vals
        self.bad_pred.to_csv(self.path["bad_pred"])

    def subj_vals(self, res_type, subj):
        rt = getattr(self, res_type)
        return rt.loc[rt["subj"].str.contains(subj)]

    def get_times(self, subj):
        info = self.subj_vals("times", subj)
        info = info[['r_beg', 'r_end', 'l_beg', 'l_end']].to_numpy()
        info = info.astype("int")
        return np.squeeze(info)

def load_timestamps(subj, paths):
    fname_tstm = f"{paths['timestamps']}{subj}_Finger_Tapping_ts.xlsx"
    timestmp = pd.read_excel(fname_tstm,header=None).values
    timestmp -= timestmp[0]
    return timestmp


def load_finger_pred(path_s, pred_theshold):
    fingers = pd.read_csv(path_s).iloc[2:, 1:].to_numpy().astype("float")
    # (R-index,thumb, wrist, L-index,thumb, wrist) x (x, y, prob) x frames
    fingers = np.reshape(fingers, (-1, 6, 3))
    fingers = np.transpose(fingers, (1, 2, 0))
    prob = fingers[:,2,:]
    f_prob = np.tile(prob, (3,1,1)).transpose((1, 0, 2))
    fingers[f_prob < pred_theshold] = None
    fingers = fingers[:,:2,:]
    return fingers, prob

# def import_previous_res(res, paths, root_dir, previous_model_name=None):
    
#     if previous_model_name is None or len(res.times) > 0:
#         return res

#     path_old = get_paths(previous_model_name, root_dir)
#     res_old = results_dic(path_old)
#     res_with_insp = [ k for k, v in res.res_files.items() if "inspected" in v]
    
#     for res_type in res_with_insp:
#         rt = getattr(res, res_type)
#         if len(rt)==0:
#             rt_old = getattr(res_old, res_type)
#             rt_old['inspected'] = 'no'
#             rt_old['time_stmp'] = 'NaN'
#             res.(resType)
            





def get_prediction_frame(path_s, subj, frame_num):

    dir_vid = op.dirname(path_s)
    path_vid = glob.glob(op.join(dir_vid, subj + '*.avi'))
    
    cap = cv2.VideoCapture(path_vid[0])

    if cap.isOpened():
        hasFrame, frame = cap.read()
    else:
        raise AttributeError('Video exists but cannot be opened')
        
    cap.set(1, frame_num)
    ret, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

