#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:33:45 2020

@author: adonay
"""


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
import numpy as np
import pandas as pd
import seaborn as sns
import utils_io as uio
from pypls import meancentered_pls 


def run_pls(db, inx_feat, groups, group_col_name, n_perm=1000, n_boot=1000):
    gname = group_col_name
    # sort db based on group
    db = db.sort_values(by=gname)
    # get the number of subjects per group
    group_ns = [sum(db[gname]==t) for t in groups]
    mpls = meancentered_pls(db.iloc[:,inx_feat].values, groups=group_ns,
                            n_perm=n_perm, n_boot=n_boot, mean_centering=2)
    
    pvals = mpls.permres['pvals']
    zscores = mpls.bootres.x_weights_normed
    contrasts = mpls.y_weights
    
    print(f"Pvals are {pvals}")
    return zscores, pvals, contrasts, mpls


def plot_pls(zscores, feat_names, pvals, contrast, nplots, groups, feat_sort=True, feat_thr=3):
    
    fig, axs = plt.subplots(nplots)
    feat_sig = []
    for n_cont in range(nplots):
        
        # Plot group contrasts
        ttl = f'PLS {groups}. Contrast {n_cont+1}. Pval: {pvals[n_cont]:.4}'
        grp_ixs = np.arange(contrast.shape[0])
        cont_w = contrast[:,n_cont]
        
        axs[n_cont].set_title(ttl)
        axs[n_cont].bar(grp_ixs, cont_w)
        
        axs[n_cont].set_xticks(grp_ixs)
        axs[n_cont].set_xticklabels(groups)
        
        # Plot features
        figx, axsx = plt.subplots(1)
        ttl = f'PLS {groups}. Contrast {n_cont+1}, weights {cont_w}. Pval: {pvals[n_cont]:.4}'
        axsx.set_title(ttl)
        z_cont = zscores[:,n_cont]
        
        # Threshold features
        f_th_inx = [True if f > feat_thr or f < -feat_thr else False for f in z_cont]\
            if feat_thr is not None else np.arange(z_cont.size)
        z_cont = z_cont[f_th_inx]
        z_feat_names = feat_names[f_th_inx]
        z_inxs = np.arange(z_cont.size)
        
        inx_sort = np.argsort(z_cont)[::-1] if feat_sort \
            else z_inxs
        
        feat_sig.append(z_feat_names[inx_sort])
        axsx.barh(z_inxs, z_cont[inx_sort])
        
        axsx.set_yticks(z_inxs)
        axsx.set_yticklabels(z_feat_names[inx_sort])
        if len(z_inxs):
            axsx.set_ylim(z_inxs[0]-1, z_inxs[-1]+1)
        
            u_vals, u_cnt = feat_name_parser(feat_sig[0], parser="_")
            
            u_ix = np.argsort(u_cnt)[::-1]
            u_vals, u_cnt  = u_vals[u_ix], u_cnt[u_ix]
            feat_res = "".join(f"{v}: {c} \n" for v, c in zip(u_vals, u_cnt))
            print(feat_res)
            plt.text(10, 10, feat_res, FontSize=15)
        figx.tight_layout()
        
    fig.tight_layout()
    
    return feat_sig


# def plot_corr()

def feat_name_parser(feat_names, parser="_"):
        
    vals_all = [iii for ii in [i.split(parser) for i in feat_names] for iii in ii]
    u_vals, u_cnt = np.unique(vals_all, return_counts=True)
    
    return u_vals, u_cnt


# Set features and groups SAME STR LENGTH
exlude_ending_with = []# ["_l", "_r"]
exlude_starting_with = ["out_"]
groups = ["Ataxia", "PD"]#["Control", "Ataxia", "PD"]
group_col_name = 'gen_diagnosis'

len_end = 0 if exlude_ending_with == [] else len(exlude_ending_with[0])
len_beg = 0 if exlude_starting_with == [] else len(exlude_starting_with[0])


# Path definitions
root_dir = '/home/adonay/Desktop/projects/Ataxia'
model_name = '_resnet152_FingerTappingJan29shuffle1_650000'
paths = uio.get_paths(model_name, root_dir)

sfreq_common = 60
BP_filr = [1, 10]
n_sbj = 350

# load db
fname_out = paths['out'] + f'FT_feat_{n_sbj}subj_{BP_filr[0]}_{BP_filr[1]}hz_{sfreq_common}Fs_{model_name}.csv'

df = pd.read_csv(fname_out, index_col=0)
dataset = df.copy()
# dataset = dataset.drop(['ts_pos_std_l', 'ts_pos_std_r'], 1)

use_feat = [i for i in dataset.columns if i[-len_end:]  not in exlude_ending_with ]
use_feat = [i for i in use_feat if i[:len_beg]  not in exlude_starting_with ]
use_groups =  [True if i in groups else False for i in dataset[group_col_name]]

dataset = dataset.loc[use_groups, use_feat]


inx_feat = [ i for i, n in enumerate(dataset.columns) if n[:2] in ["ts", "pk"]]
feat_names = dataset.columns[inx_feat]


zscores, pvals, contrasts, _ = run_pls(dataset, inx_feat, groups, group_col_name)
feat_sig = plot_pls(zscores, feat_names, pvals, contrasts, nplots=2, groups=groups)

xx
for g in groups:
    dg = dataset[dataset[group_col_name]==g]
    dg = dg.iloc[:, inx_feat].corr()
    
    plt.figure(), sns.heatmap(dg, xticklabels=1,  yticklabels=1)
     
# plt.title(group)



import scipy.cluster.hierarchy as sch

inx_feat = [ i for i, n in enumerate(dataset.columns) if n[:2] in ["ts"]]


X_corr = dataset.iloc[:, inx_feat].corr()
d = sch.distance.pdist(X_corr)   
L = sch.linkage(d, method='complete')
ind = sch.fcluster(L, 0.5*d.max(), 'distance')
columns = [X_corr.columns.tolist()[i] for i in list((np.argsort(ind)))]
X_clst = X_corr.reindex(columns, axis=1)

mask = (X_clst > -.25) & (X_clst < .25)
X_clst[mask] = 0

plt.figure(), sns.heatmap(X_clst, xticklabels=1,  yticklabels=1)
plt.tight_layout()
plt.grid()
plt.matshow(X_clst)


"ts_acc_entropy_r"

var_n= "ts_vel_entropy_l"
yy = dataset.loc[dataset[group_col_name]=="Ataxia", var_n ].values
plt.figure(), plt.plot(np.ones(yy.size), yy, "ro")
yy = dataset.loc[dataset[group_col_name]=="PD", var_n ].values
plt.plot(np.zeros(yy.size), yy, "bo")
plt.title(f"{var_n} red AT, Blue PD")



b = "bars_total" # "bars_arm_L"
b = "bars_arm_L"
yy = dataset.loc[dataset[group_col_name]=="Ataxia", var_n ].values
xx = dataset.loc[dataset[group_col_name]=="Ataxia", b ].values
plt.figure(), plt.plot(xx, yy, "ro")
plt.title(f"{var_n} red AT, Blue PD")



var_n= "pk_acc_slope_r"
b = "bars_arm_R"
yy = dataset.loc[dataset[group_col_name]=="Ataxia", var_n ].values
xx = dataset.loc[dataset[group_col_name]=="Ataxia", b ].values
np.corrcoef(xx[~np.isnan(xx)],yy[~np.isnan(xx)])


