#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:49:04 2020

@author: adonay
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import (Perceptron, RidgeClassifier,
                                  PassiveAggressiveClassifier, BayesianRidge,
                                  RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
import utils_io as uio
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import matplotlib        
matplotlib.rcParams.update({'font.size': 20}) 


def make_groups(g_type):
    """ g_type can be = ["Ataxia", "PD" ("Control")], or Ataxia/PD severity """
    if len(g_type) == 2:
        g1, g2 = g_type[0], g_type[1] 
        msk_1 = dataset_r['gen_diagnosis']== g1
        
    elif g_type == "Mild Ataxia":
        dominant_arm = [ 'bars_arm_R' if s == "R" else 'bars_arm_L'  for s in dataset_r['hand']]        
        dominant_score = np.array([dataset_r[c].iloc[i] for i, c in enumerate(dominant_arm)])
        
        g1, g2 = 'Ataxia', "Control"
        msk_1 = (dataset_r['gen_diagnosis'] == g1) & (dominant_score < .6)
        
    elif g_type == "Mild PD":
        dominant_arm = ['updrs_brady_FT_R' if s == "R" else 'updrs_brady_FT_L'  for s in dataset_r['hand']]        
        dominant_score = np.array([dataset_r[c].iloc[i] for i, c in enumerate(dominant_arm)])
        
        g1, g2 = 'PD', "Control"
        msk_1 = (dataset_r['gen_diagnosis'] == g1) & (dominant_score < .6)
    
    msk_2 = dataset_r['gen_diagnosis']== g2  
    return msk_1.values + msk_2.values, g1, g2, msk_1.values, msk_2.values 


def log_reg_search(x_scaled, y, cv):        
    clf = LogisticRegression(tol=0.01)    
    param_grid = {'C': [.1, .5, .7, 1, 1.2, 1.5, 1.75, 2, 2.5, 3],
                  'solver': ['liblinear', 'saga'],
                  'penalty': ['l1', 'l2']}
    g_search = GridSearchCV(clf, param_grid, cv=cv)
    g_search.fit(x_scaled, y)
    
    print(g_search.best_estimator_, g_search.best_estimator_.penalty)
    
    C = g_search.best_estimator_.C
    solver = g_search.best_estimator_.solver
    penalty = g_search.best_estimator_.penalty
    return C, solver, penalty


def make_feats(data, method):
    
    if method == "feat_types":
        feat_in = [n for n in dataset_r.columns if n.split("_")[0][:2] in ['ts', 'pk', 'th' ]]
        data = data[feat_in].values
        return data, feat_in
    
    elif method == "PCA":
        
        
        pcas = []
        pca_weigh = []
        pca_desc = []
        feat_types = ["thth", "pkth", "pk", "th", "ts"]
        feats_in = []
        for ft in feat_types:
            feat_in = [n for n in dataset_r.columns if n.split("_")[0] == ft]
            # npcs = max(2, int(len(feat_in)/4))
            pca = PCA(n_components=2, svd_solver='full')
            
            data_f = data[feat_in]
            feats_in.append(feat_in)
            pcas.append(pca.fit_transform(data_f.values))
            pca_weigh.append(pca.components_)
            pca_desc.append (np.array([ft+"_0", ft+"_1"]))
            
        data_f, pca_desc = np.hstack(pcas), np.hstack(pca_desc)
        return data_f, pca_desc, pca_weigh, feats_in


def make_classification(x_scaled, y, cv):
    C, solver, penalty = log_reg_search(x_scaled, y, cv)
    # clf = LogisticRegressionCV(Cs=[.1, .5, .7, 1, 1.2, 1.5, 1.75, 2, 2.5, 3], cv=cv, penalty='l1', tol=0.01, solver='saga')

    print('Model Name,  \tauc   \tacc  \tPPV  \tSens   \tSpec')
    for clf, desc in (
                (RidgeClassifier(tol=1e-2, solver="saga"), "Ridge Class\t"),
                (LogisticRegression(C=C, penalty=penalty, tol=0.01, solver=solver), "LogReg L1 \t"),
                (SVC(), "SVC  \t\t")
                ):
        pred = cross_val_predict(clf, x_scaled, y, cv=cv)
        tp = np.sum(y[y==1] == pred[y==1])/ sum(y==1)
        tn = np.sum(y[y==0] == pred[y==0])/ sum(y==0)
        
        auc = np.mean(cross_val_score(clf, x_scaled, y, cv=cv, scoring='roc_auc'))
        acc = np.mean(cross_val_score(clf, x_scaled, y, cv=cv, scoring='accuracy'))
        prec = np.mean(cross_val_score(clf, x_scaled, y, cv=cv, scoring='precision'))
        # rec = np.mean(cross_val_score(clf, x_scaled, y, cv=cv, scoring='recall'))
        
        print(f'{desc} \t{auc:.2f} \t{acc:.2f} \t{prec:.2f} \t{tp:.2f} \t{tn:.2f}')
        
# Path definitions
root_dir = '/home/adonay/Desktop/projects/Ataxia'
model_name = '_resnet152_FingerTappingJan29shuffle1_650000'
paths = uio.get_paths(model_name, root_dir)

sfreq_common, BP_filr, n_sbj  = 60, [1, 10], 350
fname_out = paths['out'] + f'FT_feat_{n_sbj}subj_{BP_filr[0]}_{BP_filr[1]}hz_{sfreq_common}Fs_{model_name}.csv'
df = pd.read_csv(fname_out, index_col=0)
dataset = df.copy()

# dataset = dataset[ dataset['age'] >20]
side = "_r"
contrasides =  [ s for s in ["_l", "_r","_b"] if not s == side]
cols_out = [col for col in dataset.columns if col[-2:] in contrasides]
dataset_r = dataset.copy()#.drop(cols_out, axis=1)

# Classification
msk_grps, g1, g2, msk1, msk2 = make_groups("Mild Ataxia")#["Ataxia", 'Control']) #"Mild PD")#['Ataxia', 'Control'])

print(f"{g1} vs {g2}")
data = dataset_r.copy()
data = data.iloc[msk_grps, :]
y = msk1[msk_grps] + 0 

 
feat_meth = "PCA"
if feat_meth == "feat_types":
    data, feat_in = make_feats(data, feat_meth)
elif feat_meth == "PCA":
    data, pca_desc, pca_weigh, feats_in = make_feats(data, feat_meth)
    
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X=data)

n_CV=10
cv = StratifiedKFold(n_CV)
make_classification(x_scaled, y, cv)
    


clf = LogisticRegressionCV(Cs=[.1, .5, .7, 1, 1.2, 1.5, 1.75, 2, 2.5, 3], cv=cv, penalty='l1', tol=0.01, solver='saga')
probs = cross_val_predict(clf, x_scaled, y, cv=cv, method='predict_proba')

probs_corr = np.array([p[g] for p, g in zip(probs, y)])
# C, solver = log_reg_search(x_scaled, y, cv)
# clf, desc = LogisticRegression(C=C, penalty='l1', tol=0.01, solver='saga'), f"LogReg L1  C= {C}"
# 

# p_pred, = np.where(y[y==1] != pred[y==1])
# n_pred, = np.where(y[y==0] != pred[y==0])


data_f = dataset_r.copy()
data_f = data_f.iloc[msk_grps, :]

ages = data_f['age'].values

fig, axs = plt.subplots(1,2, figsize=(10,5))
axs[0].scatter(ages[msk1[msk_grps]], probs_corr[msk1[msk_grps]])
axs[0].scatter(ages[msk2[msk_grps]], probs_corr[msk2[msk_grps]])
axs[0].axhline(.5, color='r')

axs[0].set_title(f'Age - probab correct class {g1} and {g2}')

axs[1].set_title(f'Age dist {g1} and {g2}')
axs[1].hist(ages[msk1[msk_grps]], alpha=.5)
axs[1].hist(ages[msk2[msk_grps]],alpha=.5)

# fig, axs = plt.subplots(1,2, figsize=(10,5))
# axs[0].hist(data_f['age'].iloc[p_pred])
# axs[1].hist(data_f['age'].iloc[n_pred])
# axs[0].set_title(f"False negatives {sum(pred[y==1]==0)}/{y[y==1].size}")
# axs[1].set_title(f"False positives {sum(pred[y==0]==1)}/{y[y==0].size}")



# clf.fit(x_scaled,y)
# predictions = clf.predict(x_scaled)
# clf.score(x_scaled,y)

# par_bool = clf.coef_[0]!=0
# n_par = np.arange(np.sum(par_bool))


# yticks = feat_in if feat_meth == "feat_types" else pca_desc

    
# plt.figure(figsize=(15,15)), plt.barh(n_par, clf.coef_[0,par_bool])
# plt.yticks(n_par, yticks[par_bool]) 
# plt.xlabel('Coeff')
# plt.title(f'{desc} coefficients {g1} vs {g2}')
# plt.tight_layout()

# if feat_meth == "PCA":
#     # plot components 
#     pcs = yticks[par_bool]
#     n_plts = len(pcs)
    
#     fig, axs = plt.subplots(1,n_plts, figsize=(15,15))
#     for i, pc in enumerate(pcs):
#         ix, = np.where(pc == pca_desc)
#          # pca_weight sahpe = (2,n)
#         pc_w, = pca_weigh[int(ix/2)][ix%2,:]
#         w_bool = np.abs(pc_w) > 10e-5
#         axs[i].barh(np.arange(pc_w[w_bool].size), pc_w[w_bool])
#         yticks = np.array(feats_in[int(ix/2)])
#         axs[i].set_yticks(np.arange(len(yticks[w_bool])))
#         axs[i].set_yticklabels(yticks[w_bool])
#         axs[i].set_xlabel('Coeff ' + pc)
#     fig.tight_layout() 
# from sklearn.model_selection import GridSearchCV

# from sklearn.pipeline import Pipeline
# param_grid = {
# 'pca__n_components': np.arange(2,48),
# 'log__C':[.1, .5, .7, 1, 1.2, 1.5, 1.75, 2, 2.5, 3]}
        
# grid_search = GridSearchCV(Pipeline([('pca', pca), ('log',  clf) ]), param_grid, cv=10)


# grid_search.fit(x_scaled, y)
# print(grid_search.best_estimator_)
    


# from sklearn.model_selection import permutation_test_score, StratifiedKFold
# clf, desc = RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Class\t"
# cv = StratifiedKFold(10)
# score, permutation_scores, pvalue = permutation_test_score(
#     clf,x_scaled, y, scoring="accuracy", cv=cv, n_permutations=1000, n_jobs=1)

# print("Classification score %s (pvalue : %s)" % (score, pvalue))


# plt.figure()
# plt.hist(permutation_scores, 20, label='Permutation scores',
#          edgecolor='black')
# ylim = plt.ylim()
# plt.vlines(score, ylim[0], ylim[1], linestyle='--',
#           color='g', linewidth=3, label='Classification Score'
#           f'pvalue {pvalue:.5f}')
# plt.vlines(np.mean(permutation_scores), ylim[0], ylim[1], linestyle='--',
#           color='k', linewidth=3, label='Chance')
# plt.title("Null distribution accuracy Ataxia vs PD")

# plt.ylim(ylim)
# plt.legend()
# plt.xlabel('Score')
# plt.show()









