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
from sklearn import linear_model, preprocessing
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import (SGDClassifier, Perceptron, RidgeClassifier,
                                  PassiveAggressiveClassifier, BayesianRidge,
                                  Lasso, Ridge, RidgeCV, LassoCV, ElasticNetCV)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import BernoulliNB 
import utils_io as uio
from sklearn.preprocessing import StandardScaler

# Path definitions
root_dir = '/home/adonay/Desktop/projects/Ataxia'
model_name = '_resnet152_FingerTappingJan29shuffle1_650000'
paths = uio.get_paths(model_name, root_dir)


sfreq_common = 60
BP_filr = [1, 10]
n_sbj = 350

fname_out = paths['out'] + f'FT_feat_{n_sbj}subj_{BP_filr[0]}_{BP_filr[1]}hz_{sfreq_common}Fs_{model_name}.csv'
# fname_out = paths['out'] + f'FT_feat_{n_sbj}subj_{sfreq_common}Fs_{model_name}.csv'


df = pd.read_csv(fname_out, index_col=0)

dataset = df.copy()
dataset.tail()

dataset = dataset[ dataset['age'] >18]

# df_old = pd.read_csv('/home/adonay/Desktop/projects/Ataxia/Finger_tapping_feat_beh_170subj.csv', index_col=0)
# old = [i for i in df_old.index if i in dataset.index]
# dataset = dataset.loc[old]

# dataset['General Diagnosis'] = dataset['General Diagnosis'].map(
#         lambda x: {1: 'Ataxia', 2: 'PD', 3: 'Control', 4: 'MG', 5: 'Unknown',
#                    6: 'FND', 7: 'Other'}.get(x))

dataset_r = dataset.copy()

side = "_r"
contrasides =  [ s for s in ["_l", "_r","_b"] if not s == side]
cols_out = [col for col in dataset.columns if col[-2:] in contrasides]
dataset_r = dataset.copy()#.drop(cols_out, axis=1)



# drop nans
dataset_r.iloc[:,3].isna().sum()
dataset_r = dataset_r.dropna(subset=['ts_vel' + side])

# See correlations
group = 'Ataxia'
msk = dataset_r['gen_diagnosis']== group
corr = dataset_r.loc[msk,:].corr()
# sns.set(rc={'figure.figsize':(11.7,8.27),"font.size":20,"axes.titlesize":20,"axes.labelsize":20},style="white")

lst_col = [ i for i, n in enumerate(corr.index) if n[:2] not in ['ts', 'pk']]
lst_inx = [ i for i, n in enumerate(corr.index) if n[:2] in ['ts', 'pk']]

corr_min = corr.iloc[lst_inx, lst_col]
# plt.figure(), sns.heatmap(corr_min, xticklabels=1,  yticklabels=1)
# plt.title(group)


# pk_jk_abs = dataset_r.loc[msk,'pk_jerk_abs' + side].values
# Atx_total =  dataset_r.loc[msk,'BARS Total'].values
# plt.figure();plt.scatter(Atx_total, pk_jk_abs)



# Classification
msk_atx = dataset_r['gen_diagnosis']=='Ataxia'
msk_PD = dataset_r['gen_diagnosis']=='PD'
# msk_ctr = dataset_r['General Diagnosis']=='Control'
msk_grps = msk_atx.values  + msk_PD.values #+ msk_ctr.values

data = dataset_r.copy()
data = data.iloc[msk_grps, :]

y = dataset_r.loc[msk_grps,'gen_diagnosis'].values
y = y == 'Ataxia'
y = y + 0

lst_inx = [ i for i, n in enumerate(dataset_r.columns) if n[:2] in ['ts', 'pk']]#, "ou", "Ag"]]
feat_in = list(data.columns[lst_inx])
data = data[feat_in]# 'BARS Total']]


x = data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X=x)

# from sklearn.decomposition import PCA, FactorAnalysis
# pca = PCA(44)
# x_scaled = pca.fit_transform(x_scaled)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

C = 2
n_CV = 10
cv = StratifiedKFold(n_CV)
cv =n_CV
print('Model Name,  \tauc   \tacc  \tprecis \trecall \ttp   \ttn')
for clf, desc in (
            (RidgeClassifier(tol=1e-2, solver="saga"), "Ridge Class\t"),
            (LogisticRegression(C=C, penalty='l1', tol=0.01, solver='saga'), "LogReg L1 \t"),
             # (LogisticRegression(C=.7, penalty='l1', tol=0.01, solver='saga'), "LogReg L1 \t")
            # (Perceptron(max_iter=50), "Perceptron  \t"),
            # (PassiveAggressiveClassifier(max_iter=50), "Pass.-Aggress."),
            # (KNeighborsClassifier(n_neighbors=10), "kNN      \t"),
            # (RandomForestClassifier(), "Random forest"),
            # # (NearestCentroid(), "NearestCentroid"),
            # (BernoulliNB(alpha=.01), "Naive Bayes\t")
            ):
    pred = cross_val_predict(clf, x_scaled, y, cv=cv)
    # pred = list(map(int, pred)) 
    # score = np.mean((y-pred)**2)
    # acc = np.sum(y== pred)/ y.size
    # print(acc, score)
    tp = np.sum(y[y==1] == pred[y==1])/ sum(y==1)
    tn = np.sum(y[y==0] == pred[y==0])/ sum(y==0)
    
    auc = np.mean(cross_val_score(clf, x_scaled, y, cv=cv, scoring='roc_auc'))
    acc = np.mean(cross_val_score(clf, x_scaled, y, cv=cv, scoring='accuracy'))
    prec = np.mean(cross_val_score(clf, x_scaled, y, cv=cv, scoring='precision'))
    rec = np.mean(cross_val_score(clf, x_scaled, y, cv=cv, scoring='recall'))
    
    print(f'{desc} \t{auc:.2f} \t{acc:.2f} \t{prec:.2f} \t{rec:.2f} \t{tp:.2f} \t{tn:.2f}')
    

# clf, desc = RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Class\t"
# C = 1
# clf, desc = LogisticRegression(C=C, penalty='l1', tol=0.01, solver='saga'), f"LogReg L1  C= {C}"

# clf.fit(x_scaled,y)
# predictions = clf.predict(x_scaled)
# clf.score(x_scaled,y)

# n_par = np.arange(clf.coef_.size)
# plt.figure(), plt.barh(n_par, clf.coef_[0])
# plt.yticks(n_par, data.columns.to_list())#,rotation=90 )
# plt.xlabel('Coeff')
# plt.title(desc + ' coefficients Ataxia vs PD')
# plt.tight_layout()
# plt.grid()




# # from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# param_grid = {
# 'pca__n_components': np.arange(2,48),
# 'log__C':[.1, .5, .7, 1, 1.2, 1.5, 1.75, 2, 2.5, 3]}
        
# grid_search = GridSearchCV(Pipeline([('pca', pca), ('log',  clf) ]), param_grid, cv=10)


# grid_search.fit(x_scaled, y)
# print(grid_search.best_estimator_)
    
# param_grid = {'C':[.1, .5, .7, 1, 1.2, 1.5, 1.75, 2, 2.5, 3], 'solver':['liblinear', 'saga']}
# grid_search = GridSearchCV(clf, param_grid, cv=10)
# grid_search.fit(x_scaled, y)

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









