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
from sklearn.feature_selection import RFECV

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.pipeline import Pipeline

sns.set(font_scale = 1.5)

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

lst_col = [ i for i, n in enumerate(corr.index) if n[:2] not in ['ts', 'pk']]
lst_inx = [ i for i, n in enumerate(corr.index) if n[:2] in ['ts', 'pk', "Ag"]]

corr_min = corr.iloc[lst_inx, lst_col]
# plt.figure(), sns.heatmap(corr_min, xticklabels=1,  yticklabels=1)
# plt.title(group)


# pca = PCA(3)

# Classification
msk_atx = dataset_r['gen_diagnosis']=='Ataxia'
msk_grps = msk_atx.values  

beh_var = 'FNF (right)' #  # 'BARS Total' #'HTS (right)' # 

beh_vars = ['bars_arm_L', 'bars_arm_R', 'bars_total', 'common_arm_score_L', 'common_arm_score_R']
beh_var =  beh_vars[1]
for beh_var in beh_vars:
    lst_inx = [ i for i, n in enumerate(dataset_r.columns) if n[:2] in ['ts', 'pk']]#, "ou", "Ag"]]
    feat_in = list(dataset_r.columns[lst_inx])
    
    data = dataset_r.copy()
    data = data.loc[msk_grps, feat_in +[beh_var]]
    
    y = data.pop(beh_var)
    data = data.loc[~y.isna()]
    y = y.loc[~y.isna()]
    y = y.values
    
    
    x = data.values 
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X=x)
    
    
    
 
    # from sklearn.model_selection import LeaveOneOut
    n_CV =  10
    models = (
        # (Ridge(alpha=0.5), "Ridge Reg."),
        # (RidgeCV(cv=n_CV), "Ridge Reg. CV"),
        (LassoCV(max_iter=90000,  tol=1e-2), "Lasso CV"),
    #        (LassoCV(), "Lasso CV"),
        # (BayesianRidge(n_iter=9000, tol=1e-2), "Bayesian Ridge"),
    #        (DecisionTreeRegressor(), "DecisionTree Reg"),
        # (GradientBoostingRegressor(), 'GradBoostReg'),
        # (MLPRegressor(solver="lbfgs", max_iter=500), "MPLreg"),
        # (ElasticNetCV(max_iter=9000, tol=1e-2), "elastic net")
        )
    
    best = []
    for clf, desc in models:
        # clf = Pipeline([('pca', pca), (desc,  clf) ])
        pred = cross_val_predict(clf, x_scaled, y, cv=n_CV, n_jobs=-1 )
        score = np.mean((y-pred)**2)
        best.append((desc, score))
        print(f'{desc}, {score}; null score: {np.mean((y - y.mean())**2):.4f}')
    
    
    
    
    # best_model = np.argmin([ b[1] for b in best])
    # best_model = models[best_model]
    # predictions = cross_val_predict(best_model[0], x_scaled, y, cv=n_CV )
    
    
    # plt.figure(); sns.regplot(x=y, y=predictions )
    # plt.xlabel(beh_var), plt.ylabel('predicted '+ beh_var)
    # plt.title(best_model[1]+ ' real vs predicted, corr: ' +
    #           f' {np.corrcoef(predictions,y)[0,1]:.2}')
    # plt.tight_layout()
      
    
    # clf, desc = best_model
    
    # clf.fit(x_scaled, y)
    
    # coef_nnz = [i for i, c in enumerate(clf.coef_) if c not in [-0.0, 0.]]
    # n_par = np.arange(len(coef_nnz))
    # plt.figure(), plt.barh(n_par, clf.coef_[coef_nnz])
    # plt.yticks(n_par, data.columns[coef_nnz])
    # plt.ylabel('Coeff')
    # plt.title( desc + ' Coeff pred '+ beh_var)
    # plt.tight_layout()
    
    # x_scaled = pca.fit_transform(x_scaled)

    for clf, desc in models:
        
        rfecv = RFECV(estimator=clf, step=1,  cv=n_CV, scoring='neg_mean_absolute_error', n_jobs=-1) # min_features_to_select=5,
        rfecv.fit(x_scaled, y)
        y_pred = cross_val_predict(rfecv.estimator_, x_scaled, y, n_jobs=-1)
        corr_pred = np.corrcoef(y, y_pred)[0,1]
        print(f"Optimal number of features :{rfecv.n_features_}, corr: {corr_pred}")
        
        feat_ix = np.where(rfecv.ranking_ == 1 )[0]       
        n_par = np.arange(len(feat_ix))
        assert(n_par.size == rfecv.n_features_)        
        fig, axes = plt.subplots(1,3)
        axes[1].barh(n_par, rfecv.estimator_.coef_)
        axes[1].set_yticks(n_par)
        axes[1].set_yticklabels(data.columns.values[feat_ix])        
        axes[1].set_title( desc + ' Coeff pred '+ beh_var)
        
        
        
        # Plot number of features VS. cross-validation scores
        axes[0].set_xlabel("Number of features selected")
        axes[0].set_ylabel("Cross validation score ")
        axes[0].set_title(beh_var + " feature selection using RFE")
        axes[0].plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        
              
        
       
        sns.regplot(x=y, y=y_pred , ax=axes[2])
        pad = (max(y) - min(y))/10
        axes[2].set_xlim(min(y) -pad, max(y)+pad)
        
        plt.xlabel(beh_var), plt.ylabel('predicted '+ beh_var)
        plt.title(desc + ' real vs predicted, corr: ' +
              f' {corr_pred:.2}, R2 {corr_pred**2:.2},')
        plt.show()
    
        # fig.tight_layout()
    # 
    
    # from sklearn.model_selection import GridSearchCV
    # param_grid = {
    # 'pca__n_components': np.arange(2,48)}
    
    # # param_grid = {'C':[.1, .5, .7, 1, 1.2, 1.5, 1.75, 2, 2.5, 3], 'solver':['liblinear', 'saga']}
    # grid_search = GridSearchCV(Pipeline([('pca', pca), (desc,  clf) ]), param_grid, cv=10)
    # grid_search.fit(x_scaled, y)
    
    # print(grid_search.best_estimator_)
    
    # from sklearn.model_selection import permutation_test_score
    
    # clf, desc = LassoCV(max_iter=90000, tol=0.01), 'Lasso CV'
    # scorer = 'neg_mean_absolute_error'
    # score, permutation_scores, pvalue = permutation_test_score(
    #     clf,x_scaled, y, scoring=scorer, cv=10, n_permutations=100, n_jobs=10)
    
    # print("Classification score %s (pvalue : %s)" % (score, pvalue))
    
    
    # plt.figure()
    # plt.hist(permutation_scores, 20, label='Permutation scores',
    #          edgecolor='black')
    # ylim = plt.ylim()
    # plt.vlines(score, ylim[0], ylim[1], linestyle='--',
    #           color='g', linewidth=3, label='Regress Score'
    #           f'pvalue {pvalue:.5f}')
    # plt.vlines(np.mean(permutation_scores), ylim[0], ylim[1], linestyle='--',
    #           color='k', linewidth=3, label='Chance')
    # plt.title("Null distribution regression  " + beh_var)
    
    # plt.ylim(ylim)
    # plt.legend()
    # plt.xlabel(scorer)
    # plt.show()
    








