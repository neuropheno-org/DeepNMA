#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:49:04 2020

@author: adonay
"""



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
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


def make_groups(df, beh_Var, group_col_name, g_type):
    """ 
    beh_Var : name column with clinical score (last letter is D(ominant), N(on dom), R or L)
    group_col_name : name column with groups
    g_type can be : "At + Control", "PD + Control", "Common", "Common + Control"
    dominant : 1 for dominant hand, 0 for non, or  None
    """
    
    if beh_Var[-1] == "D":
        dominant = 1
    elif beh_Var[-1] == "N":
        dominant = 0  
    else:        
         dominant=None

    if dominant == 1:
        dominant_arm = [beh_Var[:-1] + 'L' if s == "L" else beh_Var[:-1] + 'R' for s in df['hand']]               
        beh_score = np.array([dataset_r[c].iloc[i] for i, c in enumerate(dominant_arm)])
        beh_score = pd.Series(beh_score)
        
    elif dominant == 0:
        nondominant_arm = [beh_Var[:-1] + 'R' if s == "L" else beh_Var[:-1] + 'L' for s in df['hand']]               
        beh_score = np.array([dataset_r[c].iloc[i] for i, c in enumerate(nondominant_arm)])
        beh_score = pd.Series(beh_score)
        
    elif dominant is None:
        beh_score = df[beh_Var].copy()
    
    
    msk = []
    if "AT" in g_type or  "Common"  in g_type:
        msk.append( df[group_col_name] == "Ataxia")
    if "PD" in g_type or  "Common" in g_type:
        msk.append( df[group_col_name] == "PD")   
        
    if "CTR" in g_type: 
        # Set NaN values to 0
        ctr_bool = df[group_col_name] == "Control"
        nan_ctrl = (beh_score.isnull()) & (ctr_bool)
        beh_score.loc[nan_ctrl] = 0.0       
        msk.append(df[group_col_name] == "Control")   

    msk = np.array( np.sum(np.stack(msk).T, 1), dtype=bool) 
    beh_score = beh_score.iloc[msk]
    
    return msk, beh_score


def make_feats(data, method):
    
    if method == "feat_types":
        feat_in = np.array([n for n in dataset_r.columns if n.split("_")[0][:2] in ['ts', 'pk', 'th' ]])
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



# Path definitions
root_dir = '/home/adonay/Desktop/projects/Ataxia'
model_name = '_resnet152_FingerTappingJan29shuffle1_650000'
paths = uio.get_paths(model_name, root_dir)
group_col_name = 'gen_diagnosis'

sfreq_common = 60
BP_filr = [1, 10]
n_sbj = 350

fname_out = paths['out'] + f'FT_feat_{n_sbj}subj_{BP_filr[0]}_{BP_filr[1]}hz_{sfreq_common}Fs_{model_name}.csv'
# fname_out = paths['out'] + f'FT_feat_{n_sbj}subj_{sfreq_common}Fs_{model_name}.csv'

out_csv = paths['out'] + f'RES_FT_feat_{n_sbj}subj_{model_name}.csv'
fid= open(out_csv, mode='w')
csv_writer = csv.writer(fid, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
csv_writer.writerow(['Group', "Measure", "Model", "corr_pred", 'R2'])

df = pd.read_csv(fname_out, index_col=0)
# df['General Diagnosis'] = dataset['General Diagnosis'].map(
#         lambda x: {1: 'Ataxia', 2: 'PD', 3: 'Control', 4: 'MG', 5: 'Unknown',
#                    6: 'FND', 7: 'Other'}.get(x))
side = "_r"
contrasides =  [ s for s in ["_l", "_r","_b"] if not s == side]
cols_out = [col for col in df.columns if col[-2:] in contrasides]
dataset_r = df.copy()#.drop(cols_out, axis=1)


runs = [("AT + PD + CTR", 'common_arm_score_R'),
        ("AT + PD + CTR", 'common_arm_score_L'), 
        ("AT + CTR", 'common_arm_score_R'),
        ("AT + CTR", 'common_arm_score_L'),       
        ("PD + CTR", 'common_arm_score_R'),
        ("PD + CTR", 'common_arm_score_L'),  
        ("AT + PD", 'common_arm_score_R'),
        ("AT + PD", 'common_arm_score_L'), 
        ]
at_scores = ['bars_arm_L','bars_arm_R','bars_leg_L','bars_leg_R',
             'bars_oculomotor','bars_gait','bars_total']
at_runs = [("AT", s) for s in at_scores]

pd_scores = ['updrs_rest_trem_L','updrs_rest_trem_R','updrs_post_trem_L',
             'updrs_post_trem_R','updrs_rigid_L','updrs_rigid_R',
             'updrs_brady_FT_L','updrs_brady_FT_R','updrs_brady_OC_L',
             'updrs_brady_OC_R','updrs_brady_RAHM_L','updrs_brady_RAHM_R',
             'updrs_arm_total_L','updrs_arm_total_R','updrs_gait', 'updrs_total']

pd_runs = [("PD", s) for s in pd_scores]

runs = runs + at_runs + pd_runs


for g_type, beh_Var in runs:
            
    if beh_Var[-2:] == "_R":
        beh_Var = beh_Var[:-2] + "_D"
    elif beh_Var[-2:] == "_L":
        beh_Var = beh_Var[:-2] + "_N"
    
    # print( f"\t{g_type} : {beh_Var}")
    msk, beh_score = make_groups(dataset_r, beh_Var, group_col_name, g_type)
    
    data = dataset_r.copy().iloc[msk, :]
    y = beh_score.values
    
    data = data.iloc[~np.isnan(y)]
    y = y[~np.isnan(y)]
    gps = data[group_col_name]
    
    
    feat_meth = "PCA"
    if feat_meth == "feat_types":
        data, feat_in = make_feats(data, feat_meth)
    elif feat_meth == "PCA":
        data, feat_in, pca_weigh, feats_in = make_feats(data, feat_meth)
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X=data)
       
    
    n_CV =  20
    models = (
        # (Ridge(alpha=0.5), "Ridge Reg."),
        (RidgeCV(cv=n_CV), "Ridge Reg. CV"),
        # (LassoCV(cv=n_CV, max_iter=90000,  tol=1e-2), "Lasso CV   "),
    #        (LassoCV(), "Lasso CV"),
        # (BayesianRidge(n_iter=9000, tol=1e-2), "Bayesn Rdge"),
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
        corr_pred = np.corrcoef(y, pred)[0,1]       
        # print(f'{desc}, \t corr:{corr_pred:.2f}, \t R2:{corr_pred**2:.2f}  \t MSE:{score :.3f}; null score: {np.mean((y - y.mean())**2):.3f} ')
        print(f'{g_type}- {beh_Var}: {desc}, \t\t corr:{corr_pred:.2f}, \t R2:{corr_pred**2:.2f} ')
        
        fig, axes = plt.subplots(1, figsize=(25,8))
        sns.regplot(x=y, y=pred, ax=axes)
        axes.set_title(f'{g_type}- {beh_Var}: {desc}, \t\t corr:{corr_pred:.2f}, \t R2:{corr_pred**2:.2f} ')   
        
        csv_writer.writerow([g_type, beh_Var, desc, corr_pred, corr_pred**2])

fid.close()        
        
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
    
        # for clf, desc in models:
            
        #     rfecv = RFECV(estimator=clf, step=1,  cv=n_CV, scoring='neg_mean_absolute_error', n_jobs=-1) # min_features_to_select=5,
        #     rfecv.fit(x_scaled, y)
        #     y_pred = cross_val_predict(rfecv.estimator_, x_scaled, y, n_jobs=-1)
        #     corr_pred = np.corrcoef(y, y_pred)[0,1]
        #     print(f"Optimal number of features :{rfecv.n_features_}, corr: {corr_pred}")
            
        #     feat_ix = np.where(rfecv.ranking_ == 1 )[0]       
        #     n_par = np.arange(len(feat_ix))
        #     assert(n_par.size == rfecv.n_features_)   
        

             
        #     axes[1].barh(n_par, rfecv.estimator_.coef_)
        #     axes[1].set_yticks(n_par)
        #     axes[1].set_yticklabels(feat_in[feat_ix])        
        #     axes[1].set_title( desc + ' Coeff pred '+ beh_Var)        
            
            
        #     # Plot number of features VS. cross-validation scores
        #     axes[0].set_xlabel("Number of features selected")
        #     axes[0].set_ylabel("Cross validation score ")
        #     axes[0].set_title(beh_Var + " " + g_type)
        #     axes[0].plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
             
            
           
        #     sns.regplot(x=y, y=y_pred, ax=axes[2])
        #     pad = (max(y) - min(y))/10
        #     axes[2].set_xlim(min(y) -pad, max(y)+pad)
            
        #     for g in np.unique(gps):
        #         axes[2].scatter(y[gps==g], y_pred[gps==g])
        #         # plt.figure(), plt.hist(y[gps==g])
                
        #     plt.xlabel(beh_Var), plt.ylabel('predicted '+ beh_Var)
        #     plt.title(desc + ' real vs predicted, corr: ' +
        #           f' {corr_pred:.2}, R2 {corr_pred**2:.2},')
        #     plt.show()
    
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
    








