# coding: utf-8

def oof_cv(df, mask_trn, cols, cf = 'dt', n_folds = 5, stratified = False, fi = True, ho = False, params = {}, show_steps = True):
	import gc
	import pandas as pd
	import numpy as np
	import timeit as tm
	import itertools as it
	import lightgbm as lgb
	import seaborn as sns
	import matplotlib.pyplot as plt
	import logging as log
	
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
	from sklearn.metrics import roc_auc_score
	from sklearn.tree import DecisionTreeClassifier
	from tqdm import tnrange, tqdm, tqdm_notebook
	from imblearn.over_sampling import SMOTE

	from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

	tqdm.pandas()
	warnings.filterwarnings('ignore')
	
    if stratified: 
        print('Stratified\n') 
        cv = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 42) 
    else: 
        print('Not stratified\n') 
        cv = KFold(n_splits = n_folds, shuffle = True, random_state = 42) 

    scores = {} 
    trn_predict = np.zeros(df[mask_trn].shape[0]) 
    val_predict = np.zeros(df[mask_trn].shape[0]) 
    tst_predict = np.zeros(df[~mask_trn].shape[0]) 
    ROC_AUC_val = 0 
    ROC_AUC_trn = 0 
    if fi: 
        df_feature_importance = pd.DataFrame() 
    
    print('Number of columns = %s \n' % len(cols)) 
    # tqdm(, leave = False) 
    for fold, (trn_idx, val_idx) in enumerate(cv.split(df.loc[mask_trn, cols], df.loc[mask_trn, 'TARGET'])): 
        
        if show_steps: 
            print('1. Cross-validation data split. {}'.format(datetime.now().strftime(format = '%Y-%m-%d %H:%M:%S'))) 
        trn_x = df.loc[mask_trn, cols].iloc[trn_idx] 
        val_x = df.loc[mask_trn, cols].iloc[val_idx] 
        
        trn_y = df.loc[mask_trn, 'TARGET'].iloc[trn_idx] 
        val_y = df.loc[mask_trn, 'TARGET'].iloc[val_idx] 
        
        oof_val_start_time = tm.default_timer() 
        
        # model and parameters 
        if cf == 'dt': 
            if params == {}: 
                params = {'criterion'        : 'gini', 
                          'max_depth'        : 7, 
                          'min_samples_split': 4, 
                          'min_samples_leaf' : 3, 
                          'class_weight'     : 'balanced', 
                          'max_features'     : None, 
                          'random_state'     : 42 
                          } 
            model = DecisionTreeClassifier(**params) 
        elif cf == 'lgr': 
            if params == {}: 
                params = {'C'           : 0.1, 
                          'solver'      : 'saga', 
                          'class_weight': 'balanced', 
                          'max_iter'    : 1000, 
                          'verbose'     : 1, 
                          'random_state': 0, 
                          'n_jobs'      : 8 
                          } 
            model = LogisticRegression(**params) 
        elif cf == 'ext': 
            if params == {}: 
                params = {'criterion'        : 'gini', 
                          'max_depth'        : 10, 
                          'min_samples_split': 8, 
                          'min_samples_leaf' : 4, 
                          'class_weight'     : 'balanced', 
                          'max_features'     : None, 
                          'random_state'     : 42 
                          } 
            model = ExtraTreeClassifier(**params) 
        elif cf == 'rf': 
            if params == {}: 
                params = {'n_estimators'            : 77, 
                          'criterion'               : 'gini', 
                          'max_depth'               : 7, 
                          'min_samples_split'       : 4, 
                          'min_samples_leaf'        : 3, 
                          'min_weight_fraction_leaf': 0.0, 
                          'max_features'            : None, 
                          'max_leaf_nodes'          : None, 
                          'min_impurity_decrease'   : 0.0, 
                          'min_impurity_split'      : None, 
                          'bootstrap'               : True, 
                          'oob_score'               : False, 
                          'n_jobs'                  : 4, 
                          'random_state'            : 42, 
                          'verbose'                 : 0, 
                          'warm_start'              : False, 
                          'class_weight'            : 'balanced' 
                          } 
            model = RandomForestClassifier(**params) 
        else: 
            raise Exception('''Choose valid model: "dt"  - DecisionTreeClassifier, 
                                                   "lgr" - LogisticRegression, 
                                                   "ext" - ExtraTreeClassifier, 
                                                   "rf"  - RandomForestClassifier''') 
        
        if show_steps: 
            print('2. Fit model. {}'.format(datetime.now().strftime(format = '%Y-%m-%d %H:%M:%S'))) 
        model.fit(trn_x, trn_y) 
        
        if show_steps: 
            print('3. Predict data. {}'.format(datetime.now().strftime(format = '%Y-%m-%d %H:%M:%S'))) 
        trn_predict[trn_idx] = model.predict_proba(trn_x)[:, 1] 
        val_predict[val_idx] = model.predict_proba(val_x)[:, 1] 
        tst_predict += model.predict_proba(df.loc[~mask_trn, cols])[:, 1]/cv.n_splits 
        
        if show_steps: 
            print('4. Count score. {}'.format(datetime.now().strftime(format = '%Y-%m-%d %H:%M:%S'))) 
        trn_fold_score = roc_auc_score(trn_y, trn_predict[trn_idx]) 
        val_fold_score = roc_auc_score(val_y, val_predict[val_idx]) 
        trn_fold_gini = (trn_fold_score * 2 - 1) * 100 
        val_fold_gini = (val_fold_score * 2 - 1) * 100 
        
        ROC_AUC_trn += roc_auc_score(trn_y, trn_predict[trn_idx])/cv.n_splits 
        ROC_AUC_val += roc_auc_score(val_y, val_predict[val_idx])/cv.n_splits 
        
        if show_steps: 
            print('5. Get feature importance. {}'.format(datetime.now().strftime(format = '%Y-%m-%d %H:%M:%S'))) 
        if (fi) & (cf != 'lgr'): 
            df_fold_importance = pd.DataFrame() 
            df_fold_importance['feature'] = cols 
            df_fold_importance['importance'] = model.feature_importances_ 
            df_fold_importance['shap_values'] = abs(shap.TreeExplainer(model).shap_values(val_x)[0][:df[~mask_trn].shape[1]]).mean(axis = 0).T 
            df_fold_importance['fold'] = fold + 1 
            df_feature_importance = pd.concat([df_feature_importance, df_fold_importance], axis = 0) 
        
        oof_elapsed_time = (tm.default_timer() - oof_val_start_time)/60 
        
        print('Fold %d trained || train score = %.6f || val score = %.6f || %.3f min.' % (fold + 1, trn_fold_score, val_fold_score, oof_elapsed_time)) 
        print('############## || train gini  = %.2f    || val gini  = %.2f    || ##########\n' % (trn_fold_gini, val_fold_gini)) 
        
        del trn_x, val_x, trn_y, val_y 
        gc.collect() 
    
    if params != {}: 
        print('Model params: %s' % params) 
    
    ROC_AUC_tst = roc_auc_score(df.loc[~mask_trn, 'TARGET'], tst_predict) 
    
    Gini_trn = (ROC_AUC_trn * 2 - 1) * 100 
    Gini_val = (ROC_AUC_val * 2 - 1) * 100 
    Gini_tst = (ROC_AUC_tst * 2 - 1) * 100 
    
    print('Trn || ROC-AUC %.8f || Gini %.2f' % (ROC_AUC_trn, Gini_trn)) 
    print('Val || ROC-AUC %.8f || Gini %.2f' % (ROC_AUC_val, Gini_val)) 
    print('Tst || ROC-AUC %.8f || Gini %.2f' % (ROC_AUC_tst, Gini_tst)) 
    
    logger.info(''' 
    1. Datetime: {} 
    2. Number of Features = {} 
    3. Features: \n{} 
    4. Trn || ROC-AUC = {:.6f} || Gini = {:.2f} 
    5. Val || ROC-AUC = {:.6f} || Gini = {:.2f} 
    6. Tst || ROC-AUC = {:.6f} || Gini = {:.2f} 
    '''.format(datetime.now().strftime(format = '%Y-%m-%d %H:%M:%S'), len(cols), cols, 
               ROC_AUC_trn, Gini_trn, ROC_AUC_val, Gini_val, ROC_AUC_tst, Gini_tst)) 
    
    
    
    if (fi) & (ho == False): 
        cur_date = datetime.now().strftime(format = '%Y_%m_%d_%H_%M')
        df_feature_importance.to_csv('df_feature_importance_' + cur_date +'.csv') 
        
        return df_feature_importance 
    elif (fi == False) & (ho == False): 
        return 
    elif (fi) & (ho): 
        raise Exception('''Do not use Hyper Opt CV with FI.''') 
    else: 
        return ROC_AUC_trn, Gini_trn, ROC_AUC_val, Gini_val, ROC_AUC_tst, Gini_tst

