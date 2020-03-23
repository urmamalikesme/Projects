import numpy as np
import pandas as pd
import gc
import timeit

from sklearn.metrics import roc_auc_score, log_loss
from sklearn.inspection import permutation_importance


class CrossValidation:

    def __init__(self, cv, estimator, metric='auc'):
        metric_dict = {'auc': roc_auc_score,
                      'logloss': log_loss}
        self.cv = cv
        self.estimator = estimator
        self.metric_name = metric
        self.metric = metric_dict[metric]
        self._is_fitted = False


    def fit(self, train, cols_cv, TARGET, FIT_PARAMS, groups=None, is_silent=False):
        self.train = train
        self.cols_cv = cols_cv
        self.TARGET = TARGET
        self.FIT_PARAMS = FIT_PARAMS
        self.groups = groups
        self.is_silent = is_silent

        self.splitter = self.cv.split(self.train[self.cols_cv], self.train[self.TARGET], groups=self.groups)

        self.model_list, self.trn_score, self.val_score = [], [], []
        self.val_predict = np.zeros(self.train.shape[0])
        self.fold_data = []

        if self.is_silent:
            self.FIT_PARAMS['verbose'] = 0

        for fold, (idx_trn, idx_val) in enumerate(self.splitter):
            start_time = timeit.default_timer()

            X_trn, y_trn = self.train[self.cols_cv].iloc[idx_trn], self.train[self.TARGET].iloc[idx_trn]
            X_val, y_val = self.train[self.cols_cv].iloc[idx_val], self.train[self.TARGET].iloc[idx_val]

            self.fold_data.append([idx_trn, idx_val])

            self.estimator.fit(X_trn, y_trn, eval_set=[(X_trn, y_trn), (X_val, y_val)], **self.FIT_PARAMS)
            self.model_list.append(self.estimator)

            self.trn_score.append(self.metric(y_trn, self.estimator.predict_proba(X_trn)[:, 1]))
            self.val_predict[idx_val] = self.estimator.predict_proba(X_val)[:, 1]
            self.val_score.append(self.metric(y_val, self.val_predict[idx_val]))

            elapsed_time = timeit.default_timer() - start_time

            score = self.metric_name

            if is_silent == False:
                print(
                    f'\n---->Fold {fold + 1} ({elapsed_time:.2f}s) || trn {score}: {self.estimator.best_score_["trn"][score]:.4f} || val {score}: {self.estimator.best_score_["val"][score]:.4f}\n')
        self.trn_score_mean = np.mean(self.trn_score)
        self.val_score_mean = np.mean(self.val_score)
        if is_silent == False:
            print('-' * 69)
            print(f'Mean OOF trn {score}: {self.trn_score_mean:.4f} || val {score}: {self.val_score_mean:.4f}')
        self._is_fitted = True


    def predict_proba(self, test):
        if self._is_fitted == False:
            raise ValueError('Model not fitted, use ".fit" method.')
        self.test = test
        self.tst_predict = np.zeros(test.shape[0])
        for model in self.model_list:
            self.tst_predict += model.predict_proba(test[self.cols_cv])[:, 1] / self.cv.n_splits

        return self.tst_predict

    # TODO predict_proba
    def predict(self, test):
        pass


    def permutation_importance(self, PARAMS):
        assert self._is_fitted == True, '''Instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator'''

        df_fi = pd.DataFrame(index=self.cols_cv,
                             columns=[f'Fold {i}' for i in range(self.cv.n_splits)])
        for fold, (model, (idx_trn, idx_val)) in enumerate(zip(self.model_list, self.fold_data)):
            X_val = self.train[self.cols_cv].iloc[idx_val]
            y_val = self.train[self.TARGET].iloc[idx_val]
            result = permutation_importance(X_val, y_val, **PARAMS)

            df_fi.loc[:, f'Fold {fold}'] = result.importances_mean
        self.perm_fi = df_fi

        return df_fi.mean(axis=1)

class TrainTestSplit:
    def __init__(self, ):
        pass