import numpy as np
import pandas as pd
import os
import gc
import timeit
import pickle, yaml, json
import kaggle as kgl
from datetime import datetime

from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin


def _estimator_name(estimator):
    info = str(estimator.__class__).lower()
    if 'lightgbm' in info:
        return 'lgb'
    elif 'logistic' in info:
        return 'lr'
    else:
        pass


class CrossValidation(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator, cv, metric='auc'):
        metric_dict = {'auc': roc_auc_score,
                       'logloss': log_loss}
        self.cv = cv
        self.estimator = estimator
        self.model_name = _estimator_name(estimator)
        self.metric_name = metric
        self.metric = metric_dict[metric]
        self._is_fitted = False
        self._is_predicted = False

    def fit(self, train, cols_cv, TARGET, FIT_PARAMS, cols_encode=None, groups=None, is_silent=False):
        self.train = train
        self.cols_cv = cols_cv
        self.TARGET = TARGET
        self.is_silent = is_silent
        if self.is_silent:
            FIT_PARAMS['verbose'] = 0
        self.FIT_PARAMS = FIT_PARAMS
        self.groups = groups
        self.cols_encode = cols_encode

        self.splitter = self.cv.split(self.train[self.cols_cv], self.train[self.TARGET], groups=self.groups)

        self.model_list, self.trn_score, self.val_score = [], [], []
        self.val_predict = np.zeros(self.train.shape[0])
        self.fold_data = []

        if not is_silent:
            print(f'Number of columns = {len(cols_cv)}\n')

        if cols_encode is not None:
            self.dict_ce_global_mapping = {}
            self.grlobal_y = np.mean(self.train[self.TARGET])

        for fold, (idx_trn, idx_val) in enumerate(self.splitter):
            start_time = timeit.default_timer()

            X_trn, y_trn = self.train[self.cols_cv].iloc[idx_trn], self.train[self.TARGET].iloc[idx_trn]
            X_val, y_val = self.train[self.cols_cv].iloc[idx_val], self.train[self.TARGET].iloc[idx_val]

            self.fold_data.append([idx_trn, idx_val])

            if self.model_name == 'lgb':
                self.estimator.fit(X_trn, y_trn, eval_set=[(X_trn, y_trn), (X_val, y_val)], **self.FIT_PARAMS)
            elif self.model_name == 'lr':
                self.estimator.fit(X_trn, y_trn, **self.FIT_PARAMS)
            else:
                raise ValueError('Model not implemented.')
            self.model_list.append(self.estimator)

            trn_curr_score = self.metric(y_trn, self.estimator.predict_proba(X_trn, num_iteration=self.estimator.best_iteration_)[:, 1])
            self.trn_score.append(trn_curr_score)

            self.val_predict[idx_val] = self.estimator.predict_proba(X_val, num_iteration=self.estimator.best_iteration_)[:, 1]
            val_curr_score = self.metric(y_val, self.val_predict[idx_val])
            self.val_score.append(val_curr_score)

            elapsed_time = timeit.default_timer() - start_time

            del X_trn, X_val, y_trn, y_val
            gc.collect()

            metric = self.metric_name

            # return self.estimator


            if not is_silent:
                if self.model_name == 'lgb':
                    print(
                        f'\n---->Fold {fold + 1} ({elapsed_time:.2f}s) || trn {metric}: {self.estimator.best_score_["trn"][metric]:.6f} || val {metric}: {self.estimator.best_score_["val"][metric]:.6f}\n')
                elif self.model_name == 'lr':
                    print(
                        f'\n---->Fold {fold + 1} ({elapsed_time:.2f}s) || trn {metric}: {trn_curr_score:.6f} || val {metric}: {val_curr_score:.6f}\n')
        self.trn_score_mean = np.mean(self.trn_score)
        self.val_score_mean = np.mean(self.val_score)
        if not is_silent:
            print('-' * 69)
            print(
                f'Mean OOF trn {self.metric_name}: {self.trn_score_mean:.6f} || val {self.metric_name}: {self.val_score_mean:.6f}')

        self._is_fitted = True

    def predict_proba(self, test):
        if not self._is_fitted:
            raise ValueError('Model not fitted, use ".fit" method.')
        self.test = test
        self.tst_predict = np.zeros(test.shape[0])
        
        for model in self.model_list:
            self.tst_predict += model.predict_proba(test[self.cols_cv], num_iteration=model.best_iteration_)[:, 1] / self.cv.n_splits

        self._is_predicted = True

        return self.tst_predict

    # TODO predict_proba
    def __predict(self, test):
        pass

    def permutation_importance(self, PARAMS):
        assert self._is_fitted is True, '''Instance is not fitted yet. Call 'fit' with appropriate arguments before
        using this estimator'''

        df_fi = pd.DataFrame(index=self.cols_cv,
                             columns=[f'Fold_{i}' for i in range(self.cv.n_splits)])
        for fold, (model, (idx_trn, idx_val)) in enumerate(
                tqdm(zip(self.model_list, self.fold_data), total=self.cv.n_splits)):
            X_val = self.train[self.cols_cv].iloc[idx_val]
            y_val = self.train[self.TARGET].iloc[idx_val]
            result = permutation_importance(model, X_val, y_val, **PARAMS)

            df_fi.loc[:, f'Fold_{fold}'] = result.importances_mean
        self.perm_fi = df_fi

        return df_fi.mean(axis=1)

    def tree_importance(self, importance_type='both'):
        assert self._is_fitted is True, '''Instance is not fitted yet. Call 'fit' with appropriate arguments before
        using this estimator'''
        assert importance_type in ['both', 'gain', 'split'], f'''Importance type {importance_type} not implemented.'''

        df_fi = pd.DataFrame(np.zeros((len(self.cols_cv), len(['gain', 'split']))),
                             index=self.cols_cv,
                             columns=['gain', 'split'])

        for fold, model in enumerate(self.model_list):
            df_fi['gain'] += model.booster_.feature_importance('gain') / self.cv.n_splits
            df_fi['split'] += model.booster_.feature_importance('split') / self.cv.n_splits
        self.tree_fi = df_fi

        if importance_type == 'both':
            return df_fi
        else:
            df_fi[importance_type]

    # TODO
    def __shap_importance(self, PARAMS):
        pass

    def submit(self, cmpt_name, sub_dir='submissions', sample_sub='sample_submission.csv', prefix='sub', text_msg=None,
               send=True):
        assert self._is_predicted is True, 'There is no prediction to submit.'

        # self.submit_info = None
        text_msg = '' if text_msg is None else text_msg

        df_sub = pd.read_csv(f'{sub_dir}/{sample_sub}')
        df_sub[self.TARGET] = self.tst_predict

        cur_time = datetime.strftime(datetime.now(), '%Y_%m_%d-%H_%M_%S')
        self.sub_name = cur_time
        sub_path = f'{sub_dir}/{prefix}_{cur_time}'.replace('-', '$')
        file_name = f'{prefix}_{cur_time}.csv'

        os.mkdir(f'{sub_path}')

        df_sub.to_csv(f'{sub_path}/{file_name}', index=False)

        self._save(sub_path)

        print(f'Saved to {file_name}')

        if send:
            kgl.api.competition_submit(f'{sub_path}/{file_name}', message=text_msg, competition=cmpt_name)

    def _save(self, path, params=True, models=True, indexes=True, features=True, predicts=True):

        # save parameters
        if params:
            params_dir = f'{path}/parameters'
            os.mkdir(params_dir)
            with open(f'{params_dir}/CV_PARAMS.yml', 'a') as f:
                yaml.dump(self.cv.__dict__, f)

            with open(f'{params_dir}/MODEL_PARAMS.yml', 'a') as f:
                yaml.dump(self.estimator.get_params(), f)

            with open(f'{params_dir}/FIT_PARAMS.yml', 'a') as f:
                yaml.dump(self.FIT_PARAMS, f)

            with open(f'{path}/SUB_RESULTS.txt', 'w') as f:
                f.write(f'Mean {self.metric_name.upper()}:\n')
                f.write(f'  trn: {self.trn_score_mean:.4f} || val: {self.val_score_mean:.4f}\n')
                f.write('\n')
                f.write('FEATURES:\n')
                f.write(f'Number of features = {len(self.cols_cv)}')
                f.write('\n')
                f.write("', '".join(self.cols_cv))
                f.write('\n')
                f.write('CV_PARAMS:\n')
                for k, v in self.cv.__dict__.items():
                    f.write(f'  {k}: {v}\n')
                f.write('\n')
                f.write('MODEL_PARAMS:\n')
                for k, v in self.estimator.get_params().items():
                    f.write(f'  {k}: {v}\n')
                f.write('\n')
                f.write('FIT_PARAMS:\n')
                for k, v in self.FIT_PARAMS.items():
                    f.write(f'  {k}: {v}\n')

        # save fitted models
        if models:
            models_dir = f'{path}/models'
            os.mkdir(models_dir)
            for fold, model in enumerate(self.model_list):
                with open(f'{models_dir}/model_{fold}.pkl', 'wb') as f:
                    pickle.dump(model, f)

        # save model features
        if features:
            with open(f'{path}/columns.txt', 'w') as f:
                f.write('$$'.join(self.cols_cv))

        # save fold indexes
        if indexes:
            fold_indexes_dir = f'{path}/fold_indexes'
            os.mkdir(fold_indexes_dir)
            for fold, (idx_trn, idx_val) in enumerate(self.fold_data):
                with open(f'{fold_indexes_dir}/fold{fold}_idx_trn.pkl', 'wb') as f:
                    pickle.dump(idx_trn, f)

                with open(f'{fold_indexes_dir}/fold{fold}_idx_val.pkl', 'wb') as f:
                    pickle.dump(idx_val, f)

        # save predicts
        if predicts:
            pd.Series(self.val_predict, name='predict').to_csv(f'{path}/val_predict.csv', index=False)
            pd.Series(self.tst_predict, name='predict').to_csv(f'{path}/tst_predict.csv', index=False)

    # TODO
    def __name(self):
        pass


class TrainTestSplit:
    def __init__(self, ):
        pass
