import numpy as np
import pandas as pd




class FeatureSelection:

    def __init__(self, cv_obj):
        self.cv_obj = cv_obj


    def permutation_importance(self, PARAMS):

        df_fi = pd.DataFrame(index=self.cv_obj.cols_cv, columns=[f'Fold {i}' for i in range(self.cv_obj.cv.n_splits)])
        for fold, (model, (idx_trn, idx_val)) in enumerate(zip(self.cv_obj.model_list, self.cv_obj.fold_data)):
            X_val = train[self.cv_obj.cols_cv].iloc[idx_val]
            y_val = train[self.cv_obj.TARGET].iloc[idx_val]
            result = permutation_importance(X_val, y_val, **PARAMS)

            df_fi.loc[:, f'Fold {fold}'] = result.importances_mean
        self.perm_fi = df_fi
        return df_fi.mean(axis=1)


    def shap_importance(self, PARAMS):
        pass


    def tree_featureimportance(self, PARAMS):
        pass


    # def