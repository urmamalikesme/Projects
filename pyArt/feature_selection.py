import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def _get_metric(metric_name):
    if metric_name == 'auc':
        return roc_auc_score
    else:
        raise ValueError('Choose correct metric.')


def filter_correlated_features(df_corr: pd.DataFrame, threshold: float = 0.7) -> list:
    """
    :param threshold: a threshold for correlation level
    :param df_corr: correlation matrix with right order of features - descending value of compare metric
    :return: list of filtered columns
    """

    i = 0
    while i < df_corr.shape[0]:

        current_feature_name = df_corr.columns[i]
        current_feature = df_corr.iloc[:, i]

        mask_current_name = current_feature.index != current_feature_name
        mask_threshold = current_feature >= threshold

        correlated_columns = current_feature[mask_threshold & mask_current_name].index
        if len(correlated_columns) > 0:
            df_corr = df_corr.drop(correlated_columns, axis=0).drop(correlated_columns, axis=1)

        i += 1

    return df_corr.columns.tolist()


def single_feature_score(X, y, estimator, metric='auc'):
    """
    :param X:
    :param y:
    :param estimator:
    :param metric:
    :return:
    """

    metric_func= _get_metric(metric_name=metric)
    df_score = pd.DataFrame()

    X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

    for col in tqdm(X_trn.columns):
        score = metric_func(y_val, estimator.fit(X_trn[[col]], y_trn).predict_proba(X_val[[col]])[:, 1])
        df_score.loc[col, f'score {metric}'] = score

    return df_score.sort_values(by=f'score {metric}', ascending=False)


class FeatureSelection:

    def __init__(self, cv_obj):
        pass
