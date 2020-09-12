import numpy as np
import pandas as pd

import gc

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score

import lightgbm as lgb


# data_definitionのdata_tableを前提としている
class TableData:
    def __init__(self, data_table, accessary_columns, categorical_features=[], target_column='Target'):
        self.data_table = data_table
        self.accessary_columns = accessary_columns
        self.categorical_features = categorical_features
        self.target_column = target_column

        self.feature_columns = self.get_feature_columns()
        self.Data = self.data_table[self.feature_columns]
        self.Target = self.data_table[self.target_column]

    # data_tableからFeature Columnsだけを取得する
    def get_feature_columns(self):
        all_columns = self.data_table.columns
        feature_columns = [c for c in all_columns if c not in self.accessary_columns and c != self.target_column]
        return feature_columns


class TableDataLGBMAnalyzer:
    DEFAULT_PARAMS = {
        'objective': 'binary',
        'learning_rate': 0.01,
        'boosting': "gbdt",
        'bagging_freq': 1,
        'bagging_seed': 11,
        'metric': 'auc',
        'verbosity': -1
    }

    def __init__(self, tabledata):
        self.tabledata = tabledata  # TableDataインスタンス
        self.params = self.DEFAULT_PARAMS
        self.pbounds = {  # Bayesian Optimizationの探索範囲
            'max_depth': (4, 10),
            'num_leaves': (5, 130),
            'min_data_in_leaf': (10, 150),
            'feature_fraction': (0.7, 1.0),
            'bagging_fraction': (0.7, 1.0),
            'lambda_l1': (0, 6)
        }
        self.enable_pos_weight = False


    # param_dict: max_depth, num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction, lambda_l1
    def set_params(self, param_dict):
        # BOでfloatになるため、整数値を担保する必要がある
        for k in ['num_leaves', 'min_data_in_leaf', 'max_depth']:
            if k in param_dict:
                param_dict.update({k: int(param_dict[k])})
        self.params.update(param_dict)


    def set_pos_weight(self, train_Data, train_Target):
        T = len(train_Data)
        P = np.sum(train_Target)
        self.params.update({'scale_pos_weight': T / P - 1})


    def reset_params(self):
        self.params = self.DEFAULT_PARAMS


    def reset_pos_weight(self):
        self.params['scale_pos_weight'] = 1.0


    def basic_predict(self, train_Data, train_Target, val_Data, val_Target):
        # 誤差関数の重みをつける場合はここでつける
        if self.enable_pos_weight:
            self.set_pos_weight(train_Data, train_Target)
        else:
            self.reset_pos_weight()

        trn_data = lgb.Dataset(
            train_Data[self.tabledata.feature_columns],
            label=train_Target,
            categorical_feature=self.tabledata.categorical_features
        )

        val_data = lgb.Dataset(
            val_Data[self.tabledata.feature_columns],
            label=val_Target,
            categorical_feature=self.tabledata.categorical_features
        )

        clf = lgb.train(
            self.params,
            trn_data,
            10000,
            valid_sets = [trn_data, val_data],
            verbose_eval=False,
            early_stopping_rounds=400
        )

        pred_y = clf.predict(
            val_Data[self.tabledata.feature_columns],
            num_iteration=clf.best_iteration
        )

        del clf
        gc.collect()

        return pred_y


    def get_predict_function(self):
        def predict_function(train_Data, train_Target, val_Data, val_Target):
            return self.basic_predict(train_Data, train_Target, val_Data, val_Target)
        return predict_function


    def get_BayesianOptimization_function(self, smote_rate=0):
        def BayesianOptimization_function(max_depth, num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction, lambda_l1):
            param_dict = {
                'max_depth': max_depth,
                'num_leaves': num_leaves,
                'min_data_in_leaf': min_data_in_leaf,
                'feature_fraction': feature_fraction,
                'bagging_fraction': bagging_fraction,
                'lambda_l1': lambda_l1
            }

            self.set_params(param_dict)

            # 5回の平均値を最大化する
            auc_scores = []
            for i in range(5):
                oof = cv_predict(self.tabledata, self.get_predict_function(), n_splits=5, smote_rate=smote_rate)
                auc_scores.append(roc_auc_score(self.tabledata.Target.values, oof))
            return np.mean(auc_scores)
        return BayesianOptimization_function


    def get_best_BayesianOptimization(self, smote_rate=0, init_points=2, n_iter=20):
        BayesianOptimization_function = self.get_BayesianOptimization_function(smote_rate=smote_rate)
        BO_object = BayesianOptimization(BayesianOptimization_function, self.pbounds, verbose=0)
        BO_object.maximize(init_points=init_points, n_iter=n_iter, acq='ei', xi=0.0)
        return BO_object


# smoteアルゴリズムによるオーバーサンプリング
def create_smoteDataset(Data, Target, smote_rate):
    if smote_rate == 0:
        return Data, Target
    elif 0 < smote_rate < 1:
        sm = SMOTE(
            sampling_strategy=smote_rate,
            k_neighbors=3,
            random_state=np.random.randint(0, 10000)
        )
        X_sm, y_sm = sm.fit_sample(Data.values, Target.values)
        smData = pd.DataFrame(X_sm, columns=Data.columns)
        smTarget = pd.Series(y_sm, name='Target')
        return smData, smTarget
    else:
        raise ValueError('smote_rate must be in (0, 1)')


# tabledata: TableData object
# predict_function(train_Data, train_Target, test_Data, test_Target) -> return pred_y
def cv_predict(tabledata, predict_function, n_splits=5, smote_rate=0):
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=np.random.randint(0, 10000))

    Data = tabledata.Data
    Target = tabledata.Target
    oof = np.zeros(len(Data))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(Data.values, Target.values)):
        train_Data, train_Target = Data.iloc[trn_idx], Target.iloc[trn_idx]
        val_Data, val_Target = Data.iloc[val_idx], Target.iloc[val_idx]
        train_Data, train_Target = create_smoteDataset(train_Data, train_Target, smote_rate)

        oof[val_idx] = predict_function(train_Data, train_Target, val_Data, val_Target)
    return oof
