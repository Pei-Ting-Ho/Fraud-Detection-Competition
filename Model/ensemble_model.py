"""
STEP 3: CREATE ENSEMBLE MODEL AND MAKE FINAL PREDICTIONS
"""

import numpy as np
import pandas as pd

from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from mlxtend.classifier import LogisticRegression, StackingClassifier

from sklearn.feature_selection import RFECV
from mlxtend.feature_selection import ColumnSelector

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


def load_files():
    
    '''
    To read preprocessed files from directory  
    '''

    df_TRAIN_preprocessed = pd.read_csv("./Preprocess/df_TRAIN_preprocessed+anomaly.csv")
    df_TEST_preprocessed = pd.read_csv("./Preprocess/df_TEST_preprocessed+anomaly.csv")
    
    return df_TRAIN_preprocessed, df_TEST_preprocessed


def prepare_df(df_TRAIN_preprocessed, df_TEST_preprocessed):
    
    '''
    To prepare TRAIN & TEST dataframes   
    '''

    df_final_record = pd.concat([df_TRAIN_preprocessed, df_TEST_preprocessed], axis=0)
    df_all = df_final_record.copy()

    for customer in df_all['cust_id'].unique():
        drop_idx = df_final_record[df_final_record['cust_id'] == customer].sort_values('date', ascending=False)[1:].index
        df_final_record.drop(drop_idx, inplace=True)
    
    df_TRAIN = df_final_record[~df_final_record['sar_flag'].isnull()].reset_index(drop=1)
    df_TEST  = df_final_record[df_final_record['sar_flag'].isnull()].reset_index(drop=1)

    df_TRAIN_X = df_TRAIN.drop(['alert_key', 'cust_id', 'date', 'sar_flag'], axis=1)
    df_TRAIN_y = df_TRAIN['sar_flag'].astype('bool')

    df_TEST_X = df_TEST.drop(['alert_key', 'cust_id', 'date', 'sar_flag'], axis=1)
    df_TEST_y = df_TEST['sar_flag'].astype('bool')
    
    return df_TRAIN, df_TEST, df_TRAIN_X, df_TRAIN_y, df_TEST_X


def eval_metric(y, y_pred):
    
    '''
    To create custom evaluation metric 
    '''

    total_n = sum(y)
    rank_index = sorted(range(len(y_pred)), key=lambda k: -y_pred[k])
    cur = 0
    for i_idx, r_idx in enumerate(rank_index):
        cur += y[r_idx]
        if cur == total_n-1:
            return (total_n-1) / (i_idx+1)


param_1 = {
    'learning_rate': 0.05, 
    'n_estimators': 500, 
    'grow_policy': 'lossguide', 
    'max_depth': 8,
    'min_child_weight': 15,
    'subsample': 0.85,
    'colsample_bytree': 0.8, 
    'reg_alpha': 10, 
    'reg_lambda': 10, 
    'objective': 'binary:logistic',
    'scale_pos_weight': 10,  
    'seed': 12345
} 
model_1 = XGBClassifier(**param_1)


param_2 = {
    'learning_rate': 0.01, 
    'n_estimators': 500, 
    'grow_policy': 'Lossguide', 
    'max_depth': 8,
    'subsample': 0.85,
    'loss_function': 'Logloss', 
    'scale_pos_weight': 10, 
    'random_seed': 12345, 
    'verbose': False
} 
model_2 = CatBoostClassifier(**param_2)


param_3 = {
    'boosting_type': 'goss', 
    'learning_rate': 0.05, 
    'n_estimators': 500, 
    'max_depth': 8,
    'subsample': 1.0,
    'colsample_bytree': 0.8,
    'reg_alpha': 3, 
    'reg_lambda': 3, 
    'objective': 'binary', 
    'scale_pos_weight': 10,  
    'random_state': 12345
} 
model_3 = LGBMClassifier(**param_3)


def choose_model_features(df_TRAIN_X, df_TRAIN_y, model_candidates, display_common_fs=False):
    
    '''
    To apply RFE method to help select the desired feature sets for each model
    '''

    model_features = {}
    
    models = model_candidates if isinstance(model_candidates, list) else [model_candidates]
    for model in models:

        print(type(model).__name__)

        model_features[type(model).__name__] = []

        selector = RFECV(estimator=model, step=3, min_features_to_select=30, cv=3, scoring=make_scorer(score_func=eval_metric, needs_proba=True), verbose=False)
        selector = selector.fit(df_TRAIN_X, df_TRAIN_y)

        print(f'score: {selector.cv_results_["mean_test_score"]}')
        print(f'n: {selector.n_features_}')
        print('-' * 50)

        model_features[type(model).__name__] += list(selector.feature_names_in_[selector.support_])
    
    if display_common_fs:
        features_1, features_2, features_3 = model_features.values()
        common_features = set(features_1).intersection(set(features_2)).intersection(set(features_3))
        print(common_features)
    
    return model_features


def make_model_predictions(df_TRAIN_X, df_TRAIN_y, df_TEST_X, model_features, model_candidates):
    
    '''
    To apply K-Fold-CV and Model-Calibration to evaluate model performance and make model prediction
    '''

    K_FOLD = StratifiedKFold(n_splits=10, random_state=12345, shuffle=True)

    fn_preds = []
    
    models = model_candidates if isinstance(model_candidates, list) else [model_candidates]
    for model in models:

        print(type(model).__name__)

        cv_scores = []

        df_MODEL_TRAIN_X = df_TRAIN_X[model_features[type(model).__name__]]   
        df_MODEL_TEST_X = df_TEST_X[model_features[type(model).__name__]]   

        for TRAIN_idx, VALID_idx in K_FOLD.split(df_MODEL_TRAIN_X, df_TRAIN_y):

            X_TRAIN, X_VALID = df_MODEL_TRAIN_X.iloc[TRAIN_idx], df_MODEL_TRAIN_X.iloc[VALID_idx]
            y_TRAIN, y_VALID = df_TRAIN_y.iloc[TRAIN_idx], df_TRAIN_y.iloc[VALID_idx]

            y_TRAIN = np.array(y_TRAIN, dtype=int)
            y_VALID = np.array(y_VALID, dtype=int)

            # FIT > CALIBRATE
            calibrated_clf = CalibratedClassifierCV(base_estimator=model, method="isotonic", cv=12)
            calibrated_clf.fit(X_TRAIN, y_TRAIN)

            # VALIDATE
            y_pred = calibrated_clf.predict_proba(X_VALID)[:, 1]
            y = np.array(y_VALID)

            score = eval_metric(y, y_pred)
            cv_scores.append(score)
            print("score:", score)

        calibrated_clf_fn = CalibratedClassifierCV(base_estimator=model, method="isotonic", cv=12)
        calibrated_clf_fn.fit(df_MODEL_TRAIN_X, np.array(df_TRAIN_y, dtype=int))
        preds = calibrated_clf_fn.predict_proba(df_MODEL_TEST_X)[:, 1]
        fn_preds.append(preds)

        print('-' * 50)
        print("cv_mean:", np.mean(cv_scores))
        print("cv_std:", np.std(cv_scores))
        print('\n')
    
    return fn_preds


def genertae_submission_file(fn_preds, private_keys):
    
    '''
    To generate submission file from ensemble model predictions 
    '''

    predictions = np.mean(fn_preds, axis=0)

    private_keys['sar_flag'] = predictions

    submission_file = pd.read_csv('./Submission/submission_example.csv')
    answer = pd.read_csv('./Datasets/train_y_answer_2.csv')

    pred_answer = {key: pred_y for key, pred_y in zip(private_keys['alert_key'], private_keys['sar_flag'])}
    real_answer = {key: real_y for key, real_y in zip(answer['alert_key'], answer['sar_flag'])}
    ansMap = {**pred_answer, **real_answer}
    submission_file['probability'] = submission_file['alert_key'].map(ansMap).fillna(1).astype('float32')
    
    df_TRAIN_CUST = pd.read_csv('./Datasets/public_train_x_custinfo_full_hashed.csv')
    df_TEST_CUST = pd.read_csv('./Datasets/private_x_custinfo_full_hashed.csv')
    CUST_info = pd.concat([df_TRAIN_CUST, df_TEST_CUST], axis=0)

    df_TRAIN_DATE = pd.read_csv('./Datasets/public_x_alert_date.csv')
    df_TEST_DATE = pd.read_csv('./Datasets/private_x_alert_date.csv')
    DATE_info = pd.concat([df_TRAIN_DATE, df_TEST_DATE], axis=0)
    
    fn_preprocess = submission_file.merge(CUST_info)[['alert_key', 'probability', 'cust_id']].merge(DATE_info)
    fn_map = {key: prob for key, prob in zip(fn_preprocess['alert_key'], fn_preprocess['probability'])}    
    submission_file['probability'] = submission_file['alert_key'].map(fn_map).astype('float32')
    submission_file.to_csv('./Submission/fn_submission.csv', index=0)


def main():
    
    '''
    To run the whole workflow 
    '''

    # STEP 1
    print('To read preprocessed files ...')
    df_TRAIN_preprocessed, df_TEST_preprocessed = load_files()
    
    # STEP 2
    df_TRAIN, df_TEST, df_TRAIN_X, df_TRAIN_y, df_TEST_X = prepare_df(df_TRAIN_preprocessed, df_TEST_preprocessed)
    
    # STEP 3
    print('To apply feature selection ...')
    model_features = choose_model_features(df_TRAIN_X, df_TRAIN_y, model_candidates=[model_1, model_2, model_3], display_common_fs=False)
    print('To make model prediction ...')
    pred_TEST = make_model_predictions(df_TRAIN_X, df_TRAIN_y, df_TEST_X, model_features, model_candidates=[model_1, model_2, model_3])
    
    # STEP 4
    print('Let\'s submit the file !')
    genertae_submission_file(fn_preds=pred_TEST, private_keys=df_TEST)


if __name__ == "__main__":
    main()