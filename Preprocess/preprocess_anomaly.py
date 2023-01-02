"""
STEP 2: GENERATE CATEGORICAL ENCODING AND ANOMALY SCORE
"""

import numpy as np
import pandas as pd

from category_encoders import *
from sklearn.ensemble import IsolationForest

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


def load_files():
    
    '''
    To read preprocessed files from directory  
    '''

    df_TRAIN_preprocessed = pd.read_csv("./Preprocess/df_TRAIN_preprocessed.csv")
    df_TEST_preprocessed = pd.read_csv("./Preprocess/df_TEST_preprocessed.csv")

    return df_TRAIN_preprocessed, df_TEST_preprocessed


def encode_categoricals(df_TRAIN_preprocessed, df_TEST_preprocessed):
    
    '''
    To transform categorical variables with target encoder  
    '''

    df = pd.concat([df_TRAIN_preprocessed, df_TEST_preprocessed], axis=0)
    
    customer_occupation_map = {}
    for cid, occupation in zip(df.cust_id, df.occupation_code):
        customer_occupation_map[cid] = occupation
    df_TRAIN_preprocessed['occupation_code'] = df_TRAIN_preprocessed['cust_id'].map(customer_occupation_map)
    
    df_TRAIN_preprocessed['weekday'] = df_TRAIN_preprocessed['date'] % 7
    df_TEST_preprocessed['weekday'] = df_TEST_preprocessed['date'] % 7
    
    X = df_TRAIN_preprocessed.drop('sar_flag', axis=1)
    y = df_TRAIN_preprocessed['sar_flag']

    encoder = TargetEncoder(cols=['occupation_code', 'weekday'], smoothing=10).fit(X, y)
    X_encoded = encoder.transform(X)
    df_TRAIN_preprocessed = pd.concat([X_encoded, y], axis=1)
    df_TEST_preprocessed = encoder.transform(df_TEST_preprocessed)
    
    df_TRAIN_preprocessed.fillna(0, inplace=True)
    df_TEST_preprocessed.fillna(0, inplace=True)

    return df_TRAIN_preprocessed, df_TEST_preprocessed


def generate_anomaly_score(df_TRAIN_preprocessed, df_TEST_preprocessed):
    
    '''
    To apply unsupervised anomaly detection method
    '''

    print(f'contamination ratio = {sum(df_TRAIN_preprocessed.sar_flag) / len(df_TRAIN_preprocessed.sar_flag)}')
    
    X_TRAIN = df_TRAIN_preprocessed.drop(['alert_key', 'date', 'cust_id', 'sar_flag'], axis=1)
    X_TEST = df_TEST_preprocessed.drop(['alert_key', 'date', 'cust_id'], axis=1)

    clf = IsolationForest(n_estimators=500, max_samples=1.0, max_features=1.0, contamination=0.01, bootstrap=True, random_state=0).fit(X_TRAIN)

    TRAIN_pred_score = [*map(lambda x: -x, clf.score_samples(X_TRAIN))]
    df_TRAIN_preprocessed_anomaly = pd.concat([df_TRAIN_preprocessed, pd.Series(TRAIN_pred_score, name='anomaly_score')], axis=1)

    TEST_pred_score = [*map(lambda x: -x, clf.score_samples(X_TEST))]
    df_TEST_preprocessed_anomaly = pd.concat([df_TEST_preprocessed, pd.Series(TEST_pred_score, name='anomaly_score')], axis=1)
    
    print(f'TRAIN DIM: {df_TRAIN_preprocessed_anomaly.shape}')
    print(f'TEST DIM: {df_TEST_preprocessed_anomaly.shape}')

    df_TRAIN_preprocessed_anomaly.to_csv('./Preprocess/df_TRAIN_preprocessed+anomaly.csv', index=0)
    df_TEST_preprocessed_anomaly.to_csv('./Preprocess/df_TEST_preprocessed+anomaly.csv', index=0)


def main():
    
    '''
    To run the whole workflow 
    '''

    # STEP 1
    print('To read preprocessed files ...')
    df_TRAIN_preprocessed, df_TEST_preprocessed = load_files()
    
    # STEP 2
    print('To encode categorical variables ...')
    df_TRAIN_preprocessed_encoded, df_TEST_preprocessed_encoded = encode_categoricals(df_TRAIN_preprocessed, df_TEST_preprocessed)
    
    # STEP 3
    print('To apply unsupervised method ...')
    generate_anomaly_score(df_TRAIN_preprocessed_encoded, df_TEST_preprocessed_encoded)


if __name__ == "__main__":
    main()