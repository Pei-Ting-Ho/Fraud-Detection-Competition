"""
STEP 1: PREPROCESS RAW FILES AND GENERATE AGGREGATED FEATURES
"""

import numpy as np
import pandas as pd

from collections import Counter

from tqdm import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


def load_files():
    
    '''
    To read csv files from directory 
    '''

    print('To read files ...')
    
    TRAIN_raw = []
    df_TRAIN_1 = pd.read_csv('./Datasets/public_train_x_ccba_full_hashed.csv')
    df_TRAIN_2 = pd.read_csv('./Datasets/public_train_x_cdtx0001_full_hashed.csv')
    df_TRAIN_3 = pd.read_csv('./Datasets/public_train_x_custinfo_full_hashed.csv')
    df_TRAIN_4 = pd.read_csv('./Datasets/public_train_x_dp_full_hashed.csv')
    df_TRAIN_5 = pd.read_csv('./Datasets/public_train_x_remit1_full_hashed.csv')
    df_TRAIN_6 = pd.read_csv('./Datasets/public_x_alert_date.csv')
    df_TRAIN_7 = pd.read_csv('./Datasets/train_x_alert_date.csv')
    df_TRAIN_8 = pd.read_csv('./Datasets/train_y_answer_1.csv')
    df_TRAIN_9 = pd.read_csv('./Datasets/train_y_answer_2.csv')
    TRAIN_raw += [df_TRAIN_1, df_TRAIN_2, df_TRAIN_3, df_TRAIN_4, df_TRAIN_5, df_TRAIN_6, df_TRAIN_7, df_TRAIN_8, df_TRAIN_9]
    
    print(f'TRAIN: {len(set(df_TRAIN_3.alert_key).intersection(set(df_TRAIN_8.alert_key)))} + {len(set(df_TRAIN_3.alert_key).intersection(set(df_TRAIN_9.alert_key)))}')
    TRAIN_id = pd.concat([df_TRAIN_8, df_TRAIN_9], axis=0).alert_key
    
    TEST_raw = []
    df_TEST_1 = pd.read_csv('./Datasets/private_x_ccba_full_hashed.csv')
    df_TEST_2 = pd.read_csv('./Datasets/private_x_cdtx0001_full_hashed.csv')
    df_TEST_3 = pd.read_csv('./Datasets/private_x_custinfo_full_hashed.csv')
    df_TEST_4 = pd.read_csv('./Datasets/private_x_dp_full_hashed.csv')
    df_TEST_5 = pd.read_csv('./Datasets/private_x_remit1_full_hashed.csv')
    df_TEST_6 = pd.read_csv('./Datasets/private_x_alert_date.csv')
    TEST_raw += [df_TEST_1, df_TEST_2, df_TEST_3, df_TEST_4, df_TEST_5, df_TEST_6]

    print(f'TEST: {len(set(df_TEST_3.alert_key).intersection(set(df_TEST_6.alert_key)))}')
    TEST_id = df_TEST_6.alert_key

    return TRAIN_raw, TEST_raw, TRAIN_id, TEST_id


def CUSTINFO(TRAIN_raw, TEST_raw, TRAIN_id, TEST_id):
    
    '''
    To preprocess CUSTINFO subsets 
    '''

    df_TRAIN_1, df_TRAIN_2, df_TRAIN_3, df_TRAIN_4, df_TRAIN_5, df_TRAIN_6, df_TRAIN_7, df_TRAIN_8, df_TRAIN_9 = TRAIN_raw
    df_TEST_1, df_TEST_2, df_TEST_3, df_TEST_4, df_TEST_5, df_TEST_6 = TEST_raw
    
    y_TRAIN = pd.concat([df_TRAIN_8, df_TRAIN_9], axis=0)
    y_TRAIN_DATE = pd.concat([df_TRAIN_6, df_TRAIN_7], axis=0)
    df_TRAIN = df_TRAIN_3[df_TRAIN_3.alert_key.isin(TRAIN_id)].merge(y_TRAIN_DATE, how='inner', on='alert_key').merge(y_TRAIN, how='inner', on='alert_key')

    y_TEST = pd.DataFrame({'alert_key': df_TEST_6['alert_key'], 'sar_flag': [np.nan]*len(df_TEST_6)})
    y_TEST_DATE = df_TEST_6
    df_TEST = df_TEST_3[df_TEST_3.alert_key.isin(TEST_id)].merge(y_TEST_DATE, how='inner', on='alert_key').merge(y_TEST, how='inner', on='alert_key')

    df_MERGED = pd.concat([df_TRAIN, df_TEST], axis=0)
    ASSET_DIFF = []
    ASSET_NUNIQUE = []
    RISK_DIFF = []
    RISK_NUNIQUE = []

    for cust_id, alert_date in zip(df_MERGED.cust_id, df_MERGED.date):
        df_SUBSET = df_MERGED[(df_MERGED.cust_id == cust_id) & (df_MERGED.date <= alert_date)]
        ASSET_DIFF += [df_SUBSET['total_asset'].max() - df_SUBSET['total_asset'].min()]
        ASSET_NUNIQUE += [df_SUBSET['total_asset'].nunique()]
        RISK_DIFF += [df_SUBSET['risk_rank'].max() - df_SUBSET['risk_rank'].min()]
        RISK_NUNIQUE += [df_SUBSET['risk_rank'].nunique()]

    df_MERGED['DIFF(total_asset)'] = ASSET_DIFF
    df_MERGED['NUNIQUE(total_asset)'] = ASSET_NUNIQUE
    df_MERGED['DIFF(risk_rank)'] = RISK_DIFF
    df_MERGED['NUNIQUE(risk_rank)'] = RISK_NUNIQUE
    df_MERGED['RANK(alert_key)'] = df_MERGED.groupby(['cust_id'])['alert_key'].rank()
    
    df_TRAIN = df_MERGED[~df_MERGED.sar_flag.isnull()]
    df_TEST = df_MERGED[df_MERGED.sar_flag.isnull()].drop('sar_flag', axis=1)
    
    print('Preprocess CUSTINFO done ...')

    return df_TRAIN, df_TEST


def CCBA(TRAIN_raw, TEST_raw):
    
    '''
    To preprocess CCBA subsets
    '''

    df_TRAIN_1, df_TRAIN_2, df_TRAIN_3, df_TRAIN_4, df_TRAIN_5, df_TRAIN_6, df_TRAIN_7, df_TRAIN_8, df_TRAIN_9 = TRAIN_raw
    df_TEST_1, df_TEST_2, df_TEST_3, df_TEST_4, df_TEST_5, df_TEST_6 = TEST_raw
    
    CCBA_df = pd.concat([df_TRAIN_1, df_TEST_1], axis=0)
    CCBA_df.drop_duplicates(inplace=True)
    
    CCBA_df['exceed_cycam'] = CCBA_df['usgam'] > CCBA_df['cycam']
    CCBA_df['negative_usgam'] = CCBA_df['usgam'] < 0
    CCBA_df['loan'] = CCBA_df['clamt'] + CCBA_df['csamt'] + CCBA_df['cucah']
    CCBA_df['consumption'] = CCBA_df['inamt'] + CCBA_df['cucsm']
    CCBA_df = CCBA_df.drop(['clamt', 'csamt', 'cucah', 'inamt', 'cucsm'], axis=1)
    
    print('Preprocess CCBA done ...')
    
    return CCBA_df


def CDTX(TRAIN_raw, TEST_raw):
    
    '''
    To preprocess CDTX subsets
    '''

    df_TRAIN_1, df_TRAIN_2, df_TRAIN_3, df_TRAIN_4, df_TRAIN_5, df_TRAIN_6, df_TRAIN_7, df_TRAIN_8, df_TRAIN_9 = TRAIN_raw
    df_TEST_1, df_TEST_2, df_TEST_3, df_TEST_4, df_TEST_5, df_TEST_6 = TEST_raw
    
    CDTX_df = pd.concat([df_TRAIN_2, df_TEST_2], axis=0)
    CDTX_df.drop_duplicates(inplace=True)
    
    print('Preprocess CDTX done ...')
    
    return CDTX_df


def DP(TRAIN_raw, TEST_raw):
    
    '''
    To preprocess DP subsets
    '''

    df_TRAIN_1, df_TRAIN_2, df_TRAIN_3, df_TRAIN_4, df_TRAIN_5, df_TRAIN_6, df_TRAIN_7, df_TRAIN_8, df_TRAIN_9 = TRAIN_raw
    df_TEST_1, df_TEST_2, df_TEST_3, df_TEST_4, df_TEST_5, df_TEST_6 = TEST_raw
    
    DP_df = pd.concat([df_TRAIN_4, df_TEST_4], axis=0)
    DP_df.drop_duplicates(inplace=True)
    
    DP_df = pd.get_dummies(DP_df, columns=["debit_credit", "tx_type"])
    DP_df['tx_amt_ntd'] = DP_df['tx_amt'] * DP_df['exchg_rate']
    DP_df['cash_oc'] = (DP_df['info_asset_code'] == 12) & (DP_df['tx_type_1'] == 1)
    DP_df = DP_df.drop(['tx_amt', 'exchg_rate'], axis=1)
    
    print('Preprocess DP done ...')
    
    return DP_df


def REMIT(TRAIN_raw, TEST_raw):
    
    '''
    To preprocess REMIT subsets
    '''

    df_TRAIN_1, df_TRAIN_2, df_TRAIN_3, df_TRAIN_4, df_TRAIN_5, df_TRAIN_6, df_TRAIN_7, df_TRAIN_8, df_TRAIN_9 = TRAIN_raw
    df_TEST_1, df_TEST_2, df_TEST_3, df_TEST_4, df_TEST_5, df_TEST_6 = TEST_raw
    
    REMIT_df = pd.concat([df_TRAIN_5, df_TEST_5], axis=0)
    REMIT_df.drop_duplicates(inplace=True)
    
    REMIT_df = pd.get_dummies(REMIT_df, columns=["trans_no"])
    
    print('Preprocess REMIT done ...')
    
    return REMIT_df


def create_primitive_features(TRAIN_raw, TEST_raw, TRAIN_id, TEST_id):
    
    '''
    To create primitive features from five subsets (CUSTINFO, CCBA, CDTX, DP, REMIT)
    '''

    print('To create primitive features ...')
    
    df_TRAIN, df_TEST = CUSTINFO(TRAIN_raw, TEST_raw, TRAIN_id, TEST_id)
    
    CCBA_df = CCBA(TRAIN_raw, TEST_raw)
    
    CDTX_df = CDTX(TRAIN_raw, TEST_raw)
    
    DP_df = DP(TRAIN_raw, TEST_raw)
    
    REMIT_df = REMIT(TRAIN_raw, TEST_raw)
    
    print('Finished !')
    
    return df_TRAIN, df_TEST, CCBA_df, CDTX_df, DP_df, REMIT_df


def create_master_info_dictionary(df_TRAIN, df_TEST, CCBA_df, CDTX_df, DP_df, REMIT_df):
    
    '''
    To store customer historical records prior to further feature generation
    '''

    master_info = {}
    master_info_pred = {}
    
    pairs = [[master_info, df_TRAIN], [master_info_pred, df_TEST]]
    
    print('To store customer historical records, pls be patient ...')
    
    for pair in pairs:
        
        container = pair[0]
        df = pair[1]

        for row in df.itertuples():
            
            container[str(row.alert_key)] = {}
            container[str(row.alert_key)]['cid'] = row.cust_id
            container[str(row.alert_key)]['critical_date'] = row.date
            container[str(row.alert_key)]['records'] = []
    
        for key_id, key_info in tqdm(container.items()):

            cid = key_info['cid']
            critical_date = key_info['critical_date']

            customer_CCBA = CCBA_df[(CCBA_df.cust_id == cid) & (CCBA_df.byymm <= critical_date)]
            customer_CDTX = CDTX_df[(CDTX_df.cust_id == cid) & (CDTX_df.date <= critical_date)]
            customer_DP = DP_df[(DP_df.cust_id == cid) & (DP_df.tx_date <= critical_date)]
            customer_REMIT = REMIT_df[(REMIT_df.cust_id == cid) & (REMIT_df.trans_date <= critical_date)]

            container[key_id]['records'].extend([customer_CCBA, customer_CDTX, customer_DP, customer_REMIT])
    
    print(f'TRAIN KEYS: {len(master_info.keys())}')
    print(f'TEST KEYS: {len(master_info_pred.keys())}')

    return master_info, master_info_pred


def CCBA_DFS(customer_df, cid):
    
    '''
    To perform Deep Feature Synthesis for CCBA subsets
    '''

    CCBA_customer_df = pd.DataFrame({
        'cust_id': [cid], 
        'MIN(lupay)': [0], 'MAX(lupay)': [0], 'MEDIAN(lupay)': [0], 'MEAN(lupay)': [0], 'STD(lupay)': [0], 'ACF(lupay)': [0], 
        'MIN(cycam)': [0], 'MAX(cycam)': [0], 'MEDIAN(cycam)': [0], 'MEAN(cycam)': [0], 'STD(cycam)': [0], 'ACF(cycam)': [0],
        'MIN(usgam)': [0], 'MAX(usgam)': [0], 'MEDIAN(usgam)': [0], 'MEAN(usgam)': [0], 'STD(usgam)': [0], 'ACF(usgam)': [0],
        'MIN(loan)': [0], 'MAX(loan)': [0], 'MEDIAN(loan)': [0], 'MEAN(loan)': [0], 'STD(loan)': [0], 'ACF(loan)': [0],
        'MIN(consumption)': [0], 'MAX(consumption)': [0], 'MEDIAN(consumption)': [0], 'MEAN(consumption)': [0], 'STD(consumption)': [0], 'ACF(consumption)': [0], 
        'FREQ(exceed_cycam)': [0],
        'FREQ(negative_usgam)': [0]
    })
    
    if len(customer_df) > 0:
        
        col_func = {
            'lupay': ['min', 'max', 'median', 'mean', 'std', lambda x: x.autocorr()], 
            'cycam': ['min', 'max', 'median', 'mean', 'std', lambda x: x.autocorr()], 
            'usgam': ['min', 'max', 'median', 'mean', 'std', lambda x: x.autocorr()], 
            'exceed_cycam': ['sum', 'size'], 
            'negative_usgam': ['sum', 'size'],
            'loan': ['min', 'max', 'median', 'mean', 'std', lambda x: x.autocorr()],
            'consumption': ['min', 'max', 'median', 'mean', 'std', lambda x: x.autocorr()]
        }

        CCBA_grouped = customer_df.groupby('cust_id', as_index=False).agg(col_func)
        CCBA_grouped.columns = ["_".join(multi_cols) for multi_cols in CCBA_grouped.columns.values]
        
        CCBA_grouped['FREQ(exceed_cycam)'] = CCBA_grouped['exceed_cycam_sum'] / CCBA_grouped['exceed_cycam_size'] 
        CCBA_grouped['FREQ(negative_usgam)'] = CCBA_grouped['negative_usgam_sum'] / CCBA_grouped['negative_usgam_size']
        CCBA_grouped = CCBA_grouped.drop(['negative_usgam_sum', 'negative_usgam_size', 'exceed_cycam_sum', 'exceed_cycam_size'], axis=1)
        
        CCBA_customer_df = CCBA_grouped.rename(columns={
            'cust_id_': 'cust_id', 
            'lupay_min': 'MIN(lupay)', 'lupay_max': 'MAX(lupay)', 'lupay_median': 'MEDIAN(lupay)', 'lupay_mean': 'MEAN(lupay)', 'lupay_std': 'STD(lupay)', 'lupay_<lambda_0>': 'ACF(lupay)', 
            'cycam_min': 'MIN(cycam)', 'cycam_max': 'MAX(cycam)', 'cycam_median': 'MEDIAN(cycam)', 'cycam_mean': 'MEAN(cycam)', 'cycam_std': 'STD(cycam)', 'cycam_<lambda_0>': 'ACF(cycam)',
            'usgam_min': 'MIN(usgam)', 'usgam_max': 'MAX(usgam)', 'usgam_median': 'MEDIAN(usgam)', 'usgam_mean': 'MEAN(usgam)', 'usgam_std': 'STD(usgam)', 'usgam_<lambda_0>': 'ACF(usgam)',
            'loan_min': 'MIN(loan)', 'loan_max': 'MAX(loan)', 'loan_median': 'MEDIAN(loan)', 'loan_mean': 'MEAN(loan)', 'loan_std': 'STD(loan)', 'loan_<lambda_0>': 'ACF(loan)', 
            'consumption_min': 'MIN(consumption)', 'consumption_max': 'MAX(consumption)', 'consumption_median': 'MEDIAN(consumption)', 'consumption_mean': 'MEAN(consumption)', 'consumption_std': 'STD(consumption)', 'consumption_<lambda_0>': 'ACF(consumption)'
        })
    
    return CCBA_customer_df


def CDTX_DFS(customer_df, cid):
    
    '''
    To perform Deep Feature Synthesis for CDTX subsets
    '''

    CDTX_customer_df = pd.DataFrame({
        'cust_id': [cid], 
        'DISTINCTCOUNT(cur_type)': [0],  
        'MIN(amt)': [0], 'MAX(amt)': [0], 'MEDIAN(amt)': [0], 'MEAN(amt)': [0], 'STD(amt)': [0], 'ACF(amt)': [0],
        'COUNT(cdtx)': [0], 'DISTINCTCOUNT(country)': [0],
        'NUMDAYS(date)': [0], 
        'FREQ(roc)': [0], 
        'FREQ(ntd)': [0]
    })
                
    if len(customer_df) > 0:
        
        col_func = {
            'cur_type': ['nunique', (lambda x: sum(x == 47))],
            'date': ['max', 'min'],
            'amt': ['min', 'max', 'median', 'mean', 'std', lambda x: x.autocorr()], 
            'country': ['size', 'nunique', (lambda x: sum(x == 130))]
        }

        CDTX_grouped = customer_df.groupby(['cust_id'], as_index=False).agg(col_func)
        CDTX_grouped.columns = ["_".join(map(str, multi_cols)) for multi_cols in CDTX_grouped.columns.values]
        
        CDTX_grouped['NUMDAYS(date)'] = CDTX_grouped['date_max'] - CDTX_grouped['date_min']
        CDTX_grouped['FREQ(roc)'] = CDTX_grouped['country_<lambda_0>'] / CDTX_grouped['country_size'] 
        CDTX_grouped['FREQ(ntd)'] = CDTX_grouped['cur_type_<lambda_0>'] / CDTX_grouped['country_size']
        CDTX_grouped = CDTX_grouped.drop(['date_max', 'date_min', 'country_<lambda_0>', 'cur_type_<lambda_0>'], axis=1)
        
        CDTX_customer_df = CDTX_grouped.rename(columns={
            'cust_id_': 'cust_id',  
            'cur_type_nunique': 'DISTINCTCOUNT(cur_type)',  
            'amt_min': 'MIN(amt)', 'amt_max': 'MAX(amt)', 'amt_median': 'MEDIAN(amt)', 'amt_mean': 'MEAN(amt)', 'amt_std': 'STD(amt)', 'amt_<lambda_0>': 'ACF(amt)',  
            'country_size': 'COUNT(cdtx)', 'country_nunique': 'DISTINCTCOUNT(country)'
        })

    return CDTX_customer_df


def DP_DFS(customer_df, cid):
    
    '''
    To perform Deep Feature Synthesis for DP subsets
    '''

    DP_customer_df = pd.DataFrame({
        'cust_id': [cid], 
        'COUNT(CR)': [0], 'COUNT(DB)': [0],
        'COUNT(tx_type_1)': [0], 'COUNT(tx_type_2)': [0], 'COUNT(tx_type_3)': [0], 
        'MIN(tx_amt_ntd)': [0], 'MAX(tx_amt_ntd)': [0], 'MEDIAN(tx_amt_ntd)': [0], 'MEAN(tx_amt_ntd)': [0], 'STD(tx_amt_ntd)': [0], 'ACF(tx_amt_ntd)': [0],  
        'COUNT(dp)': [0],
        'NUMDAYS(tx_date)': [0],  
        'FREQ(cross_bank)': [0], 
        'FREQ(ATM)': [0], 
        'FREQ(cash_oc)': [0]
    })
    
    if len(customer_df) > 0:
    
        col_func = {
            'tx_date': ['min', 'max'],
            'debit_credit_CR': 'sum',
            'debit_credit_DB': 'sum',  
            'tx_type_1': 'sum', 
            'tx_type_2': 'sum', 
            'tx_type_3': 'sum',
            'tx_amt_ntd': ['min', 'max', 'median', 'mean', 'std', lambda x: x.autocorr()], 
            'cash_oc': 'sum', 
            'cross_bank': 'sum', 
            'ATM': 'sum', 
            'fiscTxId': ['size']
        }

        DP_grouped = customer_df.groupby(['cust_id'], as_index=False).agg(col_func)
        DP_grouped.columns = ["_".join(map(str, multi_cols)) for multi_cols in DP_grouped.columns.values]
        
        DP_grouped['NUMDAYS(tx_date)'] = DP_grouped['tx_date_max'] - DP_grouped['tx_date_min']
        DP_grouped['FREQ(cross_bank)'] = DP_grouped['cross_bank_sum'] / DP_grouped['fiscTxId_size'] 
        DP_grouped['FREQ(ATM)'] = DP_grouped['ATM_sum'] / DP_grouped['fiscTxId_size']
        DP_grouped['FREQ(cash_oc)'] = DP_grouped['cash_oc_sum'] / DP_grouped['fiscTxId_size']
        DP_grouped = DP_grouped.drop(['tx_date_max', 'tx_date_min', 'cross_bank_sum', 'ATM_sum', 'cash_oc_sum'], axis=1)
        
        DP_customer_df = DP_grouped.rename(columns={
            'cust_id_': 'cust_id',
            'debit_credit_CR_sum': 'COUNT(CR)', 'debit_credit_DB_sum': 'COUNT(DB)', 
            'tx_type_1_sum': 'COUNT(tx_type_1)', 'tx_type_2_sum': 'COUNT(tx_type_2)', 'tx_type_3_sum': 'COUNT(tx_type_3)',
            'tx_amt_ntd_min': 'MIN(tx_amt_ntd)', 'tx_amt_ntd_max': 'MAX(tx_amt_ntd)', 'tx_amt_ntd_median': 'MEDIAN(tx_amt_ntd)', 'tx_amt_ntd_mean': 'MEAN(tx_amt_ntd)', 'tx_amt_ntd_std': 'STD(tx_amt_ntd)', 'tx_amt_ntd_<lambda_0>': 'ACF(tx_amt_ntd)',  
            'fiscTxId_size': 'COUNT(dp)'
        })
        
    return DP_customer_df


def REMIT_DFS(customer_df, cid):
    
    '''
    To perform Deep Feature Synthesis for REMIT subsets
    '''

    REMIT_customer_df = pd.DataFrame({
        'cust_id': [cid], 
        'COUNT(trans_no_0)': [0], 'COUNT(trans_no_1)': [0], 'COUNT(trans_no_2)': [0], 'COUNT(trans_no_3)': [0], 'COUNT(trans_no_4)': [0], 
        'MIN(trade_amount_usd)': [0], 'MAX(trade_amount_usd)': [0], 'MEDIAN(trade_amount_usd)': [0], 'MEAN(trade_amount_usd)': [0], 'STD(trade_amount_usd)': [0], 'ACF(trade_amount_usd)': [0],   
        'COUNT(remit)': [0], 
        'NUMDAYS(trans_date)': [0]
    })
    
    if len(customer_df) > 0:
        
        col_func = {
            'trans_no_0': 'sum', 
            'trans_no_1': 'sum', 
            'trans_no_2': 'sum', 
            'trans_no_3': 'sum', 
            'trans_no_4': 'sum', 
            'trade_amount_usd': ['min', 'max', 'median', 'mean', 'std', 'size', lambda x: x.autocorr()], 
            'trans_date': ['min', 'max']
        }

        REMIT_grouped = customer_df.groupby(['cust_id'], as_index=False).agg(col_func)
        REMIT_grouped.columns = ['_'.join(multi_cols) for multi_cols in REMIT_grouped.columns.values]
        
        REMIT_grouped['NUMDAYS(trans_date)'] = REMIT_grouped['trans_date_max'] - REMIT_grouped['trans_date_min']
        REMIT_grouped = REMIT_grouped.drop(['trans_date_max', 'trans_date_min'], axis=1)
        
        REMIT_customer_df = REMIT_grouped.rename(columns={
            'cust_id_': 'cust_id',  
            'trans_no_0_sum': 'COUNT(trans_no_0)', 'trans_no_1_sum': 'COUNT(trans_no_1)', 'trans_no_2_sum': 'COUNT(trans_no_2)', 'trans_no_3_sum': 'COUNT(trans_no_3)', 'trans_no_4_sum': 'COUNT(trans_no_4)', 
            'trade_amount_usd_min': 'MIN(trade_amount_usd)', 'trade_amount_usd_max': 'MAX(trade_amount_usd)', 'trade_amount_usd_median': 'MEDIAN(trade_amount_usd)', 'trade_amount_usd_mean': 'MEAN(trade_amount_usd)', 'trade_amount_usd_std': 'STD(trade_amount_usd)', 
            'trade_amount_usd_size': 'COUNT(remit)', 'trade_amount_usd_<lambda_0>': 'ACF(trade_amount_usd)'
        })
        
    return REMIT_customer_df


def create_aggregated_features(df_TRAIN, df_TEST, CCBA_df, CDTX_df, DP_df, REMIT_df):
    
    '''
    To create aggregated features from four subsets (CCBA, CDTX, DP, REMIT)
    '''

    master_info, master_info_pred = create_master_info_dictionary(df_TRAIN, df_TEST, CCBA_df, CDTX_df, DP_df, REMIT_df)
    
    print('To create customer aggregated features, pls be patient ...')
    
    for container in [master_info, master_info_pred]:
        
        for key_id, key_info in tqdm(container.items()):
    
            customer_id = key_info['cid']

            key_info['master_records'] = []

            customer_CCBA = key_info['records'][0]
            customer_CDTX = key_info['records'][1]
            customer_DP = key_info['records'][2]
            customer_REMIT = key_info['records'][3]

            CCBA_df = CCBA_DFS(customer_CCBA, customer_id)
            CDTX_df = CDTX_DFS(customer_CDTX, customer_id)
            DP_df = DP_DFS(customer_DP, customer_id)
            REMIT_df = REMIT_DFS(customer_REMIT, customer_id)

            key_info['master_records'].extend([CCBA_df, CDTX_df, DP_df, REMIT_df])
    
    print('Finished !')
    
    return master_info, master_info_pred


def compile_final_results(df_TRAIN, df_TEST, master_info, master_info_pred):
    
    '''
    To compile final results and create preprocessed dataframes  
    '''

    pairs = [[master_info, df_TRAIN, pd.DataFrame()], [master_info_pred, df_TEST, pd.DataFrame()]]
    
    print('To compile final results ...')
    
    for pair in pairs:
        
        container = pair[0]
        df = pair[1]
        df_fn = pair[2]
        
        for key_id, key_info in tqdm(container.items()):

            CCBA_df = container[key_id]['master_records'][0]
            CDTX_df = container[key_id]['master_records'][1]
            DP_df = container[key_id]['master_records'][2]
            REMIT_df = container[key_id]['master_records'][3]

            customer_prior_info = CCBA_df.merge(CDTX_df).merge(DP_df).merge(REMIT_df)

            df_key = df[df.alert_key.astype(str) == key_id]
            df_key = df_key.merge(customer_prior_info, on='cust_id')
            df_fn = pd.concat([df_fn, df_key], axis=0, ignore_index=True)
        
        pair[2] = df_fn
    
    df_TRAIN_ENRICHED = pairs[0][2]
    df_TEST_ENRICHED = pairs[1][2]
    
    print(f'TRAIN DIM: {df_TRAIN_ENRICHED.shape}')
    print(f'TEST DIM: {df_TEST_ENRICHED.shape}')
    
    df_TRAIN_ENRICHED.to_csv('./Preprocess/df_TRAIN_preprocessed.csv', index=False)
    df_TEST_ENRICHED.to_csv('./Preprocess/df_TEST_preprocessed.csv', index=False)


def main():
    
    '''
    To run the whole workflow 
    '''

    # STEP 1
    TRAIN_raw, TEST_raw, TRAIN_id, TEST_id = load_files()
    
    # STEP 2
    df_TRAIN, df_TEST, CCBA_df, CDTX_df, DP_df, REMIT_df = create_primitive_features(TRAIN_raw, TEST_raw, TRAIN_id, TEST_id)
    
    # STEP 3
    master_info, master_info_pred = create_aggregated_features(df_TRAIN, df_TEST, CCBA_df, CDTX_df, DP_df, REMIT_df)
    
    # STEP 4
    compile_final_results(df_TRAIN, df_TEST, master_info, master_info_pred)


if __name__ == "__main__":
    main()