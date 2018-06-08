import os
import datetime
import numpy as np
from scipy import sparse
from scipy.stats import mstats
import pandas as pd
from pandas.api.types import is_numeric_dtype
import re
import pickle
import shutil

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold


DIR_TRAIN_Y_GG = '../data/train.csv'
DIR_TRAIN_Y = '../data/meinian_round1_train_20180408.csv'
DIR_TEST_Y = '../data/meinian_round1_test_b_20180505.csv'
DIR_PART1 = '../data/meinian_round1_data_part1_20180408.txt'
DIR_PART2 = '../data/meinian_round1_data_part2_20180408.txt'
DIR_SUB = '../data/'
CACHE_Y_TRAIN = '../data/cache/Y_train.csv'
CACHE_X_RAW = '../data/cache/X_raw.csv'
CACHE_X_P1 = '../data/cache/X_p1.csv'
CACHE_X_PROCESSED = '../data/cache/X_all.csv'
CACHE_X_TO_MERGE_2 = '../data/cache/X_to_merge_2.csv'
CACHE_X_FINAL = '../data/cache/X_final.csv'


IF_VAL = True
IF_S2 = False
IF_AUG = False
NUM_FOLDS = 20
RANDOM_SEED = 2019

MODEL = 'lgb' # 'xgb'
PARAMS = {
    'xgb': {
        'objective': 'reg:linear',
        'seed': RANDOM_SEED,
        'nthread': 56,
        'eta': 0.01,
        'min_child_weight': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'silent': 1,
        'max_depth': 7,
        'num_rounds': 10000,
        'num_early_stop': 50
    },
    'lgb': {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'mse',
        'metric': 'l2',
        'num_leaves': 31,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 2,
        'verbose': 0,
        'bagging_seed': RANDOM_SEED,
        'num_rounds_lgb': 10000,
        'num_early_stop': 200
    }
}


def save_obj(obj, name):
    with open('../data/cache/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('../data/cache/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def generate_process_Y():
    if os.path.isfile(CACHE_Y_TRAIN):
        print('Load Y_train from cache!')
        return pd.read_csv(CACHE_Y_TRAIN)
    Y_df = pd.read_csv(DIR_TRAIN_Y, encoding='GBK')
    print('Before process, Y length: {}'.format(Y_df.shape[0]))
    Y_df = Y_df.loc[~Y_df['收缩压'].isin(['未查', '弃查', '0'])]
    
    Y_df = Y_df.loc[~Y_df['舒张压'].isin(['未查', '弃查', '0'])]
    Y_df.loc[Y_df['舒张压']=='100164', '舒张压'] = '164'
    Y_df.loc[Y_df['舒张压']=='974', '舒张压'] = '97'

    Y_df['血清甘油三酯'] = Y_df['血清甘油三酯'].apply(lambda line: "".join(
        [i for i in line if i not in ['>', ' ', '+', '轻', '度', '乳', '糜']]))
    Y_df.loc[Y_df['血清甘油三酯']=='2.2.8', '血清甘油三酯'] = '2.28'
    
    Y_df[['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']] = Y_df[
        ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']].astype(np.float32)
    
#     for feat in ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']:
#         tmp_mean = np.mean(Y_df[feat])
#         tmp_sd = np.std(Y_df[feat])
#         Y_df = Y_df.loc[Y_df[feat].between(tmp_mean-7*tmp_sd, tmp_mean+7*tmp_sd)]
    Y_df.to_csv(CACHE_Y_TRAIN, index=False)
    print('Y_train has been saved {}'.format(CACHE_Y_TRAIN))
    print('After process, Y length: {}'.format(Y_df.shape[0]))
    return Y_df


def merge_table(df):
    df['field_results'] = df['field_results'].astype(str)
    if df.shape[0] > 1:
        merge_df = " ".join(list(df['field_results']))
    else:
        merge_df = df['field_results'].values[0]
    return merge_df


def load_raw_data(vid_to_load):
    if os.path.isfile(CACHE_X_RAW):
        print('Load X_raw from cache!')
        return pd.read_csv(CACHE_X_RAW, dtype=object)
    X_1 = pd.read_csv(DIR_PART1, sep='$')
    X_2 = pd.read_csv(DIR_PART2, sep='$')
    X_all = pd.concat((X_1, X_2)).reset_index(drop=True)
    X_all = X_all.loc[X_all['vid'].isin(vid_to_load)]
    # merge duplicate table_id
    table_size = X_all.groupby(['vid','table_id']).size().reset_index()
    table_size['new_index'] = table_size['vid'] + '_' + table_size['table_id']
    table_dup = table_size[table_size[0]>1]['new_index']
    X_all['new_index'] = X_all['vid'] + '_' + X_all['table_id']
    dup_part = X_all[X_all['new_index'].isin(list(table_dup))]
    dup_part = dup_part.sort_values(['vid','table_id'])
    unique_part = X_all[~X_all['new_index'].isin(list(table_dup))]
    X_all_dup = dup_part.groupby(['vid','table_id']).apply(merge_table).reset_index()
    X_all_dup.rename(columns={0:'field_results'},inplace=True)
    X_all_final = pd.concat([X_all_dup,unique_part[['vid','table_id','field_results']]])
    # pivot X_raw
    X_p = X_all_final.pivot(index='vid', columns='table_id', values='field_results').reset_index()
    
    # Save raw_X
    X_p.to_csv(CACHE_X_RAW, index=False)
    print('X_raw has been saved {}'.format(CACHE_X_RAW))
    return X_p


def filter_low_frequency(df, thresh):
    print('Removing low frequent features ...')
    print('Before: {} columns'.format(len(df.columns)))
    null_count = df.isnull().mean(axis=0)
    cols_to_drop = list(null_count[null_count>(1-thresh)].index)
    print('After: {} columns'.format(len(df.columns)-len(cols_to_drop)))
    return df[[i for i in df.columns if i not in cols_to_drop]]


def FullToHalf(s):
    if isinstance(s, float) or s is None:
        return s
    n = []
    for char in s:
        num = ord(char)
        if num == 12288:
            num = 32
        elif (num >= 65281 and num <= 65374):
            num -= 65248
        num = chr(num)
        n.append(num)
    return ''.join(n)


def generate_process_X(df):
    if os.path.isfile(CACHE_X_PROCESSED):
        print('Load all X_processed from cache!')
        return pd.read_csv(CACHE_X_PROCESSED)
    # filter columns
    df_p = df.copy()
    df_filtered = filter_low_frequency(df_p, 0.1)
    # convert columns based on the table_field type
    df_converted, col_types, cols_to_keep = convert_by_field_type(df_filtered)
    
    df_out = df_converted[['vid']+cols_to_keep]
    df_out.to_csv(CACHE_X_PROCESSED, index=False)
    print('X_processed has been saved {}'.format(CACHE_X_PROCESSED))
    return df_out


def process_num_col(s):
    s = s.apply(FullToHalf)
    s = s.str.extract('([-+]?\d*\.\d+|\d+)', expand=False).astype(float).round(2)
    # deal with outliers
    s = pd.Series(mstats.winsorize(s, limits=[0.01, 0.01])) 
    return s


def process_cat_col(df, col, values_to_onehot):
    new_cols = []
    for value in values_to_onehot:
        tmp_col = col+'_'+value
        df[tmp_col] = (df[col] == value).astype(int)
        new_cols.append(tmp_col)
    return df, new_cols


def process_positive_negative(s):
    s_out = pd.Series([np.nan]*len(s))
    s_out[s.str.contains('未做|未查|弃查', na=False)] = np.nan
    s_out[s.str.contains('\+|阳性|查见|检到|检出', na=False)] = 1
    s_out[s.str.contains('\-|阴性|未查见|未检到|未检出|未见', na=False)] = 0
    s_out[s_out]
    s_out = s_out.astype(float)
    return s_out
    
def convert_by_field_type(df_in):
    df = df_in.copy()
    num_top = 20
    cols_to_convert = [i for i in df.columns if i not in ['vid']]
    cols_num, cols_text, cols_text_num, cols_cat = [], [], [], []
    cols_to_keep = []
    for i, col in enumerate(cols_to_convert):
        tmp_s = df[col]
        print('Processing {}: {}'.format(i, col))
        null_frac_before = np.mean(tmp_s.isnull())
        print('Null fraction (before): {}'.format(null_frac_before))
        tmp_counts = tmp_s.value_counts()
        tmp_counts_top = tmp_counts[:num_top]
        tmp_num_unique = np.sum(tmp_counts>10)
        tmp_counts_num = pd.to_numeric(tmp_counts_top.index.to_series(), errors='coerce')
        tmp_counts_num_extract = tmp_counts_top.index.to_series().str.extract(
            '([-+]?\d*\.\d+|\d+)', expand=False).astype(float).round(2)
        print('Top{} counts of field values: {}'.format(num_top, tmp_counts_top))
        if tmp_num_unique <= num_top:
            print('It is categorical!')
            df, tmp_cols = process_cat_col(df, col, list(tmp_counts[tmp_counts>10].index))
            cols_cat.append(col)
            cols_to_keep.extend(tmp_cols)
        elif (np.sum(tmp_counts_num.isnull())<0.2*num_top or is_numeric_dtype(tmp_s)):
            print('It is numeric!')
            df[col] = process_num_col(df[col])
            cols_num.append(col)
            cols_to_keep.append(col)
        elif np.sum(tmp_counts_num_extract.notnull())>0.2*num_top:
            print('It is numeric with text!')
            df[col+'_num'] = process_num_col(df[col])
            cols_to_keep.append(col+'_num')
            if '阴性' in list(tmp_counts_top.index):
                df[col+'_pn'] = process_positive_negative(df[col])
                cols_to_keep.append(col+'_pn')
            cols_text_num.append(col)
        else:
            print('It is text!')
            df[col+'_len'] = df[col].str.len()
            df_to_concat = process_text(df[col])
            cols_to_keep.append(col+'_len')
            if df_to_concat is not None:
                df = pd.concat((df, df_to_concat), axis=1)
                cols_to_keep.extend(df_to_concat.columns)
            cols_text.append(col)
            

    col_types = {'num':cols_num, 'text':cols_text, 'text_num':cols_text_num, 'cat': cols_cat}
    print({k:len(v) for k, v in col_types.items()})
    return df, col_types, cols_to_keep


def create_text_feats(df):
    if os.path.isfile(CACHE_X_TO_MERGE_2):
        print('Load X_to_merge2 from cache!')
        return pd.read_csv(CACHE_X_TO_MERGE_2)
    cols_text = ['A302', '0117', '0409', '0987', '1305', '1308', 'A201', '0201',
                 '0113', '0207', '0116', '1001', '0114', '1303', '0501', '0225',
                 '4001', '0216', '0421', '1313', '1330', '0985', '0978', '0929',
                 '0420', '0222', '0984', '0124', '0115', '1102', '0440', '0516',
                 '0426', '0119', '0912', '0954', '0983', '1316', '0423', '0123',
                 '1315', '0120', '1301', '0973', '1402', 'A202', '0209', '0975',
                 '0901', '0509', '0707', '0911', '0405', '0949', '0503', '0202',
                 '0215', '0434', '0203', '0217', '0427', '0213', '0979', '2501',
                 'A301', '3601', '0537', '0947', '0972', '0541', '0435', '0210',
                 '0118', '0539', '0715', '0546', '0208', '0974', '0101', '0436',
                 '0413', '1302', '0102', '0406', '1314', '0122', '1103', '0431', '0121']
    df_to_merge = pd.DataFrame()
    df_to_merge['vid'] = df['vid']
    # sex columns
    df_to_merge['num_woman_cols'] = np.sum(df[['0501', '0503', '0509', '0516', '0539', '0537', '0539', '0541', '0550',
                                              '0551', '0549', '0121', '0122', '0123']].notnull(), axis=1)
    df_to_merge['num_man_cols'] = np.sum(df[['0120', '0125', '0981', '0984', '0983']].notnull(), axis=1)
    # keywords for diseases
#     X_1 = pd.read_csv(DIR_PART1, sep='$')
#     X_2 = pd.read_csv(DIR_PART2, sep='$')
#     X_all = pd.concat((X_1, X_2)).reset_index(drop=True)
# '舒张期杂音', '收缩期杂音','呼吸音清', '呼吸音粗', '呼吸音减弱','减慢', '减弱', '增快', '降低','低盐', '低脂'
    keywords_ch = ['糖尿病', '高血压', '血脂', '治疗中', '肥胖', '血糖', '血压高', '血脂偏高', '血压高偏高', '冠心病',
                  '脂肪肝', '不齐', '过缓', '血管弹性', '脂'
                  '硬化', '舒张期杂音', '收缩期杂音', '低盐', '低脂']
    keywords_en = ['disease_'+str(i) for i in range(len(keywords_ch))]
    tmp_df = df[cols_text]
    for i, col in enumerate(keywords_en):
        tmp_df_2 = tmp_df.applymap(lambda x: keywords_ch[i] in str(x))
        df_to_merge[col] = np.sum(tmp_df_2, axis=1)
#     df_to_merge['disease_sum'] = np.sum(df_to_merge[keywords_en], axis=1)
    # 2302
    df_to_merge['if_subhealth'] = (df['2302'] == '亚健康').astype(int)
    df_to_merge['if_ill'] = (df['2302'] == '疾病').astype(int)
    df_to_merge['if_health'] = (~(df_to_merge['if_subhealth']&df_to_merge['if_ill'])).astype(int)
#     # 1402: 减慢 增快 降低
#     df_to_merge['if_jianman'] = df['1402'].apply(lambda x: '减慢' in str(x)).astype(int)
#     df_to_merge['if_zengkuai'] = df['1402'].apply(lambda x: '增快' in str(x)).astype(int)
#     df_to_merge['if_jiangdi'] = df['1402'].apply(lambda x: '降低' in str(x)).astype(int)

    df_to_merge.to_csv(CACHE_X_TO_MERGE_2, index=False)
    print('X_to_merge2 has been saved {}'.format(CACHE_X_TO_MERGE_2))
    
    return df_to_merge


def process_text(s):
    keywords_ch = ['糖尿病', '高血压', '血脂', '脂肪肝', '慢性胃炎', '阑尾炎', '甲肝', '肾结石',
                   '胆囊切除', '甲肝', '冠心病', '胆结石', '甲状腺', '脑梗塞', '胆囊炎', '脑溢血', 
                  '早搏', '杂音', '心动过缓', '心律不齐', '心动过速']
    keywords_en = ['disease_'+str(i) for i in range(len(keywords_ch))]
    dict_out = {}
    if s.name in ['0434', '0409']:
        for i, kw in enumerate(keywords_en):
            s_out = pd.Series([np.nan]*len(s))
            s_out[s.str.contains('{}|阳性|查见|检到|检出'.format(keywords_ch[i]), na=False)] = 1
            s_out[s.str.contains('无|未查见|健康|未见', na=False)] = 0
            if np.sum(s_out) > 20:
                dict_out[s.name+'_'+kw] = s_out
        df_to_concat = pd.DataFrame(dict_out)
    elif s.name == '0113':
        s_out = pd.Series([np.nan]*len(s))
        s_out[s.str.contains('无|正常|清晰', na=False)] = 0
        s_out[s.str.contains('弥漫性增强|不清晰|斑点状强回声|欠清晰', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name == '0114':
        s_out = pd.Series([np.nan]*len(s))
        s_out[s.str.contains('无|正常|清晰', na=False)] = 0
        s_out[s.str.contains('强回声|毛糙|增厚|伴声影', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)        
    elif s.name == '0115':
        s_out = pd.Series([np.nan]*len(s))
        s_out[s.str.contains('无|正常', na=False)] = 0
        s_out[s.str.contains('增强', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name == '0116':
        s_out = pd.Series([np.nan]*len(s))
        s_out[s.str.contains('无|正常', na=False)] = 0
        s_out[s.str.contains('结构回声|回声增强|强回声|高回声', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)  
    elif s.name == '0117':
        s_out = pd.Series([np.nan]*len(s))
        s_out[s.str.contains('无|正常', na=False)] = 0
        s_out[s.str.contains('无回声|强回声|高回声|回声增强', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name == '0118':
        s_out = pd.Series([np.nan]*len(s))
        s_out[s.str.contains('无|正常', na=False)] = 0
        s_out[s.str.contains('无回声|强回声|高回声|回声增强', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)    
    elif s.name == '0209':
        s_out = pd.Series([np.nan]*len(s))
        s_out[s.str.contains('无|正常|未见', na=False)] = 0
        s_out[s.str.contains('充血|鼻炎|息肉', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name == '0215':
        s_out = pd.Series([np.nan]*len(s))
        s_out[s.str.contains('无|正常|未见', na=False)] = 0
        s_out[s.str.contains('充血|咽炎|增生', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name == '0912':
        s_out = pd.Series([np.nan]*len(s))
        s_out[s.str.contains('无|正常|未见|不肿大|未肿大', na=False)] = 0
        s_out[s.str.contains('结节|略大|欠光滑|甲状腺肿大', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name == '1001':
        s_out = pd.Series([np.nan]*len(s))
        s_out[s.str.contains('无|正常', na=False)] = 0
        s_out[s.str.contains(
            '不齐|心动过缓|电轴左偏|低电压|电轴右偏|高电压|T波|心动过速|肥厚', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name == '1302':
        s_out = pd.Series([np.nan]*len(s))
        s_out[s.str.contains('无|正常|未见异常', na=False)] = 0
        s_out[s.str.contains(
            '结膜炎|胬肉|结膜结石|素斑|裂斑|高电压|T波|心动过速|肥厚', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name in ['1330', '1316']:
        s_out = pd.Series([np.nan]*len(s))
        s_out[s.str.contains('无|正常|未见异常', na=False)] = 0
        s_out[s.str.contains(
            '动脉硬化|动脉变细|黄斑|白内障|变性|高电压|弧形斑|视网膜病变', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name in ['1402']:
        s_out = pd.Series([np.nan]*len(s))
        s_out[s.str.contains('无|正常|未见异常|未见狭窄|弹性良好', na=False)] = 0
        s_out[s.str.contains(
            '速度减慢|弹性降低|顺应性降低|速度增快|脑血管痉挛|略增快', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name in ['3601']:
        s_out = pd.Series([np.nan]*len(s))
        s_out[s.str.contains('无|正常|未见异常', na=False)] = 0
        s_out[s.str.contains(
            '减少|降低|骨密度降低|疏松', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)        
    elif s.name in ['4001']:
        s_out = pd.Series([np.nan]*len(s))
        s_out[s.str.contains('无|正常|未见异常|未见狭窄|弹性良好', na=False)] = 0
        s_out[s.str.contains(
            '轻度减弱|减弱趋势|中度减弱|重度减弱|稍硬|略增快|轻度硬化|动脉硬化|血压升高|堵塞|狭窄', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    else:
        return None
    return df_to_concat


def interact_feats(df):
    df_in = df.copy()
    df_in['w2h'] = df_in['2403'] / df_in['2404']
    # 血清甘油三酯 :193, 1117, 191, 1850, 10004, 1814
#     df_in['2_top3'] = df_in['191'] * df_in['1117'] * df_in['193']
    df_in['2_top5'] = df_in['191'] * df_in['1117'] * df_in['193'] * df_in['1850'] * df_in['10004']
#     # 4 top5: 193, 1850, 192, 1117, 1107, 10004
#     df_in['4_top5'] = df_in['192'] * df_in['1107']
    return df_in


def count_location_per_vid(df):
    my_cols = list(df.columns)
    cols_six = re.findall(r'\d{6}', '{}'.format(my_cols))
    pnt_l, pnt_r = 0, 0
    col_first2_cur = cols_six[pnt_l][:2]
    tmp_list = []
    df_to_merge = pd.DataFrame()
    df_to_merge['vid'] = df['vid']
    while pnt_r!=len(cols_six)-1:
        pnt_r += 1
        if cols_six[pnt_r][:2] != col_first2_cur:
            tmp_list.append(cols_six[pnt_r-1])
            df_to_merge['num_'+col_first2_cur] = np.sum(df[tmp_list].notnull(), axis=1)
            tmp_list = []
            pnt_l = pnt_r
            col_first2_cur = cols_six[pnt_l][:2]
        else:
            tmp_list.append(cols_six[pnt_r])
    df_to_merge['num_'+col_first2_cur] = np.sum(df[tmp_list].notnull(), axis=1)
    return df_to_merge


def create_low_freq_feats(df, thresh):
    null_count = df.isnull().mean(axis=0)
    cols_to_drop = list(null_count[null_count>(1-thresh)].index)
    df['count_low_freq'] = df[cols_to_drop].notnull().sum(axis=1)
    return df[['vid', 'count_low_freq']]


class ModelRegVal():
    def __init__(self, model_name, params):
        self.name = model_name
        self.params = params
    
    def train(self, X_train, y_train, X_val, y_val):
        if self.name == 'lgb':
            num_rounds = self.params['num_rounds_lgb']
            num_early_stop = self.params['num_early_stop']
            dtrain, dval = lgb.Dataset(X_train, y_train), lgb.Dataset(X_val, y_val)
            self.model = lgb.train(self.params, dtrain, num_boost_round=num_rounds,
                                   valid_sets=dval, early_stopping_rounds=num_early_stop,
                                   verbose_eval=False, feval=log1p_mse)
            print(self.model.best_iteration)
            return (self.model,
                    self.model.best_iteration,
                    self.model.best_score['valid_0']['l2'])
        
    def predict(X_test):
        if self.name == 'lgb':
            dtest = X_test
            num_it = self.model.best_iteration
            return self.model.predict(dtest, num_iteration=num_it)
        

def train_folds_targets(kf, vid_train, labels, X_processed, Y_train, stage=None, **kwargs):
    loss, models, counter = {}, {}, 1
    X_df_append = None
    if stage is not None:
        X_df_append = pd.DataFrame()
    print('Validating ...')
    kf_vid_dict = {}
    for tr_idx, val_idx in kf.split(vid_train):
        kf_vid_dict[counter-1] = list(np.array(vid_train)[val_idx])
        print('Fold{}'.format(counter))
        X_tr, X_val = (X_processed.loc[X_processed['vid'].isin([vid_train[i] for i in tr_idx])
                                      ].sort_values(by='vid').reset_index(drop=True),
                       X_processed.loc[X_processed['vid'].isin([vid_train[i] for i in val_idx])
                                      ].sort_values(by='vid').reset_index(drop=True))
        if IF_AUG:
            X_tr = augment_training_data(X_tr, 0.1, 0.1, 5)

        X_tr_array, X_val_array = (
            sparse.csr_matrix(X_tr.drop(columns='vid').values),
            sparse.csr_matrix(X_val.drop(columns='vid').values))
        
        Y_tr, Y_val = pd.DataFrame(), pd.DataFrame()
        Y_tr['vid'], Y_val['vid'] = X_tr['vid'], X_val['vid']
        Y_tr = Y_tr.merge(Y_train, how='left', on='vid')
        Y_val = Y_val.merge(Y_train, how='left', on='vid')
        
        tmp_loss = {}
        tmp_model = {}
        tmp_df = pd.DataFrame()
        tmp_df['vid'] = X_val['vid']
        for label in labels:
            if stage == 2:
                tmp_X_tr = np.append(X_tr.drop(columns='vid').values, 
                                     np.abs(Y_tr.drop(columns=['vid', label]).values), axis=1)
                tmp_X_val = np.append(X_val.drop(columns='vid').values, 
                                      np.abs(Y_val.drop(columns=['vid', label]).values), axis=1)
                X_tr_array, X_val_array = (sparse.csr_matrix(tmp_X_tr), sparse.csr_matrix(tmp_X_val))
            if 'feat_imp' in kwargs:
                X_tr_array_imp = X_tr_array[:,kwargs['feat_imp'][label]]
                X_val_array_imp = X_val_array[:,kwargs['feat_imp'][label]]
            else:
                X_tr_array_imp = X_tr_array
                X_val_array_imp = X_val_array
            y_tr_array = np.log1p(np.abs(Y_tr[label].values))
            y_val_array = np.log1p(np.abs(Y_val[label].values))
            tmp_regressor = ModelRegVal(MODEL, PARAMS[MODEL])
            model_bst,_,loss_bst = tmp_regressor.train(X_tr_array_imp, y_tr_array, X_val_array_imp, y_val_array)
            tmp_loss[label] = loss_bst
            tmp_model[label] = model_bst
            if stage is not None:
                tmp_pred = model_bst.predict(X_val_array)
                tmp_df[label+'_'+str(stage)] = tmp_pred
            print('  {} loss: {}'.format(label, loss_bst))
        X_df_append = X_df_append.append(tmp_df, ignore_index=True)
        loss[counter] = tmp_loss
        models[counter] = tmp_model
        counter += 1
    # save kf_dict
    save_obj(kf_vid_dict, 'kf_vid_dict_wangsheng')
    return models, X_df_append, loss


def calc_loss_statistics(loss_dict):
    loss_list = []
    for k,v in loss_dict.items():
        tmp_loss = []
        for k_1,v_1 in v.items():
            tmp_loss.append(v_1)
        loss_list.append(tmp_loss)
    loss_a = np.array(loss_list)
    loss_folds = np.mean(loss_a, axis=1)
    loss_m = np.mean(loss_folds)
    loss_std = np.std(loss_folds)
    return loss_folds, loss_m, loss_std


def predict_folds(models, X_test, **kwargs):
    pred_list = []
    for k, v in models.items():
        tmp_pred_1 = []
        for k_1,v_1 in v.items():
            if 'feat_imp' in kwargs:
                X_test_imp = X_test[:, kwargs['feat_imp'][label]]
            else:
                X_test_imp = X_test
            tmp_pred_1.append(v_1.predict(X_test_imp))
        pred_list.append(tmp_pred_1)
    return np.median(np.array(pred_list), axis=0)


def predict_s2(models, X_test_df, pred_test):
    pred_list = []
    for k, v in models.items():
        tmp_pred_1 = []
        counter = 0
        for k_1,v_1 in v.items():
            tmp_X_test = np.append(
                X_test_df.drop(columns='vid').values, np.delete(pred_test, counter, 0).transpose(), axis=1)
            tmp_pred_1.append(v_1.predict(tmp_X_test))
        pred_list.append(tmp_pred_1)
    return np.mean(np.array(pred_list), axis=0)


def log1p_mse(preds, train_data):
    labels = train_data.get_label()
    result = np.mean((np.log1p(preds) - np.log1p(labels))**2)
    return 'error', result, False


def augment_training_data(df, row_fraction, col_fraction, rep):
    cols_feats = [i for i in df.columns if i!='vid']
    for tmp_rep in range(rep):
        tmp_pd_to_append = df.sample(frac=row_fraction, weights=None, random_state=RANDOM_SEED+tmp_rep)
        np.random.seed(RANDOM_SEED+tmp_rep)
        tmp_col = np.random.choice(cols_feats, int(len(cols_feats)*col_fraction), replace=False)
        tmp_pd_to_append[tmp_col] = np.nan
        df = pd.concat((df, tmp_pd_to_append))
    return df


if __name__ == '__main__':
    shutil.rmtree('../data/cache', ignore_errors=True)
    os.makedirs('../data/cache')

    Y_train = generate_process_Y()
    labels = [i for i in Y_train.columns if i!='vid']
    vid_train = list(pd.read_csv(DIR_TRAIN_Y)['vid'].values)
    vid_test = list(pd.read_csv(DIR_TEST_Y, encoding='GBK')['vid'].values)
    X_raw = load_raw_data(vid_train+vid_test)
    print('Processing X_train ...')
    # 生成四类特征：1:纯文本2:纯数值;3:文本+数值;4:categorical
    X_processed = generate_process_X(X_raw)
    X_processed = interact_feats(X_processed)
    # 临时加一波特征
    X_to_merge1 = count_location_per_vid(X_raw)
    X_processed = X_processed.merge(X_to_merge1, how='left', on='vid')
    # 再加一波特征
    X_to_merge2 = create_text_feats(X_raw)
    X_processed = X_processed.merge(X_to_merge2, how='left', on='vid')
    # 第三波特征
    X_to_merge3 = create_low_freq_feats(X_raw, 0.1)
    X_processed = X_processed.merge(X_to_merge3, how='left', on='vid')

    print(X_processed.columns)
    print('Number of columns:', len(X_processed.columns))   


    if IF_VAL:
        kf = KFold(n_splits=NUM_FOLDS, random_state=188, shuffle=True)
        # train stage_1 models and predict for the training dataset
        models, pred_train, loss = train_folds_targets(kf, vid_train, labels, X_processed, Y_train, stage=1)
        loss_folds, loss_m, loss_std = calc_loss_statistics(loss)
        print('The loss of folds:{}. Mean={}; std={}'.format(loss_folds, loss_m, loss_std))
        # predict for test
        print('Predicting s1...')
        X_test_df = X_processed.loc[X_processed['vid'].isin(vid_test)].sort_values(by='vid').reset_index(drop=True)
        X_test = sparse.csr_matrix(X_test_df.drop(columns='vid').values)
        pred_test_s1 = predict_folds(models, X_test)
        # stage 1 submit
        Y_test_s1 = pd.read_csv(DIR_TEST_Y, encoding='GBK')
        pred_df_s1 = pd.DataFrame()
        pred_df_s1['vid'] = X_test_df['vid']
        for i, label in enumerate(labels):
            pred_df_s1[label] = np.expm1(np.round(pred_test_s1[i], 6))
        Y_test_s1 = Y_test_s1.merge(pred_df_s1, how='left', on='vid')
        Y_test_s1 = Y_test_s1[['vid']+[i+'_y' for i in labels]]
        now = datetime.datetime.now()
        sub_name = DIR_SUB + '{}_pred_test_awang.csv'.format(MODEL)
        Y_test_s1.to_csv(sub_name, index=False, header=False)

        pred_train.set_index('vid', drop=True, inplace=True)
        pred_train_reorder = pred_train.loc[vid_train]
        pred_train_reorder.to_csv(DIR_SUB + '{}_pred_train_awang.csv'.format(MODEL), index=False)

        print('Done!')