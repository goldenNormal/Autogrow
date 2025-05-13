import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

import json
import polars as pl

'''function  read_dataset/read_avazu/read_criteo/read_movielens1m is copied from https://github.com/Applied-Machine-Learning-Lab/ERASE'''
def read_dataset(dataset_name, data_path, batch_size, shuffle, num_workers, use_fields=None, machine_learning_method=False):
    if dataset_name == 'avazu':
        return read_avazu(data_path, batch_size, shuffle, num_workers, use_fields)
    elif dataset_name == 'criteo':
        return read_criteo(data_path, batch_size, shuffle, num_workers, use_fields)
    elif dataset_name == 'movielens-1m':    
        return read_movielens1m(data_path, batch_size, shuffle, num_workers, use_fields)
    elif dataset_name == 'aliccp':
        return read_aliccp(data_path, batch_size, shuffle, num_workers, use_fields)

def read_avazu(data_path, batch_size, shuffle, num_workers, use_fields=None):
    dtypes = {
        'click': np.int8,
        'hour':np.int16,
        'C1':np.int8,
        'banner_pos':np.int8,
        'site_id':np.int16,
        'site_domain':np.int16,
        'site_category':np.int8,
        'app_id':np.int16,
        'app_domain':np.int16,
        'app_category':np.int8,
        'device_id':np.int32,
        'device_ip':np.int32,
        'device_model':np.int16,
        'device_type':np.int8,
        'device_conn_type':np.int8,
        'C14':np.int16,
        'C15':np.int8,
        'C16':np.int8,
        'C17':np.int16,
        'C18':np.int8,
        'C19':np.int8,
        'C20':np.int16,
        'C21':np.int8
    }
    print('start reading avazu...')
    if use_fields is None:
        df = pd.read_csv(os.path.join(data_path, 'avazu/preprocessed_avazu.csv'), dtype = dtypes)
    else:
        df = pd.read_csv(os.path.join(data_path, 'avazu/preprocessed_avazu.csv'), dtype = dtypes, usecols=list(use_fields)+['click'])
    print('finish reading avazu.')
    train_idx = int(df.shape[0] * 0.7)
    val_idx = int(df.shape[0] * 0.9)
    features = [f for f in df.columns if f not in ['click']]
    unique_values = [df[col].max()+1 for col in features]
    label = 'click'
    
    return features, label, train_idx,val_idx, unique_values,df

def read_criteo(data_path, batch_size, shuffle, num_workers, use_fields=None):
    dtypes = {
        '0': np.int8,
        '1': np.int8,
        '2': np.int8,
        '3': np.int8,
        '4': np.int8,
        '5': np.int16,
        '6': np.int16,
        '7': np.int8,
        '8': np.int8,
        '9': np.int8,
        '10': np.int8,
        '11': np.int8,
        '12': np.int8,
        '13': np.int8,
        '14': np.int16,
        '15': np.int16,
        '16': np.int32,
        '17': np.int32,
        '18': np.int16,
        '19': np.int8,
        '20': np.int16,
        '21': np.int16,
        '22': np.int8,
        '23': np.int32,
        '24': np.int16,
        '25': np.int32,
        '26': np.int16,
        '27': np.int8,
        '28': np.int16,
        '29': np.int32,
        '30': np.int8,
        '31': np.int16,
        '32': np.int16,
        '33': np.int8,
        '34': np.int32,
        '35': np.int8,
        '36': np.int8,
        '37': np.int32,
        '38': np.int8,
        '39': np.int32
    }
    print('start reading criteo...')
    if use_fields is None:
        df = pd.read_csv(os.path.join(data_path, 'criteo/preprocessed_criteo.csv'), dtype = dtypes)
#         df.drop(columns=['index:float'], inplace=True)
    else:
        df = pd.read_csv(os.path.join(data_path, 'criteo/preprocessed_criteo.csv'), dtype = dtypes, usecols=list(use_fields)+['0'])
    print('finish reading criteo.')
    train_idx = int(df.shape[0] * 0.7)
    val_idx = int(df.shape[0] * 0.9)
    features = [f for f in df.columns if f not in ['0']]
    unique_values = [df[col].max()+1 for col in features]
    label = '0'
    
    return features, label, train_idx, val_idx, unique_values,df


def read_movielens1m(data_path, batch_size, shuffle, num_workers, use_fields=None):
    print('start reading movielens 1m...')
    if use_fields is None:
        df = pd.read_csv(os.path.join(data_path, 'movielens-1m/ml-1m.csv'))
    else:
        df = pd.read_csv(os.path.join(data_path, 'movielens-1m/ml-1m.csv'), usecols=list(use_fields)+['rating'])
    print('finish reading movielens 1m.')
    df['rating'] = df['rating'].apply(lambda x: 1 if x > 3 else 0)
    df = df.sample(frac=1, random_state=43) # shuffle
    train_idx = int(df.shape[0] * 0.7)
    val_idx = int(df.shape[0] * 0.9)
    features = [f for f in df.columns if f not in ['rating']]
    for feature in features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
    unique_values = [df[col].max()+1 for col in features]
    label = 'rating'
    
    
    return features, label,train_idx, val_idx, unique_values, df


def read_aliccp(data_path, batch_size, shuffle, num_workers, use_fields=None):
    print('start reading aliccp...')
    data_type = {'click':np.int8, 'purchase': np.int8, '101':np.int32, '121':np.uint8, '122':np.uint8, '124':np.uint8, '125':np.uint8, '126':np.uint8, '127':np.uint8, '128':np.uint8, '129':np.uint8, '205':np.int32, '206':np.int16, '207':np.int32, '210':np.int32, '216':np.int32, '508':np.int16, '509':np.int32, '702':np.int32, '853':np.int32, '301':np.int8, '109_14':np.int16, '110_14':np.int32, '127_14':np.int32, '150_14':np.int32, 'D109_14': np.float16, 'D110_14': np.float16, 'D127_14': np.float16, 'D150_14': np.float16, 'D508': np.float16, 'D509': np.float16, 'D702': np.float16, 'D853': np.float16}
    if use_fields is None:
        df1 = pd.read_csv(os.path.join(data_path, 'aliccp/ali_ccp_train.csv'), dtype=data_type)
        df2 = pd.read_csv(os.path.join(data_path, 'aliccp/ali_ccp_val.csv'), dtype=data_type)
        df3 = pd.read_csv(os.path.join(data_path, 'aliccp/ali_ccp_test.csv'), dtype=data_type)
        df = pd.concat([df1, df2, df3])
    else:
        df1 = pd.read_csv(os.path.join(data_path, 'aliccp/ali_ccp_train.csv'), usecols=list(use_fields)+['click'], dtype=data_type)
        df2 = pd.read_csv(os.path.join(data_path, 'aliccp/ali_ccp_val.csv'), usecols=list(use_fields)+['click'], dtype=data_type)
        df3 = pd.read_csv(os.path.join(data_path, 'aliccp/ali_ccp_test.csv'), usecols=list(use_fields)+['click'], dtype=data_type)
        df = pd.concat([df1, df2, df3])
    print('finish reading aliccp.')
    # df = df.sample(frac=1) # shuffle
    train_idx = int(df.shape[0] * 0.5)
    val_idx = int(df.shape[0] * 0.75)
    features = []
    for f in df.columns:
        if f not in ['click','purchase'] and f[:1] != 'D':
            features.append(f)
    if '301' in features:
        df['301'] = df['301'] - 1
    unique_values = [df[col].max()+1 for col in features]
    label = 'click'
    return features, label, train_idx, val_idx, unique_values,df


def produce_quick_data(data_path='./data'):
    for dataset in ['movielens-1m', 'aliccp', 'avazu', 'criteo']:
        print(f'... begin {dataset}...')
        shuffle=True
        num_workers = 32
        use_fields = None
        batch_size = 4096
        features, label, train_idx,val_idx, unique_values,df = read_dataset(dataset, data_path, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, use_fields=None, machine_learning_method=False)

        import polars as pl
        pl.from_pandas(df).write_parquet(f'{data_path}/{dataset}.parquet')

        data_meta = {'features': features, 'label':label}
        import json
        with open(f'{data_path}/{dataset}.json','w') as f:
            json.dump(data_meta,f)
        print(f'... end {dataset}...')
    
def quick_read_dataset(dataset, data_path = './quick_data', selected_features=None, train_ratio_pair =(0.0,0.8), val_ratio_pair=(0.8,0.9), test_ratio_pair =(0.9,1.0)):
    df = pl.read_parquet(f'{data_path}/{dataset}.parquet').to_pandas()
    with open(f'{data_path}/{dataset}.json','r') as f:
        data_meta = json.load(f)
    if selected_features is None:
        features = data_meta['features']
    else:
        features = selected_features
    label = data_meta['label']
    if type(label) == type('str'):
        features = [str(f) for f in features]
    use_cols = features + [label]
    df = df[use_cols]
    unique_values = [df[col].max()+1 for col in features]

    # split train val test
    ratio2idx = lambda ratio_pair: (int(df.shape[0] * ratio_pair[0]),int(df.shape[0] * ratio_pair[1]))
    train_idxs = ratio2idx(train_ratio_pair)
    val_idxs = ratio2idx(val_ratio_pair)
    test_idxs = ratio2idx(test_ratio_pair)


    train_x, val_x, test_x = df[features][train_idxs[0]:train_idxs[1]], df[features][val_idxs[0]:val_idxs[1]], df[features][test_idxs[0]:test_idxs[1]]
    train_y, val_y, test_y = df[label][train_idxs[0]:train_idxs[1]], df[label][val_idxs[0]:val_idxs[1]], df[label][test_idxs[0]:test_idxs[1]]
    
    return features, label, unique_values, (train_x, train_y, val_x, val_y, test_x, test_y)




if __name__ == 'main':
    # First, download the dataset from https://huggingface.co/datasets/Jia-py/ERASE_Dataset to the `./data` directory,
    # then execute the function `produce_quick_data` below to generate a new dataset in the `./quick_data` directory, which has smaller storage and faster reading speed.
    produce_quick_data()