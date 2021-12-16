import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_data(data_type, stack, data_path, window, use_cols):
    """ return the processed data

    data_type: str
        the type of finance data (ex.FX)

    stack: int
        stack the data for input to ML model

    data_path: str
        path of the data

    window: Tuple
        (num: int, period: str)
        num is only used if period is 'minute' or 'hour'.
        you choose anyone of ['minute', 'hour', 'daily', 'weekly', 'monthly'] as period.

    use_cols: list
        the features of use in DataFrame
        use_cols -> ['Close', 'Open', 'High', 'Low', 'Volume]

    
    """
    if data_type == 'FX':
        data, raw_return = get_fx_dataset(data_path, window, use_cols)
    else:
        raise NotImplementedError(f'Dataset {data_type} not valid')
    assert data.shape[0] == 1
    input_data = _data_stack(data[0], stack)
    raw_return = raw_return[:-stack]
    return input_data, raw_return

def get_fx_dataset(data_path, window, use_cols):
    df_raw = pd.read_csv(data_path)
    df_raw = _extract_data(df_raw, window)

    if 'Close' not in use_cols:
        raise NotImplementedError()
    price = df_raw[['Close']].values
    raw_return = (price[1:] - price[:-1]).flatten()

    feature_lists = []
    for col in use_cols:
        if col == 'Volume':
            feature = np.log(df_raw[[col]].values[1:]).reshape(1, -1, 1)
        else:
            log_price = np.log(df_raw[[col]].values)
            feature = (log_price[1:] - log_price[:-1]).reshape(1, -1, 1)

        feature_lists.append(feature)
    data_preprocessed = np.concatenate(feature_lists, axis=-1)
    data = standardscaler(data_preprocessed)
    return data, raw_return

def standardscaler(x, axis=1):
    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    return (x - mean) / std

def _data_stack(x, x_lag, add_batch_dim=True):
    if add_batch_dim:
        x = x[None, ...]
    return np.concatenate([x[:, t:t + x_lag] for t in range(x.shape[1] - x_lag)], axis=0)

def _extract_data(df, window):
    if 'Datetime' in df.keys():
        df['Datetime'] = pd.to_datetime(df['Datetime'])
    else:
        raise NotImplementedError()
    
    if window is not None:
        num, period = window
        if period == 'minute':
            index = (df['Datetime'].dt.minute % num == 0)
        elif period == 'hour':
            index = (df['Datetime'].dt.hour % num == 0) & (df['Datetime'].dt.minute == 0)
        elif period == 'daily':
            index = (df['Datetime'].dt.hour == 0) & (df['Datetime'].dt.minute == 0)
        elif period == 'weekly':
            index = (df['Datetime'].dt.dayofweek == 0) & (df['Datetime'].dt.hour == 0) & (df['Datetime'].dt.minute == 0)
        elif period == 'monthly':
            index = (df['Datetime'].dt.day == 1) & (df['Datetime'].dt.hour == 0) & (df['Datetime'].dt.minute == 0)
        else:
            raise NotImplementedError()
    return df[index]
