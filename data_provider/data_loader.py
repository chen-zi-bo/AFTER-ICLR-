import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


DATASET_CONFIGS = {
    'agriculture': {
        'file_name': 'Agriculture.csv',
        'text_col': 'fact',
    },
    'climate': {
        'file_name': 'Climate.csv',
        'text_col': 'fact',
    },
    'economy': {
        'file_name': 'Economy_sort.csv',
        'text_col': 'fact',
    },
    'energy': {
        'file_name': 'Energy.csv',
        'text_col': 'fact',
    },
    'environment': {
        'file_name': 'Environment.csv',
        'text_col': 'fact',
    },
    'health': {
        'file_name': 'Health.csv',
        'text_col': 'fact',
    },
    'security': {
        'file_name': 'Security.csv',
        'text_col': 'fact',
    },
    'socialgood': {
        'file_name': 'SocialGood.csv',
        'text_col': 'fact',
    },
    'traffic': {
        'file_name': 'Traffic.csv',
        'text_col': 'fact',
    },
    'weather': {
        'file_name': 'weather.csv',
        'text_col': 'fact',
    },
}


def _default_window_size(size):
    if size is None:
        return 24 * 4 * 4, 24 * 4, 24 * 4
    return size[0], size[1], size[2]


def _split_borders(length, seq_len):
    num_train = int(length * 0.7)
    num_test = int(length * 0.2)
    num_vali = length - num_train - num_test
    border1s = [0, num_train - seq_len, length - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, length]
    return border1s, border2s


def _build_time_stamp(df_stamp, timeenc, freq, date_col):
    df_stamp = df_stamp.copy()
    df_stamp[date_col] = pd.to_datetime(df_stamp[date_col])
    if timeenc == 0:
        df_stamp['month'] = df_stamp[date_col].apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp[date_col].apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp[date_col].apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp[date_col].apply(lambda row: row.hour, 1)
        return df_stamp.drop(columns=[date_col]).values

    data_stamp = time_features(pd.to_datetime(df_stamp[date_col].values), freq=freq)
    return data_stamp.transpose(1, 0)


class Dataset_TextTimeSeries(Dataset):
    def __init__(self, args, dataset_name, flag='train', size=None,
                 target='OT', scale=True, timeenc=0, freq='h', channel_independent=False):
        self.args = args
        self.dataset_name = dataset_name.lower()
        if self.dataset_name not in DATASET_CONFIGS:
            raise ValueError(f'Unknown dataset: {dataset_name}')

        self.seq_len, self.label_len, self.pred_len = _default_window_size(size)
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.channel_independent = channel_independent
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.config = DATASET_CONFIGS[self.dataset_name]
        self.__read_data__()
        self.num_features = self.data_x.shape[-1]
        self.variable_names = [self.target]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()

        file_path = os.path.join('./data', self.config['file_name'])
        df_raw = pd.read_csv(file_path)
        if 'date' not in df_raw.columns:
            raise ValueError(f'{self.config["file_name"]} must contain a date column')

        df_raw = df_raw.copy()
        df_raw['date'] = pd.to_datetime(df_raw['date'], errors='coerce')

        self.text_col = self.config['text_col']
        text_series = df_raw[self.text_col]

        border1s, border2s = _split_borders(len(df_raw), self.seq_len)
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw[[self.target]]
        self.data_raw = df_data.values

        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']].iloc[border1:border2].copy()
        self.data_stamp = _build_time_stamp(df_stamp, self.timeenc, self.freq, 'date')
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.date = df_raw['date'].iloc[border1:border2].values
        self.text = text_series.iloc[border1:border2].tolist()

    def get_text(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()
        indices = np.asarray(indices)

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        return [self.text[s_begin:s_end] for s_begin, s_end in zip(s_begins, s_ends)]

    def get_date(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()
        indices = np.asarray(indices)

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        return np.array([self.date[s_beg:s_end] for s_beg, s_end in zip(s_begins, s_ends)])

    def get_forecast_date(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()
        indices = np.asarray(indices)

        s_begins = indices % self.tot_len
        r_begins = s_begins + self.seq_len
        r_ends = r_begins + self.pred_len
        return np.array([self.date[r_beg:r_end] for r_beg, r_end in zip(r_begins, r_ends)])

    def __getitem__(self, index):
        feat_id = index // self.tot_len

        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if self.channel_independent:
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        else:
            seq_x = self.data_x[s_begin:s_end, :]
            seq_y = self.data_y[r_begin:r_end, :]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, index

    def __len__(self):
        if self.channel_independent:
            return max(self.tot_len * self.num_features, 0)
        return max(len(self.data_x) - self.seq_len - self.pred_len + 1, 0)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Agriculture(Dataset_TextTimeSeries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_name='agriculture', **kwargs)


class Dataset_Climate(Dataset_TextTimeSeries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_name='climate', **kwargs)


class Dataset_Economy(Dataset_TextTimeSeries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_name='economy', **kwargs)


class Dataset_Energy(Dataset_TextTimeSeries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_name='energy', **kwargs)


class Dataset_Environment(Dataset_TextTimeSeries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_name='environment', **kwargs)


class Dataset_Health(Dataset_TextTimeSeries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_name='health', **kwargs)


class Dataset_Security(Dataset_TextTimeSeries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_name='security', **kwargs)


class Dataset_SocialGood(Dataset_TextTimeSeries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_name='socialgood', **kwargs)


class Dataset_Traffic(Dataset_TextTimeSeries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_name='traffic', **kwargs)


class Dataset_Weather(Dataset_TextTimeSeries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_name='weather', **kwargs)


class Dataset_GDELT(Dataset):
    def __init__(self, args, flag='train', size=None,
                 event_root_code=1, target='NumMentions',
                 scale=True, freq='h', percent=100,
                 channel_independent=False, timeenc=0):
        self.args = args
        self.seq_len, self.label_len, self.pred_len = _default_window_size(size)

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.num_features = 3
        self.target = target
        self.scale = scale
        self.freq = freq
        self.percent = percent
        self.timeenc = timeenc
        self.event_root_code = event_root_code
        self.channel_independent = channel_independent
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.variable_names = ['NumMentions', 'NumSources', 'NumArticles']
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join('./data/gpt4mts', 'US-2022-08-17-2023-07-31-text_ts.csv'))
        df_raw = df_raw[df_raw['EventRootCode'] == self.event_root_code].copy()
        df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
        df_raw = df_raw.sort_values('Date', kind='mergesort').reset_index(drop=True)

        border1s, border2s = _split_borders(len(df_raw), self.seq_len)
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw[['NumMentions', 'NumSources', 'NumArticles']]
        self.data_raw = df_data.values

        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['Date']].iloc[border1:border2].copy()
        self.data_stamp = _build_time_stamp(df_stamp, self.timeenc, self.freq, 'Date')
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.date = df_raw['Date'].iloc[border1:border2].values
        self.text = df_raw['summary'].replace('', np.nan).fillna('No information available').iloc[border1:border2].tolist()
    def get_text(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()
        indices = np.asarray(indices)

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        return [self.text[s_begin:s_end] for s_begin, s_end in zip(s_begins, s_ends)]

    def get_date(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()
        indices = np.asarray(indices)

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        dates = np.array([self.date[s_begin:s_end] for s_begin, s_end in zip(s_begins, s_ends)])
        return dates

    def get_forecast_date(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()
        indices = np.asarray(indices)

        s_begins = indices % self.tot_len
        r_begins = s_begins + self.seq_len
        r_ends = r_begins + self.pred_len
        dates = np.array([self.date[r_begin:r_end] for r_begin, r_end in zip(r_begins, r_ends)])
        return dates

    def __len__(self):
        if self.channel_independent:
            return max(self.tot_len * self.num_features, 0)
        return max(len(self.data_x) - self.seq_len - self.pred_len + 1, 0)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if self.channel_independent:
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        else:
            seq_x = self.data_x[s_begin:s_end, :]
            seq_y = self.data_y[r_begin:r_end, :]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, index
