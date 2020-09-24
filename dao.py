import abc
from functools import reduce

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

from lib.encoders import OrdinalEncoder


class LearningDao(abc.ABC):
    def __init__(self, seq_max_len, split_train_rate=0.6):
        self.seq_max_len = seq_max_len
        self.split_train_rate = split_train_rate
        self.sequence_columns = []
        self.columns = []
        self.label_column = ''
        self.df = None
        self.training_indices = None
        self.test_indices = None
        self.feature_to_encoder = {}
        self.features_info = None
        self.X_train = []
        self.X_test = []
        self.y_train = None
        self.y_test = None

    def build(self):
        self.load_df()
        self.set_split_indices(self.split_train_rate)
        self.build_encoders()
        self.build_features_info()
        self.build_dataset()

    def _convert_onehot_to_category(self, df, id_col, one_hot_columns, category_col='category'):
        df_concat = pd.DataFrame(columns=[id_col, category_col])
        for col in one_hot_columns:
            # 値が1以上のもののみ残す
            df_each = df[df[col] >= 1][[id_col, col]]
            # 値をカテゴリー値に置き換える
            df_each[col] = col

            df_each.columns = [id_col, category_col]
            df_concat = pd.concat([df_concat, df_each], axis=0)
        # 重複削除
        df_concat = df_concat.drop_duplicates().reset_index(drop=True)

        return df_concat

    def _build_vocabulary(self, col):
        if col in self.sequence_columns:
            vocabulary = self.df[col].values[self.training_indices]
            vocabulary = reduce(lambda a, b: a + b, vocabulary)
            vocabulary = list(set(vocabulary))
        else:
            vocabulary = list(set(self.df[col].values[self.training_indices].tolist()))
        return vocabulary

    def set_split_indices(self, split_train_rate):
        split_indices = np.random.permutation(len(self.df.index))
        train_num = int(np.floor(len(self.df.index) * split_train_rate))
        self.training_indices = split_indices[:train_num]
        self.test_indices = split_indices[train_num:]

    def build_encoders(self):
        for col in self.columns:
            vocabulary = self._build_vocabulary(col)
            # sequenceの場合はpadding行うため1-item_numでencode
            first_index = 1 if col in self.sequence_columns else 0
            encoder = OrdinalEncoder(first_index=first_index)
            encoder.fit(vocabulary)
            self.feature_to_encoder[col] = encoder

    def build_features_info(self):
        features_info = []
        for col in self.columns:
            feature = {}
            feature['name'] = col
            feature['is_sequence'] = True if col in self.sequence_columns else False
            # 不明値にはmax_value+1を割り当てる
            feature['dim'] = self.feature_to_encoder[col].get_max_value() + 1
            features_info.append(feature)
        self.features_info = features_info

    def build_dataset(self):
        for feature in self.features_info:
            values = self.df[feature['name']].values
            encoder = self.feature_to_encoder[feature['name']]
            if feature['is_sequence']:
                # encode
                values = np.asarray([encoder.transform(value) for value in values])
                # padding
                values = pad_sequences(values, maxlen=self.seq_max_len, padding='post', truncating='post')
            else:
                values = encoder.transform(values)
            # split
            self.X_train.append(values[self.training_indices])
            self.X_test.append(values[self.test_indices])
        y = np.ravel(self.df[[self.label_column]].values)
        self.y_train = y[self.training_indices]
        self.y_test = y[self.test_indices]

    def fetch_dataset(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    @abc.abstractmethod
    def load_df(self):
        pass


class MovieLens100kDao(LearningDao):
    def __init__(self, seq_max_len=5, split_train_rate=0.6):
        super(MovieLens100kDao, self).__init__(seq_max_len, split_train_rate)
        self.sequence_columns = ['genre', 'movie_title']
        self.columns = ['user_id', 'movie_id', 'age', 'gender', 'occupation', 'zip_code', 'movie_title'
            , 'release_date', 'genre']
        # user*itemのinteractionのみで学習したい時用
        # self.sequence_columns = []
        # self.columns = ['user_id', 'movie_id']
        self.label_column = 'rating'

    def load_df(self):
        # read rating data
        data_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
        df = pd.read_csv('data/original/ml-100k/u.data', sep='\t', header=None)
        df.columns = data_cols

        # read user data
        user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
        df_user = pd.read_csv('data/original/ml-100k/u.user', sep='|', header=None)
        df_user.columns = user_cols

        # read item and genre data
        item_cols = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'imdb_url']
        genre_cols = ['unknown', 'action', 'adventure', 'animation', 'children', 'comedy', 'crime', 'documentary',
                      'drama',
                      'fantasy', 'film_noir', 'horror', 'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war',
                      'western']
        df_item_genre = pd.read_csv('data/original/ml-100k/u.item', sep='|', header=None, encoding='latin-1')
        df_item_genre.columns = item_cols + genre_cols
        df_item = df_item_genre[item_cols]
        df_genre = df_item_genre[['movie_id'] + genre_cols]

        # leave columns to use
        use_data_cols = ['user_id', 'movie_id', 'rating']
        df = df[use_data_cols]
        use_user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
        df_user = df_user[use_user_cols]
        use_item_cols = ['movie_id', 'movie_title', 'release_date']
        df_item = df_item[use_item_cols]
        use_genre_cols = ['action', 'adventure', 'animation', 'children', 'comedy', 'crime', 'documentary', 'drama',
                          'fantasy', 'film_noir', 'horror', 'musical', 'mystery', 'romance', 'sci_fi', 'thriller',
                          'war',
                          'western']
        df_genre = df_genre[['movie_id'] + use_genre_cols]

        # 各前処理

        # one-hotをカテゴリー値に変換
        df_genre = self._convert_onehot_to_category(df_genre, 'movie_id', use_genre_cols, category_col='genre')
        # 複数ジャンルがひもづくため、sequenceとして扱う
        df_genre = df_genre.groupby('movie_id').agg(list).reset_index()

        # movie_titleから年代を削除
        df_item['movie_title'] = df_item['movie_title'].replace('\([0-9]{4}\)', '', regex=True)
        df_item['movie_title'] = df_item['movie_title'].replace(',', '')
        df_item['movie_title'] = df_item['movie_title'].str.split()
        # release_dateはmonth-yearに変換
        df_item['release_date'] = df_item['release_date'].replace('^[0-9]{2}-', '', regex=True)

        # 4以上をpositive, それ以外をnegativeに変換
        df['rating'] = np.where(df['rating'] >= 4, 1, 0)

        # merge
        df = pd.merge(df, df_item, on='movie_id')
        df = pd.merge(df, df_genre, on='movie_id')
        df = pd.merge(df, df_user, on='user_id')

        self.df = df

