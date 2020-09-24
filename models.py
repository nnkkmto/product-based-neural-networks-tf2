"""
ちょっと論文と相違あるが、
https://github.com/Atomu2014/product-nets
をベースに実装する
"""
import itertools
from collections import OrderedDict
import tensorflow as tf


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, features_info, emb_dim, name_prefix=''):
        """
        sequence対応のembedding layer
        """
        super(EmbeddingLayer, self).__init__()
        self.features_info = features_info
        self.feature_to_embedding_layer = OrderedDict()
        for feature in features_info:
            initializer = tf.keras.initializers.RandomNormal(stddev=0.01, seed=None)
            if feature['is_sequence']:
                # sequenceのembedding
                self.feature_to_embedding_layer[feature['name']] = tf.keras.layers.Embedding(
                    feature['dim'],
                    emb_dim,
                    mask_zero=True,
                    name=f"embedding_{name_prefix}{feature['name']}",
                    embeddings_initializer=initializer)
            else:
                self.feature_to_embedding_layer[feature['name']] = tf.keras.layers.Embedding(
                    feature['dim'],
                    emb_dim,
                    name=f"embedding_{name_prefix}{feature['name']}",
                    embeddings_initializer=initializer)

    def concatenate_embeddings(self, embeddings, name_prefix=''):
        if len(embeddings) >= 2:
            embeddings = tf.keras.layers.Concatenate(axis=1, name=name_prefix+'embeddings_concat')(embeddings)
        else:
            embeddings = embeddings[0]
        return embeddings

    def call(self, inputs):
        embeddings = []
        for feature_input, feature in zip(inputs, self.features_info):
            # embeddingの作成
            embedding = self.feature_to_embedding_layer[feature['name']](feature_input)
            if feature['is_sequence']:
                # sequenceの場合はaverage pooling
                embedding = tf.math.reduce_mean(embedding, axis=1, keepdims=True)
            embeddings.append(embedding)

        # concatenate
        embeddings = self.concatenate_embeddings(embeddings)
        return embeddings


class InnerProductLayer(tf.keras.layers.Layer):
    def __init__(self, emb_dim, pair_num):
        super(InnerProductLayer, self).__init__()
        self.inner_weights = tf.Variable(tf.random.normal([pair_num, emb_dim],0.0,0.01))

    def call(self, p, q):
        weights = tf.expand_dims(self.inner_weights, 0)  # (1, pair_num, emb_dim)
        lp = tf.reduce_sum(p * q * weights, -1)  # (batch_size, pair_num)
        return lp


class OuterProductLayer(tf.keras.layers.Layer):
    def __init__(self, emb_dim, pair_num):
        super(InnerProductLayer, self).__init__()
        self.outer_weights = tf.Variable(tf.random.normal([emb_dim, pair_num, emb_dim],0.0,0.01))

    def call(self, p, q):
        p = tf.expand_dims(p, 1)  # (batch_size, 1, pair_num, emb_dim)
        p = tf.multiply(p, self.outer_weights)  # (batch_size, emb_dim, pair_num, emb_dim)
        p = tf.reduce_sum(p, -1)  # (batch_size, emb_dim, pair_num)
        p = tf.transpose(p, [0, 2, 1])  # (batch_size, pair_num, emb_dim)
        lp = tf.multiply(p, q)  # (batch_size, pair_num, emb_dim)
        lp = tf.reduce_sum(lp, -1)  # (batch_size, pair_num)
        return lp


class ProductLayer(tf.keras.layers.Layer):
    def __init__(self, features_info, emb_dim, implementation_type='inner'):
        super(ProductLayer, self).__init__()
        self.implementation_type = implementation_type
        self.field_index_pairs = list(itertools.combinations(range(len(features_info)), 2))
        pair_num = len(self.field_index_pairs)
        self.field_index_row = [combination[0] for combination in self.field_index_pairs]
        self.field_index_col = [combination[1] for combination in self.field_index_pairs]

        if implementation_type == 'inner':
            self.layer = InnerProductLayer(emb_dim, pair_num)
        elif implementation_type == 'outer':
            self.layer = OuterProductLayer(emb_dim, pair_num)
        else:
            # 指定なければ両方のconcatenateを返す
            self.inner_layer = InnerProductLayer(emb_dim, pair_num)
            self.outer_layer = OuterProductLayer(emb_dim, pair_num)

    def call(self, embeddings):
        p = tf.gather(tf.transpose(embeddings, [1, 0, 2]), self.field_index_row)  # (pair_num, batch_size, emb_dim)
        p = tf.transpose(p, [1, 0, 2])  # (batch_size, pair_num, emb_dim)

        q = tf.gather(tf.transpose(embeddings, [1, 0, 2]), self.field_index_col)
        q = tf.transpose(q, [1, 0, 2])

        if self.implementation_type in ['inner', 'outer']:
            lp = self.layer(p, q)
        else:
            lp_inner = self.inner_layer(p, q)
            lp_outer = self.outer_layer(p, q)
            lp = tf.keras.layers.Concatenate(axis=1, name='lp_concat')([lp_inner, lp_outer])

        return lp


class PNN(tf.keras.Model):
    def __init__(self, features_info, emb_dim=20, deep_layer_size=30, dropout_rate=0.3):
        super(PNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.embedding_layer = EmbeddingLayer(features_info, emb_dim)
        self.product_layer = ProductLayer(features_info, emb_dim)
        self.d1 = tf.keras.layers.Dense(deep_layer_size, activation='relu')
        self.d2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        embeddings = self.embedding_layer(inputs)  # (batch_size, field_num, emb_dim)
        lz = tf.keras.layers.Flatten()(embeddings)  # (batch_size, field_num*emb_dim)
        lp = self.product_layer(embeddings)
        output = tf.keras.layers.Concatenate(axis=1, name='lp_concat')([lz, lp])

        # deep layer
        output = self.d1(output)
        output = tf.keras.layers.Dropout(self.dropout_rate)(output)
        output = self.d2(output)

        return output
