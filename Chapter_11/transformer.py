import tensorflow as tf

from keras.layers import Dense, LayerNormalization, Layer, Dropout


class MultiHeadAttention(Layer):
    def __init__(self, n_head, model_dim, drop_rate, **kwargs):
        super().__init__(**kwargs)
        self.head_dim = model_dim // n_head
        self.n_head = n_head
        self.model_dim = model_dim
        self.drop_rate = drop_rate
        self.w_q = Dense(model_dim)
        self.w_k = Dense(model_dim)
        self.w_v = Dense(model_dim)     # [n, step, h*h_dim]
        self.dense = Dense(model_dim)
        self.dropout = Dropout(rate=drop_rate)

    def _split_heads(self, x):
        x = tf.reshape(x, (x.shape[0], x.shape[1], self.n_head, self.head_dim))  # [n, step, h, h_dim]
        return tf.transpose(x, perm=[0, 2, 1, 3])  # [n, h, step, h_dim]

    def _attention(self, q, k, v, mask=None):
        score = tf.matmul(q, k, transpose_b=True) / (self.head_dim + 1e-8)  # [n, h_dim, q_step, step]
        if mask is not None:
            score += mask * -1e9
        attention = tf.nn.softmax(score, axis=-1)  # [n, h, q_step, step]
        attention = tf.matmul(attention, v)  # [n, h, q_step, step] @ [n, h, step, dv] = [n, h, q_step, dv]
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # [n, q_step, h, dv]
        attention = tf.reshape(attention, (attention.shape[0], attention.shape[1], -1))  # [n, q_step, h*dv]
        return attention

    def __call__(self, q, k, v, mask, training):
        _q = self.w_q(q)  # [n, q_step, h*h_dim]
        _k, _v = self.w_k(k), self.w_v(v)  # [n, step, h*h_dim]
        _q = self.split_heads(_q)  # [n, h, q_step, h_dim]
        _k, _v = self.split_heads(_k), self.split_heads(_v)  # [n, h, step, h_dim]
        attention = self._attention(_q, _k, _v, mask)  # [n, q_step, h*dv]
        o = self.dense(attention)  # [n, step, dim]
        o = self.dropout(o, training=training)
        return o


class MLP(Layer):
    def __init__(self, model_dim):
        super().__init__()
        self.dense_1 = Dense(model_dim * 4, activation='relu')
        self.dense_2 = Dense(model_dim)

    def __call__(self, x):
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x         # [n, step, dim]


class EncodeLayer(Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.ln = [LayerNormalization(axis=-1) for _ in range(2)]  # only norm z-dim
        self.multi_head_attention = MultiHeadAttention(n_head, model_dim, drop_rate)
        self.mlp = MLP(model_dim)
        self.drop = Dropout(drop_rate)

    def __call__(self, xz, training, mask):
        attn = self.multi_head_attention.call(xz, xz, xz, mask, training)       # [n, step, dim]
        o1 = self.ln[0](attn + xz)
        ffn = self.drop(self.mlp.call(o1), training)
        o = self.ln[1](ffn + o1)         # [n, step, dim]
        return o


class Encoder(Layer):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        super().__init__()
        self.ls = [EncodeLayer(n_head, model_dim, drop_rate) for _ in range(n_layer)]

    def __call__(self, xz, training, mask):
        for l in self.ls:
            xz = l.call(xz, training, mask)
        return xz  # [n, step, dim]

