import tensorflow as tf
from tensorflow import keras


class Transformer(object):
    def __init__(self, x, num_head, num_block):
        self.x = x  # [N,L,d]
        self.num_head = num_head
        self.num_block = num_block

        self.x_shape = x.get_shape()
        self.x_len = self.x_shape[-2]
        self.x_dim = self.x_shape[-1]
        self.encode_layers = []

    def attention(self, q, k, v):
        k_t = tf.transpose(k, perm=[0, 2, 1])
        score = tf.matmul(q, k_t) / tf.sqrt(tf.constant(self.x_dim, dtype=tf.float32))
        score = keras.activations.softmax(score)

        return tf.matmul(score, v)

    def head(self, x):
        q = keras.layers.Dense(self.x_dim)(x)
        k = keras.layers.Dense(self.x_dim)(x)
        v = keras.layers.Dense(self.x_dim)(x)

        return self.attention(q, k, v)

    def multi_head(self, x, num):
        initializer = tf.keras.initializers.GlorotNormal()
        w = tf.Variable(initial_value=initializer(shape=(self.x_len, self.x_len * num)), name="w_o")
        z = tf.concat([self.head(x) for _ in range(num)], axis=1)

        return tf.matmul(w, z)

    def feed_forward(self, x):
        layer = keras.Sequential([
            keras.layers.Dense(self.x_dim, activation="gelu"),
            keras.layers.Dense(self.x_dim)
        ])

        return layer(x)

    def encode_block(self, x):
        z = self.multi_head(x, self.num_head)
        z = keras.layers.LayerNormalization()(x + z)
        e = self.feed_forward(z)

        return keras.layers.LayerNormalization()(z + e)

    def encoder(self):
        z = self.x
        print(z.get_shape())
        for _ in range(self.num_block):
            e = self.encode_block(z)
            print(e.get_shape())
            self.encode_layers.append(e)
            z = e

    def decoder(self):
        pass


class Transformer2(object):
    def __init__(self, x, num_head, num_block):
        self.x = x  # [B,L,d]
        self.num_head = num_head
        self.num_block = num_block

        self.x_shape = x.get_shape()
        self.x_len = self.x_shape[-2]
        self.x_dim = self.x_shape[-1]
        self.encode_layers = []

    def attention(self, q, k, v):
        k_t = tf.transpose(k, perm=[0, 2, 1])
        score = tf.matmul(q, k_t) / tf.sqrt(tf.constant(self.x_dim, dtype=tf.float32))
        score = keras.activations.softmax(score)

        return tf.matmul(score, v)

    def parallel_head(self, x, num, name=None):
        initializer = tf.keras.initializers.GlorotNormal()
        w = tf.Variable(initial_value=initializer(shape=(self.x_len * num, self.x_len)), name=name)
        z = tf.matmul(w, x)

        return tf.concat(tf.split(z, num, axis=1), axis=0)

    def multi_head2(self, x, num):
        q_ = self.parallel_head(x, num, name="w_q")
        k_ = self.parallel_head(x, num, name="w_q")
        v_ = self.parallel_head(x, num, name="w_q")

        o_ = self.attention(q_, k_, v_)  # [B*num,L,d]
        o = tf.concat(tf.split(o_, num, axis=0), axis=1)  # [B,num*L,d]
        print(f"o.get_shape()={o.get_shape()}")

        initializer = tf.keras.initializers.GlorotNormal()
        w_o = tf.Variable(initial_value=initializer(shape=(self.x_len, self.x_len * num)), name="w_o")

        return tf.matmul(w_o, o)

    def multi_head(self, x, num):
        initializer = tf.keras.initializers.GlorotNormal()
        w_q = tf.Variable(initial_value=initializer(shape=(self.x_len * num, self.x_len)), name="w_q")
        w_k = tf.Variable(initial_value=initializer(shape=(self.x_len * num, self.x_len)), name="w_k")
        w_v = tf.Variable(initial_value=initializer(shape=(self.x_len * num, self.x_len)), name="w_v")

        q = tf.matmul(w_q, x)  # [B,num*L,d]
        k = tf.matmul(w_k, x)
        v = tf.matmul(w_v, x)

        q_ = tf.concat(tf.split(q, num, axis=2), axis=0)  # [B*num,L,d]
        k_ = tf.concat(tf.split(k, num, axis=2), axis=0)
        v_ = tf.concat(tf.split(v, num, axis=2), axis=0)

        o_ = self.attention(q_, k_, v_)  # [B*num,L,d]
        o = tf.concat(tf.split(o_, num, axis=0), axis=2)  # [B,num*L,d]

        w_o = tf.Variable(initial_value=initializer(shape=(self.x_len, self.x_len * num)), name="w_o")

        return tf.matmul(w_o, o)

    def feed_forward(self, x):
        layer = keras.Sequential([
            keras.layers.Dense(self.x_dim, activation="gelu"),
            keras.layers.Dense(self.x_dim)
        ])

        return layer(x)

    def encode_block(self, x):
        z = self.multi_head2(x, self.num_head)
        z = keras.layers.LayerNormalization()(x + z)
        e = self.feed_forward(z)

        return keras.layers.LayerNormalization()(z + e)

    def encoder(self):
        z = self.x
        print(f"z.get_shape()={z.get_shape()}")
        for _ in range(self.num_block):
            e = self.encode_block(z)
            print(f"e.get_shape()={e.get_shape()}")
            self.encode_layers.append(e)
            z = e

    def decoder(self):
        pass


class Transformer3(object):
    def __init__(self, x, num_head, num_block):
        self.x = x  # [B,L,d]
        self.num_head = num_head
        self.num_block = num_block

        self.x_shape = x.get_shape()
        self.x_len = self.x_shape[-2]
        self.x_dim = self.x_shape[-1]
        self.encode_layers = []

    def attention(self, q, k, v):
        k_t = tf.transpose(k, perm=[0, 2, 1])
        score = tf.matmul(q, k_t) / tf.sqrt(tf.constant(self.x_dim, dtype=tf.float32))
        score = keras.activations.softmax(score)

        return tf.matmul(score, v)

    def parallel_head(self, x, num, name=None):
        initializer = keras.initializers.GlorotNormal()
        w = tf.Variable(initial_value=initializer(shape=(self.x_dim, self.x_dim * num)), name=name)
        z = tf.matmul(x, w)
        print(f"z.get_shape()={z.get_shape()}")

        return tf.concat(tf.split(z, num, axis=2), axis=0)

    def multi_head(self, x, num):
        q_ = self.parallel_head(x, num, name="w_q")
        k_ = self.parallel_head(x, num, name="w_q")
        v_ = self.parallel_head(x, num, name="w_q")

        o_ = self.attention(q_, k_, v_)  # [B*num,L,d]
        o = tf.concat(tf.split(o_, num, axis=0), axis=2)  # [B,L,num*d]
        print(f"o.get_shape()={o.get_shape()}")

        initializer = keras.initializers.GlorotNormal()
        w_o = tf.Variable(initial_value=initializer(shape=(self.x_dim * num, self.x_dim)), name="w_o")

        return tf.matmul(o, w_o)

    def feed_forward(self, x):
        layer = keras.Sequential([
            keras.layers.Dense(self.x_dim, activation="gelu"),
            keras.layers.Dense(self.x_dim)
        ])

        return layer(x)

    def encode_block(self, x):
        z = self.multi_head(x, self.num_head)
        z = keras.layers.LayerNormalization()(x + z)
        e = self.feed_forward(z)

        return keras.layers.LayerNormalization()(z + e)

    def encoder(self):
        z = self.x
        print(f"z.get_shape()={z.get_shape()}")
        for _ in range(self.num_block):
            e = self.encode_block(z)
            print(f"e.get_shape()={e.get_shape()}")
            self.encode_layers.append(e)
            z = e

    def decoder(self):
        pass


class Transformer4(object):
    def __init__(self, x, num_head, num_block, mask=None):
        self.x = x  # [B,L,d]
        self.mask = mask  # [B,L]
        self.num_head = num_head
        self.num_block = num_block

        self.x_shape = x.get_shape()
        self.x_len = self.x_shape[-2]
        self.x_dim = self.x_shape[-1]
        self.encode_layers = []

    @classmethod
    def make_mask(cls, x_mask):
        '''
        :param x_mask: [B,L]
        :return: mask [B,L,L]
        '''
        B, L = x_mask.get_shape()
        x_mask = tf.cast(tf.reshape(x_mask, [B, 1, L]), tf.float32)
        b_ones = tf.ones(shape=[B, L, 1], dtype=tf.float32)

        return b_ones * x_mask

    def attention(self, q, k, v, mask=None):
        k_t = tf.transpose(k, perm=[0, 2, 1])
        score = tf.matmul(q, k_t) / tf.sqrt(tf.constant(self.x_dim, dtype=tf.float32))

        if mask is not None:
            mask = self.make_mask(mask)
            score += ((1.0 - tf.cast(mask, tf.float32)) * (-1E6))

        score = keras.activations.softmax(score)  # [B,L,L]

        return tf.matmul(score, v)

    def parallel_head(self, x, num, name=None):
        initializer = keras.initializers.GlorotNormal()
        w = tf.Variable(initial_value=initializer(shape=(self.x_dim, self.x_dim * num)), name=name)
        z = tf.matmul(x, w)
        print(f"z.get_shape()={z.get_shape()}")

        return tf.concat(tf.split(z, num, axis=2), axis=0)

    def multi_head(self, x, num):
        q_ = self.parallel_head(x, num, name="w_q")
        k_ = self.parallel_head(x, num, name="w_q")
        v_ = self.parallel_head(x, num, name="w_q")

        o_ = self.attention(q_, k_, v_)  # [B*num,L,d]
        o = tf.concat(tf.split(o_, num, axis=0), axis=2)  # [B,L,num*d]
        print(f"o.get_shape()={o.get_shape()}")

        initializer = keras.initializers.GlorotNormal()
        w_o = tf.Variable(initial_value=initializer(shape=(self.x_dim * num, self.x_dim)), name="w_o")

        return tf.matmul(o, w_o)

    def feed_forward(self, x):
        layer = keras.Sequential([
            keras.layers.Dense(self.x_dim, activation="gelu"),
            keras.layers.Dense(self.x_dim)
        ])

        return layer(x)

    def encode_block(self, x):
        z = self.multi_head(x, self.num_head)
        z = keras.layers.LayerNormalization()(x + z)
        e = self.feed_forward(z)

        return keras.layers.LayerNormalization()(z + e)

    def encoder(self):
        z = self.x
        print(f"z.get_shape()={z.get_shape()}")
        for _ in range(self.num_block):
            e = self.encode_block(z)
            print(f"e.get_shape()={e.get_shape()}")
            self.encode_layers.append(e)
            z = e

    def decoder(self):
        pass


class Encoder(object):
    def __init__(self, x, num_head, num_block, mask=None):
        self.x = x  # [B,L,d]
        self.mask = mask  # [B,L]
        self.num_head = num_head
        self.num_block = num_block

        self.x_shape = x.get_shape()
        self.x_len = self.x_shape[-2]
        self.x_dim = self.x_shape[-1]
        self.encode_layers = []

    @classmethod
    def make_mask(cls, x_mask):
        '''
        :param x_mask: [B,L]
        :return: mask [B,L,L]
        '''
        B, L = x_mask.get_shape()
        x_mask = tf.cast(tf.reshape(x_mask, [B, 1, L]), tf.float32)
        b_ones = tf.ones(shape=[B, L, 1], dtype=tf.float32)

        return b_ones * x_mask

    def attention(self, q, k, v, mask=None):
        k_t = tf.transpose(k, perm=[0, 2, 1])
        score = tf.matmul(q, k_t) / tf.sqrt(tf.constant(self.x_dim, dtype=tf.float32))

        if mask is not None:
            mask = self.make_mask(mask)
            score += ((1.0 - tf.cast(mask, tf.float32)) * (-1E6))

        score = keras.activations.softmax(score)  # [B,L,L]

        return tf.matmul(score, v)

    def parallel_head(self, x, num, name=None):
        initializer = keras.initializers.GlorotNormal()
        w = tf.Variable(initial_value=initializer(shape=(self.x_dim, self.x_dim * num)), name=name)
        z = tf.matmul(x, w)
        print(f"z.get_shape()={z.get_shape()}")

        return tf.concat(tf.split(z, num, axis=2), axis=0)

    def multi_head(self, x, num):
        q_ = self.parallel_head(x, num, name="w_q")
        k_ = self.parallel_head(x, num, name="w_q")
        v_ = self.parallel_head(x, num, name="w_q")

        o_ = self.attention(q_, k_, v_)  # [B*num,L,d]
        o = tf.concat(tf.split(o_, num, axis=0), axis=2)  # [B,L,num*d]
        print(f"o.get_shape()={o.get_shape()}")

        initializer = keras.initializers.GlorotNormal()
        w_o = tf.Variable(initial_value=initializer(shape=(self.x_dim * num, self.x_dim)), name="w_o")

        return tf.matmul(o, w_o)

    def feed_forward(self, x):
        layer = keras.Sequential([
            keras.layers.Dense(self.x_dim, activation="gelu"),
            keras.layers.Dense(self.x_dim)
        ])

        return layer(x)

    def encode_block(self, x):
        z = self.multi_head(x, self.num_head)
        z = keras.layers.LayerNormalization()(x + z)
        e = self.feed_forward(z)

        return keras.layers.LayerNormalization()(z + e)

    def encoder(self):
        z = self.x
        print(f"z.get_shape()={z.get_shape()}")
        for _ in range(self.num_block):
            e = self.encode_block(z)
            print(f"e.get_shape()={e.get_shape()}")
            self.encode_layers.append(e)
            z = e


class Decoder(object):
    def __init__(self, enc, x, num_head, num_block, pad_mask=None, look_ahead_mask=None):
        self.enc = enc
        self.x = x  # [B,L,d]
        self.pad_mask = pad_mask  # [B,L]
        self.look_ahead_mask = look_ahead_mask
        self.num_head = num_head
        self.num_block = num_block

        self.x_shape = x.get_shape()
        self.x_len = self.x_shape[-2]
        self.x_dim = self.x_shape[-1]
        self.decode_layers = []

    @classmethod
    def create_pad_mask(cls, x_mask):
        '''
        :param x_mask: [B,L]
        :return: mask [B,L,L]
        '''
        B, L = x_mask.get_shape()
        x_mask = tf.cast(tf.reshape(x_mask, [B, 1, L]), tf.float32)
        b_ones = tf.ones(shape=[B, L, 1], dtype=tf.float32)

        return b_ones * x_mask

    @classmethod
    def create_look_ahead_mask(cls, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    def make_mask(self, pad_mask=None, look_ahead_mask=None):
        if pad_mask is not None and look_ahead_mask is not None:
            p_mask = self.create_pad_mask(pad_mask)
            l_mask = self.create_pad_mask(self.x_len)
            return tf.minimum(p_mask, l_mask)
        elif pad_mask is not None:
            return self.create_pad_mask(pad_mask)
        elif look_ahead_mask is not None:
            return self.create_pad_mask(self.x_len)
        else:
            return None

    def attention(self, q, k, v, pad_mask=None, look_ahead_mask=None):
        k_t = tf.transpose(k, perm=[0, 2, 1])
        score = tf.matmul(q, k_t) / tf.sqrt(tf.constant(self.x_dim, dtype=tf.float32))

        mask = self.make_mask(pad_mask, look_ahead_mask)  # 0->mask, 1->unmask
        if mask is not None:
            score += ((1.0 - tf.cast(pad_mask, tf.float32)) * (-1E6))

        score = keras.activations.softmax(score)  # [B,L,L]

        return tf.matmul(score, v)

    def parallel_head(self, x, num, name=None):
        initializer = keras.initializers.GlorotNormal()
        w = tf.Variable(initial_value=initializer(shape=(self.x_dim, self.x_dim * num)), name=name)
        z = tf.matmul(x, w)
        print(f"z.get_shape()={z.get_shape()}")

        return tf.concat(tf.split(z, num, axis=2), axis=0)

    def multi_head(self, x, num):
        q_ = self.parallel_head(x, num, name="w_q")
        k_ = self.parallel_head(x, num, name="w_q")
        v_ = self.parallel_head(x, num, name="w_q")

        o_ = self.attention(q_, k_, v_)  # [B*num,L,d]
        o = tf.concat(tf.split(o_, num, axis=0), axis=2)  # [B,L,num*d]
        print(f"o.get_shape()={o.get_shape()}")

        initializer = keras.initializers.GlorotNormal()
        w_o = tf.Variable(initial_value=initializer(shape=(self.x_dim * num, self.x_dim)), name="w_o")

        return tf.matmul(o, w_o)

    def cross_multi_head(self, x, num):
        q_ = self.parallel_head(x, num, name="w_q")
        k_ = self.parallel_head(self.enc, num, name="w_q")
        v_ = self.parallel_head(self.enc, num, name="w_q")

        o_ = self.attention(q_, k_, v_)  # [B*num,L,d]
        o = tf.concat(tf.split(o_, num, axis=0), axis=2)  # [B,L,num*d]
        print(f"o.get_shape()={o.get_shape()}")

        initializer = keras.initializers.GlorotNormal()
        w_o = tf.Variable(initial_value=initializer(shape=(self.x_dim * num, self.x_dim)), name="w_o")

        return tf.matmul(o, w_o)

    def feed_forward(self, x):
        layer = keras.Sequential([
            keras.layers.Dense(self.x_dim, activation="gelu"),
            keras.layers.Dense(self.x_dim)
        ])

        return layer(x)

    def decode_block(self, x):
        z = self.multi_head(x, self.num_head)
        z = keras.layers.LayerNormalization()(x + z)
        u = self.cross_multi_head(z, self.num_head)
        z = keras.layers.LayerNormalization()(z + u)
        e = self.feed_forward(z)

        return keras.layers.LayerNormalization()(z + e)

    def decoder(self):
        z = self.x
        print(f"z.get_shape()={z.get_shape()}")
        for _ in range(self.num_block):
            e = self.decode_block(z)
            print(f"e.get_shape()={e.get_shape()}")
            self.decode_layers.append(e)
            z = e


class Transformer5(object):
    def __init__(self, x, y, num_head, num_enc, num_dec, pad_mask=None, look_mask=None):
        self.batch_size, self.x_len, self.x_dim = x.get_shape()
        self.x = x
        self.y = y
        self.num_head = num_head
        self.num_enc = num_enc
        self.num_dec = num_dec
        self.pad_mask = pad_mask
        self.look_mask = look_mask

    def attention(self, q, k, v, mask=None):
        score = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.constant(self.x_dim, dtype=tf.float32))
        if mask is not None:
            score += (tf.cast(mask, dtype=tf.float32) * (-1E6))

        return tf.matmul(score, v)

    def para_head(self, x, num_head):
        initializers = keras.initializers.GlorotNormal()
        w = tf.Variable(initial_value=initializers(self.x_dim, self.x_dim * num_head))
        o_ = tf.matmul(x, w)
        return tf.concat(tf.split(o_, num_head, axis=2), axis=0)

    def multi_head(self, x, num_head, mask):
        q_ = self.para_head(x, num_head)
        k_ = self.para_head(x, num_head)
        v_ = self.para_head(x, num_head)

        o = self.attention(q_, k_, v_, mask)
        o_ = tf.concat(tf.split(o, num_head, axis=0), axis=2)

        initializers = keras.initializers.GlorotNormal()
        w_o = tf.Variable(initial_value=initializers(self.x_dim * num_head, self.x_dim))

        return tf.matmul(o_, w_o)

    def ffn(self, x):
        layer = keras.Sequential([
            keras.layers.Dense(self.x_dim, activation="gelu"),
            keras.layers.Dense(self.x_dim)
        ])

        return layer(x)

    @classmethod
    def ln(cls, x):
        return keras.layers.LayerNormalization()(x)

    def encoder_block(self, x, num_head, mask):
        z = self.multi_head(x, num_head, mask)
        u = self.ln(x + z)
        z = self.ffn(u)
        return self.ln(u + z)

    def cross_multi_head(self, y, enc, num_head, mask):
        q_ = self.para_head(y, num_head)
        k_ = self.para_head(enc, num_head)
        v_ = self.para_head(enc, num_head)

        o_ = self.attention(q_, k_, v_, mask)

        initializers = keras.initializers.GlorotNormal()
        w_o = tf.Variable(initial_value=initializers(self.x_dim * num_head, self.x_dim))

        return tf.matmul(o_, w_o)

    def decoder_block(self, y, enc, num_head, mask):
        z = self.multi_head(y, num_head, mask)
        u = self.ln(y + z)
        z = self.cross_multi_head(u, enc, num_head, mask)
        u = self.ffn(z)
        return self.ln(z + u)


def main():
    x = [[3, 38], [20, 9], [31, 37, 38, 10], [1, 2, 3, 4, 5], [7, 8]]
    x_pad = keras.preprocessing.sequence.pad_sequences(x, maxlen=4, padding="post", truncating="post")
    x_mask = tf.sequence_mask([len(seq) for seq in x], maxlen=4)
    embedding = keras.layers.Embedding(50, 3, input_length=5)
    x_emb = embedding(x_pad)
    # print(x_emb)
    # print("=" * 36)

    trm = Transformer4(x_emb, 6, 2, x_mask)
    trm.encoder()
    print(trm.encode_layers[-1])


if __name__ == '__main__':
    main()
