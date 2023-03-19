import tensorflow as tf
from tensorflow import keras


class Transformer(object):
    def __init__(self, x, num_head, num_block):
        self.x = x
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


def main():
    x = [[3, 38], [20, 9], [31, 37, 38, 10], [1, 2, 3, 4, 5], [7, 8]]
    x_pad = keras.preprocessing.sequence.pad_sequences(x, maxlen=4, padding="post", truncating="post")
    # x_mask = tf.sequence_mask([len(seq) for seq in x], maxlen=4)
    embedding = keras.layers.Embedding(50, 3, input_length=5)
    x_emb = embedding(x_pad)
    # print(x_emb)
    # print("=" * 36)

    trm = Transformer(x_emb, 3, 2)
    trm.encoder()
    print(trm.encode_layers[-1])


if __name__ == '__main__':
    main()