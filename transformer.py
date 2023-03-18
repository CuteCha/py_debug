import tensorflow as tf


def linear(x, name):
    shape = x.get_shape()
    l = shape[1]
    initializer = tf.keras.initializers.GlorotNormal()
    w = tf.Variable(initial_value=initializer(shape=(l, l)), name=name)

    return tf.matmul(w, x)


def attention(q, k, v):
    d = q.get_shape()[-1]
    score = tf.matmul(q, tf.transpose(k, [0, 2, 1])) / tf.math.sqrt(d)
    score = tf.nn.softmax(score)

    return tf.matmul(score, v)


def head(x, i):
    q = linear(x, f"w_q_{i}")
    k = linear(x, f"w_k_{i}")
    v = linear(x, f"w_v_{i}")

    return attention(q, k, v)


def multi_head(x, n):
    shape = x.get_shape()
    l = shape[1]
    initializer = tf.keras.initializers.GlorotNormal()
    w = tf.Variable(initial_value=initializer(shape=(l, l * n)), name="w_o")
    z = tf.concat([head(x, i) for i in range(n)], axis=1)

    return tf.matmul(w, z)


def feed_forward(x):
    d = x.get_shape()[-1]
    layer = tf.keras.Sequential([
        tf.keras.layers.Dense(d, activation="gelu"),
        tf.keras.layers.Dense(d)
    ])

    return layer(x)


def encode_block(x):
    z = multi_head(x, 8)
    z = tf.keras.layers.LayerNormalization()(x + z)
    e = feed_forward(z)

    return tf.keras.layers.LayerNormalization()(z + e)


def encode(x, num):
    layers = []
    z = x
    for i in range(num):
        e = encode_block(z)
        layers.append(z)
        z = e

    return layers
