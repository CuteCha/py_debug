import tensorflow as tf


def parallel_head(x, num, name=None):
    B, L, d = x.get_shape().as_list()
    initializer = tf.keras.initializers.GlorotNormal()
    w = tf.Variable(initial_value=initializer(shape=(L * num, L)), name=name)
    z = tf.matmul(w, x)
    print(z.get_shape())

    return tf.concat(tf.split(z, num, axis=1), axis=0)


def debug():
    x = tf.constant([[[11, 13, 12, 0], [14, 15, 16, 0], [17, 18, 19, 0]],
                     [[1, 3, 2, 0], [4, 5, 6, 0], [7, 8, 9, 0]]],
                    dtype=tf.float32)
    B, L, d = x.get_shape().as_list()
    n = 5
    print(B, L, d)
    z_ = parallel_head(x, n)
    print(z_.get_shape())
    z = tf.concat(tf.split(z_, n, axis=0), axis=1)
    print(z.get_shape())
    initializer = tf.keras.initializers.GlorotNormal()
    w_o = tf.Variable(initial_value=initializer(shape=(L, L * n)), name="w_o")
    o = tf.matmul(w_o, z)
    print(o.get_shape())


def debug02():
    x = tf.constant([[[11, 13, 12], [14, 15, 16], [17, 18, 19], [110, 111, 112]],
                     [[1, 3, 2], [4, 5, 6], [7, 8, 9], [10, 11, 12]]],
                    dtype=tf.float32)
    y = tf.concat(tf.split(x, 2, axis=1), axis=0)
    z = tf.concat(tf.split(y, 2, axis=0), axis=1)
    print(x.get_shape())
    print(y.get_shape())
    print(z.get_shape())
    print("=" * 36)
    print(x[0])
    print(z[0])
    print("=" * 36)
    print(y)


def debug03():
    x = tf.constant([[[11, 13, 12, 14, 15, 16], [17, 18, 19, 120, 121, 122]],
                     [[1, 3, 2, 4, 5, 6], [7, 8, 9, 10, 11, 12]]],
                    dtype=tf.float32)
    y = tf.concat(tf.split(x, 2, axis=2), axis=0)
    z = tf.concat(tf.split(y, 2, axis=0), axis=2)
    print(x.get_shape())
    print(y.get_shape())
    print(z.get_shape())
    print("=" * 36)
    print(x[0])
    print(z[0])
    print("=" * 36)
    print(y)


def debug04():
    x_mask = tf.sequence_mask([1, 6, 2], 5)
    print(x_mask)
    print(1.0 - tf.cast(x_mask, dtype=tf.float32))


if __name__ == '__main__':
    debug04()
