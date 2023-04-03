import torch
from torch.nn import BatchNorm1d
from torch.nn import LayerNorm
import tensorflow as tf
from tensorflow import keras


def torch_bn_ln():
    x = torch.tensor([[4.0, 3.0, 2.0],
                      [3.0, 3.0, 2.0],
                      [2.0, 2.0, 2.0]
                      ])

    print(BatchNorm1d(x.size()[1])(x))
    print(LayerNorm(x.size()[1:])(x))
    print("=" * 36)

    y = torch.tensor([
        [[1.0, 4.0, 7.0],
         [0.0, 2.0, 4.0]
         ],
        [[1.0, 3.0, 6.0],
         [2.0, 3.0, 1.0]
         ]
    ])
    print((BatchNorm1d(2))(y))
    print((LayerNorm(3))(y))


def keras_bn_ln():
    x = tf.constant([[4.0, 3.0, 2.0, 5.0],
                     [3.0, 3.0, 2.0, 4.0],
                     [2.0, 2.0, 2.0, 2.0]
                     ])

    print(keras.layers.BatchNormalization()(x))
    print(keras.layers.LayerNormalization()(x))
    print("=" * 36)
    mu, var = tf.nn.moments(x, 0, keepdims=True)
    r_bn = (x - mu) / tf.sqrt(var + 1E-6)
    print(r_bn)
    print("=" * 36)
    mu, var = tf.nn.moments(x, 1, keepdims=True)
    r_ln = (x - mu) / tf.sqrt(var + 1E-6)
    print(r_ln)
    print("=" * 36)

    y = tf.constant([
        [[1.0, 4.0, 7.0],
         [0.0, 2.0, 4.0]
         ],
        [[1.0, 3.0, 6.0],
         [2.0, 3.0, 1.0]
         ]
    ])
    print(keras.layers.BatchNormalization()(y))
    print(keras.layers.LayerNormalization()(y))


def main():
    keras_bn_ln()


if __name__ == '__main__':
    main()
