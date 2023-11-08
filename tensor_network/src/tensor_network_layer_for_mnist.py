from typing import Any, Final

import tensorflow as tf
import tensornetwork as tn

tn.set_default_backend("tensorflow")

M: Final = 32
J: Final = 28
L: Final = 28
N: Final = 16
K: Final = 1


class TNLayer(tf.keras.layers.Layer):  # type:ignore
    def __init__(self, m: int = M, j: int = J, l: int = L, n: int = N, k: int = K) -> None:  # noqa
        super(TNLayer, self).__init__()
        # Create the variables for the layer.

        self.a_var = tf.Variable(
            tf.random.normal(shape=(m, j, k), stddev=1.0 / 32.0), name="a", trainable=True
        )
        self.c_var = tf.Variable(
            tf.random.normal(shape=(n, l, k), stddev=1.0 / 32.0), name="c", trainable=True
        )
        self.bias = tf.Variable(tf.zeros(shape=(m, n)), name="bias", trainable=True)
        self.j = j
        self.m = m
        self.n = n
        self.l = l  # noqa

    def call(self, inputs: Any) -> Any:
        # Define the contraction.
        # We break it out so we can parallelize a batch using
        # tf.vectorized_map (see below).
        def f(
            input_vec: Any, a_var: tf.Variable, c_var: tf.Variable, bias_var: tf.Variable
        ) -> tn.Tensor:
            # Reshape to a matrix instead of a vector.
            input_vec = tf.reshape(input_vec, (self.j, self.l))

            # Now we create the network.
            a = tn.Node(a_var)
            c = tn.Node(c_var)
            x_node = tn.Node(input_vec)
            a[1] ^ x_node[0]
            c[1] ^ x_node[1]
            a[2] ^ c[2]

            # The TN should now look like this
            #   |     |
            #   a --- b
            #    \   /
            #      x

            # Now we begin the contraction.
            d = a @ x_node
            result = (d @ c).tensor

            # To make the code shorter, we also could've used Ncon.
            # The above few lines of code is the same as this:
            # result = tn.ncon([x, a_var, b_var], [[1, 2], [-1, 1, 3], [-2, 2, 3]])

            # Finally, add bias. 32x32の行列ができる。32x32=1024
            return result + bias_var

        # To deal with a batch of items, we can use the tf.vectorized_map

        # function.
        # https://www.tensorflow.org/api_docs/python/tf/vectorized_map
        result = tf.vectorized_map(lambda vec: f(vec, self.a_var, self.c_var, self.bias), inputs)
        return tf.reshape(result, (-1, self.m * self.n))  # バッチの数だけの1024ベクトルができる。
