import tensorflow as tf
from tensorflow.keras import layers

class SobelEdgeLayer(layers.Layer):
    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, axis=-1)
        sobel = tf.image.sobel_edges(inputs)
        return tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1) + 1e-7)

class SpectralNormalization(layers.Wrapper):
    def __init__(self, layer, iteration=1, **kwargs):
        super().__init__(layer, **kwargs)
        self.iteration = iteration

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        self.u = self.add_weight(shape=(1, self.w_shape[-1]), 
                                initializer='random_normal',
                                trainable=False,
                                name='sn_u')

    def call(self, inputs):
        self._compute_weights()
        return self.layer(inputs)

    def _compute_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u = self.u
        for _ in range(self.iteration):
            v = tf.math.l2_normalize(tf.matmul(u, w_reshaped, transpose_b=True))
            u = tf.math.l2_normalize(tf.matmul(v, w_reshaped))
        sigma = tf.matmul(tf.matmul(u, w_reshaped, transpose_b=True), v, transpose_b=True)
        self.layer.kernel.assign(self.w / sigma)
        self.u.assign(u)

class EdgeAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sobel = SobelEdgeLayer()
        self.conv = layers.Conv2D(1, 1, activation='sigmoid')

    def build(self, input_shape):
        self.input_channels = input_shape[-1]

    def call(self, x):
        edges = self.sobel(x)
        att = self.conv(edges)
        att = tf.tile(att, [1, 1, 1, self.input_channels])
        return x * att
