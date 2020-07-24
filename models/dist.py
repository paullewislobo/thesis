import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


class CategoricalProbabilityDistribution(layers.Layer):
    def __init__(self):
        super(CategoricalProbabilityDistribution, self).__init__()

    @tf.function
    def mode(self, logits):
        return tf.argmax(logits, axis=-1)

    @tf.function
    def neg_log_p(self, x, logits):
        one_hot_actions = tf.one_hot(x, logits.get_shape().as_list()[-1])
        return tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits,
            labels=tf.stop_gradient(one_hot_actions))

    @tf.function
    def entropy(self, logits):
        a_0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        exp_a_0 = tf.exp(a_0)
        z_0 = tf.reduce_sum(exp_a_0, axis=-1, keepdims=True)
        p_0 = exp_a_0 / z_0
        return tf.reduce_sum(p_0 * (tf.math.log(z_0) - a_0), axis=-1)

    @tf.function
    def sample(self, logits):
        uniform = tf.random.uniform(tf.shape(logits), dtype=logits.dtype)
        return tf.argmax(logits - tf.math.log(-tf.math.log(uniform)), axis=-1)


class DiagGaussianProbabilityDistribution(layers.Layer):
    def __init__(self, action_size):
        super(DiagGaussianProbabilityDistribution, self).__init__()
        self.log_std = tf.Variable(tf.zeros([1, action_size], dtype=tf.float32))

    @tf.function
    def mode(self, mean):
        return mean

    @tf.function
    def neg_log_p(self, x, mean):
        return 0.5 * tf.reduce_sum(tf.square((x - mean) / tf.exp(self.log_std)), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
               + tf.reduce_sum(self.log_std, axis=-1)

    @tf.function
    def entropy(self, mean):
        return tf.reduce_sum(self.log_std + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    @tf.function
    def sample(self, mean):
        return mean + tf.exp(self.log_std) * tf.random.normal(tf.shape(mean), dtype=mean.dtype)
