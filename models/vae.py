# implementation taken from https://medium.com/@wuga/generate-anime-character-with-variational-auto-encoder-81e3134d1439

import tensorflow as tf
from tensorflow.keras import layers, models


class Encoder(layers.Layer):
    def __init__(self, z_size=32, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.z_size = z_size
        ef_dim = 32
        kernel_initializer = tf.random_normal_initializer(stddev=0.02)
        gamma_initializer = tf.random_normal_initializer(1., 0.02)

        self.conv_1 = layers.Conv2D(ef_dim, 5, 2, activation=None, padding='SAME', kernel_initializer=kernel_initializer)
        self.conv_2 = layers.Conv2D(ef_dim * 2, 5, 2, activation=None, padding='SAME', kernel_initializer=kernel_initializer)
        self.conv_3 = layers.Conv2D(ef_dim * 4, 5, 2, activation=None, padding='SAME', kernel_initializer=kernel_initializer)
        self.conv_4 = layers.Conv2D(ef_dim * 8, 5, 2, activation=None, padding='SAME', kernel_initializer=kernel_initializer)

        self.bn_1 = layers.BatchNormalization(trainable=True, gamma_initializer=gamma_initializer)
        self.bn_2 = layers.BatchNormalization(trainable=True, gamma_initializer=gamma_initializer)
        self.bn_3 = layers.BatchNormalization(trainable=True, gamma_initializer=gamma_initializer)
        self.bn_4 = layers.BatchNormalization(trainable=True, gamma_initializer=gamma_initializer)
        self.bn_5 = layers.BatchNormalization(trainable=True, gamma_initializer=gamma_initializer)

        self.flatten = layers.Flatten()
        self.dense = layers.Dense(self.z_size * 2)

        self.activation = layers.Activation(activation=tf.nn.relu)

    @tf.function
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.activation(self.bn_1(x))
        x = self.conv_2(x)
        x = self.activation(self.bn_2(x))
        x = self.conv_3(x)
        x = self.activation(self.bn_3(x))
        x = self.conv_4(x)
        x = self.activation(self.bn_4(x))
        x = self.flatten(x)
        x = self.dense(x)
        x = self.bn_5(x)
        mu = x[:, :self.z_size]
        log_var = x[:, self.z_size:]
        std_dev = tf.sqrt(tf.exp(log_var))
        epsilon = tf.random.normal(shape=std_dev.shape)
        return mu, log_var, mu + std_dev * epsilon


class Decoder(layers.Layer):
    def __init__(self, image_size=64, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        gf_dim = 32
        c_dim = 3
        s2, s4, s8, s16 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(
            image_size / 16)  # 32,16,8,4
        kernel_initializer = tf.random_normal_initializer(stddev=0.02)
        gamma_initializer = tf.random_normal_initializer(1., 0.02)

        self.dense_1 = layers.Dense(gf_dim * 4 * s8 * s8, kernel_initializer=kernel_initializer, activation=tf.identity)
        self.reshape_layer = layers.Reshape(target_shape=[s8, s8, gf_dim * 4])
        self.bn_1 = layers.BatchNormalization(trainable=True, gamma_initializer=gamma_initializer)

        self.deconv_1 = layers.Conv2DTranspose(gf_dim * 4, (5, 5), strides=(2, 2),
                          padding='SAME', kernel_initializer=kernel_initializer)
        self.bn_2 = layers.BatchNormalization(trainable=True,
                                gamma_initializer=gamma_initializer)
        self.deconv_2 = layers.Conv2DTranspose(gf_dim * 2, (5, 5), strides=(2, 2),
                          padding='SAME', kernel_initializer=kernel_initializer)
        self.bn_3 = layers.BatchNormalization(trainable=True,
                                gamma_initializer=gamma_initializer)
        self.deconv_3 = layers.Conv2DTranspose(gf_dim // 2, (5, 5), strides=(2, 2),
                          padding='SAME', kernel_initializer=kernel_initializer)
        self.bn_4 = layers.BatchNormalization(trainable=True,
                                gamma_initializer=gamma_initializer)
        self.deconv_4 = layers.Conv2DTranspose(c_dim, (5, 5), strides=(1, 1),
                          padding='SAME', kernel_initializer=kernel_initializer)

        self.tanh = layers.Activation(activation=tf.nn.tanh)

        self.flatten = layers.Flatten()

        self.relu = layers.Activation(activation=tf.nn.relu)

    @tf.function
    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.reshape_layer(x)
        x = self.relu(self.bn_1(x))
        x = self.deconv_1(x)
        x = self.relu(self.bn_2(x))
        x = self.deconv_2(x)
        x = self.relu(self.bn_3(x))
        x = self.deconv_3(x)
        x = self.relu(self.bn_4(x))
        x = self.deconv_4(x)
        x = self.tanh(x)
        x = self.flatten(x)
        return x


class ConvVAE(tf.keras.Model):
    def __init__(self, z_size=32, learning_rate=0.0001, kl_tolerance=0.5, image_size=64, observation_std=0.01, **kwargs):
        super(ConvVAE, self).__init__(**kwargs)
        self.z_size = z_size
        self.kl_tolerance = kl_tolerance
        self.image_size = image_size
        self.encoder = Encoder(z_size=z_size)
        self.decoder = Decoder()
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

        self.observation_std = observation_std

    @tf.function
    def call(self, inputs):
        mu, log_var, z = self.encoder(inputs)
        decoder_output = self.decoder(z)
        obs_epsilon = tf.random.normal(shape=decoder_output.shape)
        reconstructed = decoder_output + self.observation_std * obs_epsilon
        return mu, log_var, z, reconstructed

    @tf.function
    def _kl_diagnormal_stdnormal(self, mu, log_var):
        var = tf.exp(log_var)
        kl = 0.5 * tf.reduce_sum(tf.square(mu) + var - 1. - log_var)
        return kl

    @tf.function
    def _gaussian_log_likelihood(self, targets, mean, std):
        se = 0.5 * tf.reduce_sum(tf.square(targets - mean)) / (2 * tf.square(std)) + tf.math.log(std)
        return se

    @tf.function
    def loss(self, y):
        batch_size = y.shape[0]
        with tf.GradientTape() as tape:
            mu, log_var, z = self.encoder(y)
            decoder_output = self.decoder(z)
            obs_epsilon = tf.random.normal(shape=decoder_output.shape)
            reconstructed = decoder_output + self.observation_std * obs_epsilon

            kl = self._kl_diagnormal_stdnormal(mu, log_var)
            y = tf.reshape(y, [batch_size, -1])
            rl_loss = self._gaussian_log_likelihood(y, decoder_output, self.observation_std)
            loss = (kl + rl_loss) / batch_size
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss, kl, rl_loss, z, reconstructed, mu, log_var


