

import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_probability as tfp
import math
import numpy as np
import gym
from .dist import CategoricalProbabilityDistribution, DiagGaussianProbabilityDistribution
from util import *


class Actor(layers.Layer):
    def __init__(self, action_space, epsilon):
        super(Actor, self).__init__()
        self.epsilon = epsilon

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_size = action_space.n
            self.dist = CategoricalProbabilityDistribution()
        else:
            self.action_size = action_space.shape[0]
            self.dist = DiagGaussianProbabilityDistribution(self.action_size)
        self.dense = layers.Dense(self.action_size, activation=None,
                                  kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
                                  bias_initializer=tf.zeros_initializer())

    @tf.function
    def call(self, inputs):
        pi_latent = self.dense(inputs)
        return pi_latent

    @tf.function
    def sample(self, pi_latent):
        pi = self.dist.sample(pi_latent)
        negative_logp = self.dist.neg_log_p(pi, pi_latent)
        return pi, negative_logp

    @tf.function
    def loss(self, pi_latent, advantages, actions, neg_logp_old):
        # pi_latent = self(inputs)
        negative_logp = self.dist.neg_log_p(actions, pi_latent)
        ratio = tf.exp(neg_logp_old - negative_logp)

        pi_loss = tf.reduce_mean(tf.maximum(
            - ratio * advantages, - tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
            )
        )
        approx_kl = .5 * tf.reduce_mean(tf.square(negative_logp - neg_logp_old))
        entropy_loss = tf.reduce_mean(self.dist.entropy(pi_latent))
        return pi_loss, entropy_loss, tf.reduce_mean(neg_logp_old), tf.reduce_mean(negative_logp), approx_kl, ratio


class Critic(layers.Layer):
    def __init__(self):
        super(Critic, self).__init__()
        self.dense = layers.Dense(1, activation=None, kernel_initializer=tf.keras.initializers.Orthogonal(0.01), bias_initializer=tf.zeros_initializer())
        self.clip_range = 0.2

    @tf.function
    def call(self, inputs):
        value = self.dense(inputs)
        return tf.squeeze(value, axis=1)

    @tf.function
    def loss_clipped(self, values, returns, old_values):
        # value = self(inputs)
        pred_clipped = old_values + tf.clip_by_value(values - old_values, - self.clip_range, self.clip_range)
        loss_1 = tf.square(values - returns)
        loss_2 = tf.square(pred_clipped - returns)
        loss = 0.5 * tf.reduce_mean(tf.maximum(loss_1, loss_2))
        return loss

    @tf.function
    def loss(self, values, returns):
        loss = tf.square(values - returns)
        loss = 0.5 * tf.reduce_mean(loss)
        return loss


class CNN(layers.Layer):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_1 = layers.Conv2D(32, 8, 4, activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)), bias_initializer=tf.zeros_initializer())
        self.conv_2 = layers.Conv2D(64, 4, 2, activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)), bias_initializer=tf.zeros_initializer())
        self.conv_3 = layers.Conv2D(64, 3, 1, activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)), bias_initializer=tf.zeros_initializer())
        self.dense = layers.Dense(512, activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)), bias_initializer=tf.zeros_initializer())
        self.flatten = layers.Flatten()

    @tf.function
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


class MLP(layers.Layer):
    def __init__(self):
        super(MLP, self).__init__()
        self.dense_1 = layers.Dense(64, activation=tf.nn.tanh,
                                    kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
                                    bias_initializer=tf.zeros_initializer())
        self.dense_2 = layers.Dense(64, activation=tf.nn.tanh,
                                    kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
                                    bias_initializer=tf.zeros_initializer())

    @tf.function
    def call(self, inputs):
        common = self.dense_1(inputs)
        common = self.dense_2(common)
        return common


class PPO(tf.keras.Model):
    def __init__(self, action_space, epsilon, entropy_reg, value_coeff, initial_layer, learning_rate, max_grad_norm,
                 recurrent=False, recurrent_size=256):
        super(PPO, self).__init__()
        self.actor = Actor(action_space, epsilon)
        self.critic = Critic()
        self.value_coeff = value_coeff
        self.entropy_reg = entropy_reg
        self.max_grad_norm = max_grad_norm
        if initial_layer == "MLP":
            self.initial_layer = MLP()
        else:
            self.initial_layer = CNN()
        self.recurrent = recurrent
        self.recurrent_size = recurrent_size
        if self.recurrent:
            self.lstm = layers.LSTM(self.recurrent_size, recurrent_initializer='glorot_uniform', stateful=False,
                                    return_state=True, return_sequences=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-5)

    @tf.function
    def call(self, obs, states, masks, n_envs, seq_len):
        common, h_s, c_s = self.call_common(obs, states, masks, n_envs, seq_len)
        pi_latent = self.actor(common)
        value = self.critic(common)
        pi, logp_pi = self.actor.sample(pi_latent)
        return pi, logp_pi, value, h_s, c_s

    @tf.function
    def get_values(self, obs, states, masks, n_envs, seq_len):
        if self.recurrent:
            common, h_s, c_s = self.call_common(obs, states, masks, n_envs, seq_len)
        else:
            common = self.initial_layer(obs)
        value = self.critic(common)
        return value

    @tf.function
    def call_common(self, obs, states, masks, n_envs, n_steps=1):
        masks = 1.0 - masks
        common = self.initial_layer(obs)
        if self.recurrent:
            input_sequence = batch_to_seq(common, n_envs, n_steps)
            masks = batch_to_seq(masks, n_envs, n_steps)
            h_s = tf.convert_to_tensor(states[0])
            c_s = tf.convert_to_tensor(states[1])
            for idx, (_input, mask) in enumerate(zip(input_sequence, masks)):
                h_s = (h_s * mask)
                c_s = (c_s * mask)
                _input = tf.expand_dims(_input, axis=1)
                rnn_output, h_s, c_s = self.lstm(_input, initial_state=[h_s, c_s])
                input_sequence[idx] = rnn_output
            common = seq_to_batch(input_sequence)
        else:
            h_s = tf.convert_to_tensor(states[0])
            c_s = tf.convert_to_tensor(states[1])
        return common, tf.stop_gradient(h_s), tf.stop_gradient(c_s)

    @tf.function
    def loss(self, obs, actions, returns, advantages, logp_old, old_values, masks, states, n_envs, n_steps):

        with tf.GradientTape() as tape:
            common, h_s, c_s = self.call_common(obs, states, masks, n_envs, n_steps)
            pi_latent = self.actor(common)
            values = self.critic(common)
            value_loss = self.critic.loss_clipped(values, returns, old_values)
            pi_loss, entropy_loss, old_neg_log_val, neg_log_val, approx_kl, ratio = self.actor.loss(pi_latent, advantages, actions, logp_old)
            loss = pi_loss - entropy_loss * self.entropy_reg + value_loss * self.value_coeff
        grads = tape.gradient(loss, self.trainable_weights)
        grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return pi_loss, value_loss, entropy_loss, loss,  old_neg_log_val, neg_log_val, approx_kl, ratio, h_s, c_s