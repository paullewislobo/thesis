import tensorflow as tf
from tensorflow.keras import layers, models


class NextStatePrediction(tf.keras.Model):
    def __init__(self, learning_rate, prediction_size, n_steps, n_envs, use_rnn):
        super(NextStatePrediction, self).__init__()
        self.hidden_layers = []
        self.use_rnn = use_rnn
        self.dense_1 = layers.Dense(256, activation=tf.nn.relu)
        if self.use_rnn:
            self.lstm = layers.LSTM(512, recurrent_initializer='glorot_uniform', stateful=False,
                                    return_state=True, return_sequences=True)
        else:
            self.lstm = layers.Dense(512, activation=tf.nn.relu)
        self.dense_2 = layers.Dense(256, activation=tf.nn.relu)
        self.dense_3 = layers.Dense(256, activation=tf.nn.relu)
        self.dense_4 = layers.Dense(prediction_size, activation=None)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-3)
        self.n_envs = n_envs
        self.n_steps = n_steps

        self.hidden_state = tf.zeros([n_envs, 512])
        self.cell_state = tf.zeros([n_envs, 512])

    @tf.function
    def call(self, inputs, h_s, c_s):
        inputs = self.dense_1(inputs)
        hidden_state = h_s
        cell_state = c_s
        if self.use_rnn:
            inputs = tf.reshape(inputs, [self.n_steps, self.n_envs, -1])
            inputs = tf.transpose(inputs, [1, 0, 2])  # Convert to batch, step, data
            inputs, hidden_state, cell_state = self.lstm(inputs, initial_state=[h_s, c_s])
            inputs = tf.transpose(inputs, [1, 0, 2])  # Convert to step, batch, data
            inputs = tf.reshape(inputs, [self.n_steps * self.n_envs, -1])
        else:
            inputs = self.lstm(inputs)
        inputs = self.dense_2(inputs)
        inputs = self.dense_3(inputs)
        prediction = self.dense_4(inputs)
        return prediction, hidden_state, cell_state

    def loss(self, inputs, targets):
        h_s = self.hidden_state
        c_s = self.cell_state
        with tf.GradientTape() as tape:
            prediction, hidden_state, cell_state = self.call(inputs, h_s, c_s)
            # intrinsic_error = tf.reduce_mean(tf.square(prediction - targets), axis=-1)
            intrinsic_error = tf.square(tf.norm(prediction - targets, 2, axis=-1))
            loss = tf.reduce_mean(intrinsic_error)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        prediction, hidden_state, cell_state = self.call(inputs, h_s, c_s)
        # post_intrinsic_error = tf.reduce_mean(tf.square(prediction - targets), axis=-1)
        post_intrinsic_error = tf.square(tf.norm(prediction - targets, 2, axis=-1))
        self.hidden_state = hidden_state
        self.cell_state = cell_state
        return loss, intrinsic_error, post_intrinsic_error

    def set_hidden_state(self, n_envs):
        self.hidden_state = tf.zeros([n_envs, 512])
        self.cell_state = tf.zeros([n_envs, 512])
