import numpy as np
import tensorflow as tf
from random import sample
import numpy as np
from sklearn.preprocessing import StandardScaler


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


def group_frames(input_frames):
    input_frames = np.moveaxis(input_frames, 1, - 2)
    shape = input_frames.shape
    input_frames = np.reshape(input_frames, [*shape[:-2], -1])
    return input_frames


def ungroup_frames(input_frames, n_frame_stack):
    input_frames = np.reshape(input_frames, [*input_frames.shape[:-1], n_frame_stack, -1])
    input_frames = np.moveaxis(input_frames, 3, 1)
    return input_frames


def batch_to_seq(tensor_batch, n_batch, n_steps, flat=False):
    """
    Transform a batch of Tensors, into a sequence of Tensors for recurrent policies

    :param tensor_batch: (TensorFlow Tensor) The input tensor to unroll
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_steps: (int) The number of steps to run for each environment
    :param flat: (bool) If the input Tensor is flat
    :return: (TensorFlow Tensor) sequence of Tensors for recurrent policies
    """
    if flat:
        tensor_batch = tf.reshape(tensor_batch, [n_batch, n_steps])
    else:
        tensor_batch = tf.reshape(tensor_batch, [n_batch, n_steps, -1])
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=n_steps, value=tensor_batch)]


def seq_to_batch(tensor_sequence, flat=False):
    """
    Transform a sequence of Tensors, into a batch of Tensors for recurrent policies

    :param tensor_sequence: (TensorFlow Tensor) The input tensor to batch
    :param flat: (bool) If the input Tensor is flat
    :return: (TensorFlow Tensor) batch of Tensors for recurrent policies
    """
    shape = tensor_sequence[0].get_shape().as_list()
    if not flat:
        assert len(shape) > 1
        n_hidden = tensor_sequence[0].get_shape()[-1]
        return tf.reshape(tf.concat(axis=1, values=tensor_sequence), [-1, n_hidden])
    else:
        return tf.reshape(tf.stack(values=tensor_sequence, axis=1), [-1])


def explained_variance(y_pred, y_true):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: (np.ndarray) the prediction
    :param y_true: (np.ndarray) the expected value
    :return: (float) explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


# Taken from Large Scale Curiosity Study
class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


# Taken from Stable Baselines
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


# https://stackoverflow.com/questions/40181284/how-to-get-random-sample-from-deque-in-python-3
class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]


def explained_variance(y_pred, y_true):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: (np.ndarray) the prediction
    :param y_true: (np.ndarray) the expected value
    :return: (float) explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


# Code taken from https://github.com/adik993/ppo-pytorch

class StandardNormalizer:
    """
    Normalizes the input by subtracting the mean and dividing by standard deviation.
    Uses ``sklearn.preprocessing.StandardScaler`` under the hood.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def partial_fit(self, array):
        self.scaler.partial_fit(self._reshape_for_scaler(array))

    def transform(self, array):
        return self.scaler.transform(self._reshape_for_scaler(array)).reshape(array.shape)

    def partial_fit_transform(self, array):
        self.partial_fit(array)
        return self.transform(array)

    def partial_fit_transform_single(self, array):
        new_shape = (-1, np.prod(array.shape[1:]))
        new_array = array.reshape(new_shape)
        self.scaler.partial_fit(new_array)
        return np.reshape(self.scaler.transform(new_array), array.shape)

    def transform_single(self, array):
        new_shape = (-1, np.prod(array.shape[1:]))
        new_array = array.reshape(new_shape)
        return np.reshape(self.scaler.transform(new_array), array.shape)

    @staticmethod
    def _reshape_for_scaler(array):
        new_shape = (-1, np.prod(array.shape[2:])) if array.ndim > 2 else (-1, 1)
        return array.reshape(new_shape)
