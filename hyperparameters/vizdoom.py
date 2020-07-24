import gym
import numpy as np
from PIL import Image
from gym.spaces import Box
from gym.wrappers import FrameStack
from .wrappers import *

GAMMA = 0.99
N_STEPS = 128
ENTROPY_REG = 0.001
LEARNING_RATE = 1e-4
VALUE_COEFFICIENT = 0.5
MAX_GRAD_NORM = 0.5
LAMBDA = .95
MINI_BATCHES = 4
NO_PT_EPOCHS = 3
EPSILON = .2
EPOCHS = 100000000
N_ENVS = 20
STACKED_FRAMES = 4
INITIAL_LAYER = "CNN"
ENVIRONMENT_NAME = "VizdoomMyWayHomeSparse-v0"
RECURRENT = False
RECURRENT_SIZE = 512

PREDICTION_LEARNING_RATE = 0.001
ARCHITECTURE = [256, 256, 256, 256]
USE_DIFFERENCE = False
INTRINSIC = True
VAE_LEARNING_RATE = 0.0001
VAE_Z_SIZE = 32
VAE_KL_TOLERANCE = 0.5

NORMALIZE_REWARDS = True
VISUALIZE = True
VISUALIZE_EPOCH = 250
PREDICTION_SIZE = int(4 * 32)
PREDICTION_FRAMES = 1
INITIAL_PPO_EPOCH = 100

INTRINSIC_COEFFICIENT = 0.000005
EXTRINSIC_COEFFICIENT = 10.0

MEMORY_LENGTH = 100000

VAE_BATCH_SIZE = 256
PRETRAIN_STEPS = 500
USE_DONES = True
OBS_NORMALIZATION = False
OBS_STD = 0.01

USE_RNN = True


def create_frame_stack(env):
    return FrameStack(env, STACKED_FRAMES)


def create_action_repeat(env):
    return ActionRepeat(env, 4)


WRAPPERS = [ReshapeObs, create_frame_stack, create_action_repeat]