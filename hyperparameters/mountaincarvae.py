import gym
import numpy as np
from PIL import Image
from gym.spaces import Box
from gym.wrappers import FrameStack
from .wrappers import *

GAMMA = 0.95
N_STEPS = 200
ENTROPY_REG = 0.01
LEARNING_RATE = 2e-4
VALUE_COEFFICIENT = 0.5
MAX_GRAD_NORM = 0.5
LAMBDA = 0.95
MINI_BATCHES = 4
NO_PT_EPOCHS = 10
EPSILON = .3
N_ENVS = 10
EPOCHS = int(1000 * N_STEPS * N_ENVS)
STACKED_FRAMES = 4
INITIAL_LAYER = "CNN"
ENVIRONMENT_NAME = "MountainCar-v0"
RECURRENT = False
RECURRENT_SIZE = 256

PREDICTION_LEARNING_RATE = 0.01
ARCHITECTURE = [256, 256, 256, 256]
USE_DIFFERENCE = False
VAE = True
VAE_LEARNING_RATE = 0.0001
VAE_Z_SIZE = 8
VAE_KL_TOLERANCE = 0.5

NORMALIZE_REWARDS = True
VISUALIZE = True
VISUALIZE_EPOCH = 10
PREDICTION_SIZE = 8
INITIAL_PPO_EPOCH = 100

INTRINSIC_COEFFICIENT = 0.1
EXTRINSIC_COEFFICIENT = 10.0

MEMORY_LENGTH = 10000

VAE_BATCH_SIZE = 256
PRETRAIN_STEPS = 10
USE_DONES = True
OBS_NORMALIZATION = False
OBS_STD = 0.01


def create_frame_stack(env):
    return FrameStack(env, STACKED_FRAMES)


def create_action_repeat(env):
    return ActionRepeat(env, 4, clip_reward=False)


# WRAPPERS = [ActionRepeat, ReshapeObs, create_frame_stack, GroupObs]
WRAPPERS = [MakeSparse, VecToImage, ReshapeObs, create_frame_stack, create_action_repeat]
