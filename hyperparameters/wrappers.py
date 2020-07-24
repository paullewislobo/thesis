import gym
from gym.spaces import Box
from PIL import Image
import numpy as np


class ReshapeObs(gym.Wrapper):
    def __init__(self, env=None):
        super(ReshapeObs, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

    @staticmethod
    def process_image(obs):
        obs = Image.fromarray(obs).resize((100, 100), resample=Image.BILINEAR).resize(
            (64, 64), resample=Image.BILINEAR) # .convert('L')
        return obs

    def reset(self):
        obs = self.env.reset()
        return self.process_image(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.process_image(obs), reward, done, info


class ReshapeObsGray(gym.Wrapper):
    def __init__(self, env=None):
        super(ReshapeObsGray, self).__init__(env)
        self.observation_space = Box(low=0.0, high=1.0, shape=(64, 64, 1), dtype=np.uint8)

    @staticmethod
    def process_image(obs):
        obs = Image.fromarray(obs).resize((100, 100), resample=Image.BILINEAR).resize(
            (64, 64), resample=Image.BILINEAR).convert('L')
        np.expand_dims(obs, axis=2)
        return obs

    def reset(self):
        obs = self.env.reset()
        return self.process_image(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.process_image(obs), reward, done, info


class ActionRepeat(gym.Wrapper):
    def __init__(self, env=None, frame_skip=4, clip_reward=True):
        super(ActionRepeat, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(frame_skip, 64, 64, 3), dtype=np.uint8)
        self.frame_skip = frame_skip
        self.clip_reward = clip_reward

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            if self.clip_reward:
                reward = 0.0 if reward < 0.0 else reward  # TODO Move to new wrapper later
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class VecToImage(gym.Wrapper):
    def __init__(self, env=None):
        super(VecToImage, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(400, 600, 3), dtype=np.uint8)

    @staticmethod
    def process_obs(obs):
        return np.asarray(obs, dtype=np.uint8)

    def reset(self):
        self.env.reset()
        obs = self.env.render(mode='rgb_array')
        return self.process_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.env.render(mode='rgb_array')
        return self.process_obs(obs), reward, done, info


class GroupObs(gym.Wrapper):
    def __init__(self, env=None):
        super(GroupObs, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(64, 64, 4), dtype=np.uint8)

    @staticmethod
    def process_image(obs):
        obs = np.moveaxis(obs, 0, 2)
        return obs

    def reset(self):
        obs = self.env.reset()
        return self.process_image(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.process_image(obs), reward, done, info


class MakeSparse(gym.Wrapper):
    def __init__(self, env=None):
        super(MakeSparse, self).__init__(env)
        self.steps = 0

    def reset(self):
        self.steps = 0
        return self.env.reset()

    def step(self, action):
        self.steps += 1
        obs, reward, done, info = self.env.step(action)
        if done and self.steps < 200:
            reward = 1.0
        else:
            reward = 0.0
        return obs, reward, done, info