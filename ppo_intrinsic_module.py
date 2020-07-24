import logging
import argparse
import gym
from collections import deque
import datetime
from PIL import Image
from models.ppo import PPO
from models.vae import ConvVAE
from models.next_state_prediction import NextStatePrediction
import imageio
import importlib
from util import *
import pickle
import os
import numpy as np
import vizdoomgym


def update(update_obs, update_returns, update_masks, update_actions, update_values, old_log_prob, true_reward, epoch,
           states):
    advantages = update_returns - update_values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    with tf.name_scope('ppo'):
        tf.summary.scalar('advantages', np.mean(advantages), step=epoch)
        tf.summary.scalar('returns', np.mean(update_returns), step=epoch)

        explained_var = explained_variance(update_values, update_returns)
        tf.summary.scalar('explained_var', float(explained_var), step=epoch)

        if hyperparams.RECURRENT:
            n_envs = hyperparams.N_ENVS
            n_steps = hyperparams.N_STEPS
            n_batch = n_envs * n_steps
            batch_size = n_batch // hyperparams.MINI_BATCHES
            env_indices = np.arange(n_envs)
            flat_indices = np.arange(n_envs * n_steps).reshape(n_envs, n_steps)
            envs_per_batch = batch_size // n_steps
            for i in range(hyperparams.NO_PT_EPOCHS):
                np.random.shuffle(env_indices)
                for start in range(0, n_envs, envs_per_batch):
                    end = start + envs_per_batch
                    mb_env_inds = env_indices[start:end]
                    mb_flat_inds = flat_indices[mb_env_inds].ravel()
                    slices = (arr[mb_flat_inds] for arr in (
                        update_obs, update_actions, update_returns, advantages, old_log_prob, update_values,
                        update_masks))
                    h_s, c_s = states
                    h_s = h_s[mb_env_inds]
                    c_s = c_s[mb_env_inds]
                    mb_states = [h_s, c_s]
                    pi_loss, value_loss, entropy_loss, total_loss, old_neg_log_val, neg_log_val, approx_kl, ratio, _, _ = \
                        ppo.loss(*slices, mb_states, envs_per_batch, n_steps)

                tf.summary.scalar('pi_loss', pi_loss.numpy().item(), step=epoch)
                tf.summary.scalar('value_loss', value_loss.numpy().item(), step=epoch)
                tf.summary.scalar('old_neg_log_val', old_neg_log_val.numpy().item(), step=epoch)
                tf.summary.scalar('neg_log_val', neg_log_val.numpy().item(), step=epoch)
                tf.summary.scalar('approx_kl', approx_kl.numpy().item(), step=epoch)
                tf.summary.scalar('entropy_loss', entropy_loss.numpy().item(), step=epoch)
                tf.summary.scalar('total_loss', total_loss.numpy().item(), step=epoch)
                tf.summary.scalar('ratio', np.mean(ratio.numpy()).item(), step=epoch)
        else:
            inds = np.arange(update_obs.shape[0])
            total_records = hyperparams.N_STEPS * hyperparams.N_ENVS
            records_per_batch = total_records // hyperparams.MINI_BATCHES
            mb_states = states
            for i in range(hyperparams.NO_PT_EPOCHS):
                np.random.shuffle(inds)
                for start in range(0, total_records, records_per_batch):
                    end = start + records_per_batch
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (
                        update_obs, update_actions, update_returns, advantages, old_log_prob, update_values,
                        update_masks))
                    pi_loss, value_loss, entropy_loss, total_loss, old_neg_log_val, neg_log_val, approx_kl, ratio, _, _ = \
                        ppo.loss(*slices, mb_states, 0, 0)
                    tf.summary.scalar('pi_loss', pi_loss.numpy().item(), step=epoch)
                    tf.summary.scalar('value_loss', value_loss.numpy().item(), step=epoch)
                    tf.summary.scalar('old_neg_log_val', old_neg_log_val.numpy().item(), step=epoch)
                    tf.summary.scalar('neg_log_val', neg_log_val.numpy().item(), step=epoch)
                    tf.summary.scalar('approx_kl', approx_kl.numpy().item(), step=epoch)
                    tf.summary.scalar('entropy_loss', entropy_loss.numpy().item(), step=epoch)
                    tf.summary.scalar('total_loss', total_loss.numpy().item(), step=epoch)
                    tf.summary.scalar('ratio', np.mean(ratio.numpy()).item(), step=epoch)


def calculate_intrinsic_reward(states, prediction_actions, rewards, dones):
    dones = dones.astype(dtype=np.float32)

    current_states = states[:-1]
    current_states = np.reshape(current_states, [n_steps, n_envs, -1])
    next_states = states[1:, :, -hyperparams.PREDICTION_FRAMES:, :]
    next_states = np.reshape(next_states, [n_steps, n_envs, -1])

    if isinstance(env.action_space[0], gym.spaces.Discrete):
        action_size = env.action_space[0].n
        prediction_actions = np.reshape(prediction_actions, [n_steps * n_envs])
        actions = np.eye(action_size)[prediction_actions]
    else:
        actions = np.reshape(prediction_actions, [n_steps * n_envs, -1])
    actions = np.asarray(actions, dtype=np.float32)

    current_states = np.reshape(current_states, [n_steps * n_envs, -1])
    next_states = np.reshape(next_states, [n_steps * n_envs, -1])

    prediction_input = np.concatenate([current_states, actions], axis=1)
    prediction_loss, intrinsic_error, post_intrinsic_error = prediction.loss(prediction_input, next_states)
    # prediction_loss, intrinsic_error, post_intrinsic_error = prediction.loss(prediction_input, next_states)
    if hyperparams.USE_DIFFERENCE:
        intrinsic_reward = np.maximum(0, intrinsic_error - post_intrinsic_error)
    else:
        intrinsic_reward = intrinsic_error
    intrinsic_reward = np.reshape(intrinsic_reward, [n_steps, n_envs])

    with tf.name_scope('intrinsic loss'):
        tf.summary.scalar('Prediction Loss', np.mean(prediction_loss), step=epoch)

    intrinsic_reward = hyperparams.INTRINSIC_COEFFICIENT * intrinsic_reward
    extrinsic_rewards = hyperparams.EXTRINSIC_COEFFICIENT * rewards
    combined_rewards = extrinsic_rewards + intrinsic_reward

    return combined_rewards, intrinsic_reward, extrinsic_rewards


def update_intrinsic_loss():
    with tf.name_scope('intrinsic loss'):
        for i in range(vae_training_epochs):
            batch_indices = sample(range(vae_states.size), hyperparams.VAE_BATCH_SIZE)
            vae_state = np.asarray([vae_states.buffer[index] for index in batch_indices], dtype=np.float32)
            loss, kl_loss, r_loss, z, reconstructed, mu, log_var = vae.loss(vae_state)
            tf.summary.scalar('vae_loss', loss.numpy().item(), step=epoch)
            tf.summary.scalar('kl_loss_loss', kl_loss.numpy().item(), step=epoch)
            tf.summary.scalar('r_loss_loss', r_loss.numpy().item(), step=epoch)


def visualize_agent(env_test, ppo, epoch, vae):
    images = []
    vae_images = []
    o = env_test.reset()
    d = np.zeros([1, ], dtype=np.bool)
    h_s_test = np.zeros([1, hyperparams.RECURRENT_SIZE], dtype=np.float32)
    c_s_test = np.zeros([1, hyperparams.RECURRENT_SIZE], dtype=np.float32)
    step = 0
    while not d[0]:
        o_ = Image.fromarray(np.asarray(o[0, 0], dtype=np.uint8))
        images.append(o_)
        o = o.astype(np.float32)
        o = (o - obs_mean) / obs_std
        o_ = np.reshape(o, [1 * n_frame_stack, *o.shape[2:]])
        mu, log_var, z, reconstructed = vae(o_)
        reconstructed = np.reshape(reconstructed, [1 * n_frame_stack, *o.shape[2:]])
        reconstructed = (reconstructed * obs_std) + obs_mean
        reconstructed = np.asarray(np.clip(np.floor(reconstructed), 0.0, 255.0), dtype=np.uint8)
        o = group_frames(o)
        a, n, v, h_s_test, c_s_test = ppo.call(o, [h_s_test, c_s_test],
                                               np.asarray(np.expand_dims(d, axis=1), dtype=np.float32), 1, 1)
        step += 1
        reconstructed = Image.fromarray(np.reshape(reconstructed[0], [64, 64, 3]))
        vae_images.append(reconstructed)
        o, reward, d, info = env_test.step(a.numpy())
    imageio.mimsave('agents/images' + str(epoch) + '.gif', images)
    imageio.mimsave('agents/vae_images' + str(epoch) + '.gif', vae_images)


if __name__ == '__main__':
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        parser = argparse.ArgumentParser(description='Train PPO.')
        parser.add_argument('hyperparameter_file', type=str)
        args = parser.parse_args()
        logging.basicConfig(format='%(asctime)s %(message)s', filename='custom/INTRINSIC/train_ppo_2.log',
                            level=logging.INFO)

        hyperparams = importlib.import_module("hyperparameters." + args.hyperparameter_file)

        with open("hyperparameters/" + args.hyperparameter_file + ".py", "r") as myfile:
            data = myfile.readlines()
        tf.summary.text("hyperparameters", data, 0)

        n_envs = hyperparams.N_ENVS
        epochs = hyperparams.EPOCHS
        n_steps = hyperparams.N_STEPS
        n_frame_stack = hyperparams.STACKED_FRAMES
        vae_training_epochs = 10
        env = gym.vector.make(hyperparams.ENVIRONMENT_NAME, n_envs, wrappers=hyperparams.WRAPPERS)
        if hyperparams.VISUALIZE:
            env_test = gym.vector.make(hyperparams.ENVIRONMENT_NAME, 1, wrappers=hyperparams.WRAPPERS)
        obs = env.reset()
        s = env_test.reset()
        n_action_size = env.action_space[0].n

        if hyperparams.OBS_NORMALIZATION:
            obs_norm = []
            for i in range(1000):
                action = env_test.action_space.sample()
                s, _, _, _ = env_test.step(action)
                obs_norm.extend(s[0])
            obs_norm = np.asarray(obs_norm, dtype=np.float32)
            obs_mean = np.expand_dims(np.mean(obs_norm, axis=0), axis=0)
            obs_std = np.mean(np.std(obs_norm, axis=0)) + 1e-8
            del obs_norm
        else:
            obs_mean = np.zeros([1, 64, 64, 3], dtype=np.float32)
            obs_std = 255.0

        ppo = PPO(env.action_space[0], hyperparams.EPSILON, hyperparams.ENTROPY_REG,
                  hyperparams.VALUE_COEFFICIENT, hyperparams.INITIAL_LAYER, hyperparams.LEARNING_RATE,
                  hyperparams.MAX_GRAD_NORM, hyperparams.RECURRENT, hyperparams.RECURRENT_SIZE)
        if hyperparams.INTRINSIC:
            vae = ConvVAE(hyperparams.VAE_Z_SIZE, hyperparams.VAE_LEARNING_RATE, hyperparams.VAE_KL_TOLERANCE, 64, hyperparams.OBS_STD)

            prediction = NextStatePrediction(hyperparams.PREDICTION_LEARNING_RATE, hyperparams.PREDICTION_SIZE,
                                             n_steps, n_envs, hyperparams.USE_RNN)

            vae_states = ReplayMemory(max_size=hyperparams.MEMORY_LENGTH)
        # prediction.set_hidden_states(n_envs)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'custom/INTRINSIC/' + hyperparams.ENVIRONMENT_NAME + '_' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        train_summary_writer.set_as_default()

        scores = deque(maxlen=100)
        total_combined_rewards = np.zeros([n_envs, ], dtype=np.float32)
        total_intrinsic_rewards = np.zeros([n_envs, ], dtype=np.float32)
        total_extrinsic_rewards = np.zeros([n_envs, ], dtype=np.float32)
        total_rewards = np.zeros([n_envs, ], dtype=np.float32)
        dones = np.zeros([n_envs, ], dtype=np.bool)
        finished_games = 0
        combined_finished_games = 0

        hidden_state = np.zeros([n_envs, hyperparams.RECURRENT_SIZE], dtype=np.float32)
        cell_state = np.zeros([n_envs, hyperparams.RECURRENT_SIZE], dtype=np.float32)

        epoch = 0

        if hyperparams.INTRINSIC:
            for i in range(hyperparams.PRETRAIN_STEPS):
                action = env.action_space.sample()
                s, _, _, _ = env.step(action)
                s = s.astype(np.float32)
                s = (s - obs_mean) / obs_std
                for k in range(n_envs):
                    for l in range(n_frame_stack):
                        vae_states.append(s[k, l])
                if vae_states.size >= hyperparams.VAE_BATCH_SIZE:
                    update_intrinsic_loss()
                if i % 10 == 0:
                    print("Pretrain step", i, "/", hyperparams.PRETRAIN_STEPS)

            vae_training_epochs = 5
        obs = env.reset()

        if hyperparams.NORMALIZE_REWARDS:
            ret = np.zeros([n_envs, ], dtype=np.float32)
            ret_rms = RunningMeanStd(shape=())

        frames_per_epoch = int(n_steps * n_envs)

        reward_scaler = StandardNormalizer()
        for epoch in range(0, hyperparams.EPOCHS, frames_per_epoch):
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
            mb_intrinsic_rewards, mb_intrinsic_estimates, mb_vae_states = [], [], []
            for _ in range(n_steps):
                obs = obs.astype(np.float32)
                obs = (obs - obs_mean) / obs_std
                if hyperparams.INTRINSIC:
                    for k in range(n_envs):
                        for l in range(n_frame_stack):
                            vae_states.append(obs[k, l])
                    obs_vae = np.reshape(obs, [n_envs * n_frame_stack, *obs.shape[2:]])
                    mu, log_var, z = vae.encoder(obs_vae)
                    mu = np.reshape(mu, [n_envs, n_frame_stack, -1])
                    mb_vae_states.append(mu)

                obs = group_frames(obs)
                actions, neglogpacs, values, hidden_state, cell_state = \
                    ppo.call(obs, [hidden_state, cell_state],
                             np.asarray(np.expand_dims(dones, axis=1), dtype=np.float32), n_envs, 1)
                hidden_state = hidden_state.numpy()
                cell_state = cell_state.numpy()
                mb_obs.append(obs)
                mb_actions.append(actions.numpy())
                mb_values.append(values.numpy())
                mb_neglogpacs.append(neglogpacs.numpy())
                mb_dones.append(dones)
                clipped_actions = actions.numpy()

                if isinstance(env.action_space[0], gym.spaces.Box):
                    clipped_actions = np.clip(actions, env.action_space[0].low, env.action_space[0].high)
                next_obs, rewards, dones, infos = env.step(clipped_actions)

                mb_rewards.append(rewards)
                total_rewards += rewards

                if np.any(dones):
                    with tf.name_scope('rewards'):
                        for j in range(len(total_rewards[dones])):
                            finished_games += 1
                            tf.summary.scalar('Episode Reward', total_rewards[dones][j], step=epoch)
                        scores.extend(total_rewards[dones])
                        total_rewards[dones] = 0.0
                        tf.summary.scalar('Reward Mean', np.mean(scores), step=epoch)
                        tf.summary.scalar('Reward Std', np.std(scores), step=epoch)
                        if not hyperparams.USE_DONES:
                            dones = np.zeros([n_envs, ], dtype=np.bool)
                        hidden_state[dones] = np.zeros([hyperparams.RECURRENT_SIZE, ], dtype=np.float32)
                        cell_state[dones] = np.zeros([hyperparams.RECURRENT_SIZE, ], dtype=np.float32)
                obs = next_obs
            # batch of steps to batch of rollouts
            obs_ = obs.copy()
            obs_ = obs_.astype(np.float32)
            obs_ = (obs_ - obs_mean) / obs_std
            if hyperparams.INTRINSIC:
                obs_vae = np.reshape(obs_, [n_envs * n_frame_stack, *obs.shape[2:]])
                mu, log_var, z = vae.encoder(obs_vae)
                mu = np.reshape(mu, [n_envs, n_frame_stack, -1])
                mb_vae_states.append(mu)

            obs_ = group_frames(obs_)
            last_values = ppo.get_values(obs_, [hidden_state, cell_state],
                                         np.asarray(np.expand_dims(dones, axis=1), dtype=np.float32), n_envs, 1)
            last_values = last_values.numpy()

            mb_obs = np.asarray(mb_obs, dtype=np.float32)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_actions = np.asarray(mb_actions)
            mb_values = np.asarray(mb_values, dtype=np.float32)
            mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
            mb_dones = np.asarray(mb_dones, dtype=np.bool)
            if hyperparams.INTRINSIC:
                mb_vae_states = np.asarray(mb_vae_states, dtype=np.float32)

                mb_rewards, mb_intrinsic_rewards, mb_extrinsic_rewards = calculate_intrinsic_reward(mb_vae_states, mb_actions, mb_rewards, mb_dones)
            if hyperparams.NORMALIZE_REWARDS:
                for index in range(mb_rewards.shape[0]):
                    ret = ret * hyperparams.GAMMA + mb_rewards[index]
                    ret_rms.update(ret)
                    mb_rewards[index] = mb_rewards[index] / np.sqrt(ret_rms.var + 1e-8)
                    if index == mb_rewards.shape[0] - 1:
                        ret[dones] = 0.0
                    else:
                        ret[mb_dones[index + 1]] = 0.0
                mb_rewards = np.clip(mb_rewards, -10.0, 10.0)

                with tf.name_scope('rewards'):
                    tf.summary.scalar('normalized_rewards_mean', np.mean(mb_rewards), step=epoch)

            if hyperparams.INTRINSIC:
                for index in range(mb_rewards.shape[0]):
                    total_combined_rewards += mb_rewards[index]
                    total_intrinsic_rewards += mb_intrinsic_rewards[index]
                    total_extrinsic_rewards += mb_extrinsic_rewards[index]
                    if index == mb_rewards.shape[0] - 1:
                        current_dones = dones
                    else:
                        current_dones = mb_dones[index + 1]
                    for l in range(n_envs):
                        if current_dones[l]:
                            combined_finished_games += 1

                            with tf.name_scope('rewards'):
                                tf.summary.scalar('extrinsic_rewards', total_extrinsic_rewards[l], step=combined_finished_games)
                                tf.summary.scalar('intrinsic_rewards', total_intrinsic_rewards[l], step=combined_finished_games)
                                tf.summary.scalar('combined_rewards', total_extrinsic_rewards[l] + total_intrinsic_rewards[l], step=combined_finished_games)
                                tf.summary.scalar('normalized_rewards', total_combined_rewards[l], step=combined_finished_games)

                            total_combined_rewards[l] = 0.0
                            total_intrinsic_rewards[l] = 0.0
                            total_extrinsic_rewards[l] = 0.0

            mb_dones = mb_dones.astype(dtype=np.float32)

            # discount/bootstrap off value fn
            mb_advs = np.zeros_like(mb_rewards)
            true_reward = np.copy(mb_rewards)
            last_gae_lam = 0

            for step in reversed(range(n_steps)):
                if step == n_steps - 1:
                    nextnonterminal = 1.0 - dones
                    nextvalues = last_values
                else:
                    nextnonterminal = 1.0 - mb_dones[step + 1]
                    nextvalues = mb_values[step + 1]
                delta = mb_rewards[step] + hyperparams.GAMMA * nextvalues * nextnonterminal - mb_values[step]
                mb_advs[step] = last_gae_lam = delta + hyperparams.GAMMA * hyperparams.LAMBDA * nextnonterminal * last_gae_lam
            mb_returns = mb_advs + mb_values

            mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
                map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward))

            update(mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs,
                   true_reward, epoch, [hidden_state, cell_state])
            if hyperparams.INTRINSIC:
                update_intrinsic_loss()

                if (epoch % (frames_per_epoch * hyperparams.VISUALIZE_EPOCH) == 0) and hyperparams.VISUALIZE:
                    visualize_agent(env_test, ppo, epoch, vae)

            with open('custom/INTRINSIC/obs_mean.pkl', 'wb') as file:
                pickle.dump(obs_mean, file)
            with open('custom/INTRINSIC/obs_std.pkl', 'wb') as file:
                pickle.dump(obs_std, file)

            ppo.save_weights('custom/INTRINSIC/ppo.h5')
            if hyperparams.INTRINSIC:
                vae.save_weights('custom/INTRINSIC/vae.h5')
                prediction.save_weights('custom/INTRINSIC/prediction.h5')

            print("Completed epoch", epoch)

    except Exception as ex:
        logging.exception(ex)
        print("Exception occurred", ex)
    finally:
        if env is not None:
            env.close()