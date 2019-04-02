import numpy as np
import random
import os
import time
import json

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer


class ReplayBuffer(object):
      def __init__(self, window=1e4, warm_up=0, delay=0):
            self.storage = []
            self.episode_lens = []
            self.l_margin = 0
            self.u_margin = 0

            self.window = window
            self.warm_up = warm_up
            self.delay = delay

      # Expects tuples of (state, next_state, action, reward, done)
      def add_sample(self, data):
            self.storage.append(data)

      def add_episode_len(self, data):
            self.episode_lens.append(data)

      def set_margins(self):
            u = min(len(self.episode_lens), self.warm_up) + max(0, len(self.episode_lens) - self.warm_up - self.delay)
            l = max(0, u - self.window)
            # self.u_margin = min(len(self.storage), np.sum(self.episode_lens[len(self.episode_lens) - self.warm_up:-1])) \
            #     + max(0, len(self.storage) - np.sum(self.episode_lens[len(self.episode_lens) - self.warm_up:-1]) - np.sum(self.episode_lens[len(self.episode_lens) - self.delay:-1]))
            # self.l_margin = max(0, self.u_margin - np.sum(self.episode_lens[-self.window:]))

            self.l_margin = np.sum(self.episode_lens[-l:])
            self.u_margin = np.sum(self.episode_lens[-u:])

      def sample(self, batch_size=100):

            # self.u_margin = min(len(self.storage), np.sum(self.episode_lens[-self.warm_up:])) + max(0, len(self.storage) - np.sum(self.episode_lens[-self.warm_up:]) - np.sum(self.episode_lens[-self.delay:]))
            # self.l_margin = max(0, self.u_margin - np.sum(self.episode_lens[-self.window:]))
            ind = np.random.randint(self.l_margin, self.u_margin, size=batch_size)

            x, y, u, r, d = [], [], [], [], []

            for i in ind:
                  X, Y, U, R, D = self.storage[i]
                  x.append(np.array(X, copy=False))
                  y.append(np.array(Y, copy=False))
                  u.append(np.array(U, copy=False))
                  r.append(np.array(R, copy=False))
                  d.append(np.array(D, copy=False))

            return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


def create_folder(f): return [os.makedirs(f) if not os.path.exists(f) else False]


class Logger(object):
      def __init__(self, args, experiment_name='', environment_name='', folder='./results'):
            """
            Saves experimental metrics for use later.
            :param experiment_name: name of the experiment
            :param folder: location to save data
            : param environment_name: name of the environment
            """
            self.rewards = []
            self.save_folder = os.path.join(folder, experiment_name, environment_name, time.strftime('%y-%m-%d') + '_' + str(args.window) + '_' + str(args.delay))
            create_folder(self.save_folder)
            with open(os.path.join(self.save_folder, 'params.json'), 'w') as f:
                  json.dump(dict(args._get_kwargs()), f)

      def record_reward(self, reward_return):
            self.returns_eval = reward_return

      def training_record_reward(self, reward_return):
            self.returns_train = reward_return

      def record_losses(self, critic_loss_avg, actor_loss_avg, critic_loss, actor_loss):
            self.critic_loss_avg = critic_loss_avg
            self.actor_loss_avg = actor_loss_avg
            self.critic_loss = critic_loss
            self.actor_loss = actor_loss

      def save(self):
            np.save(os.path.join(self.save_folder, "returns_eval.npy"), self.returns_eval)
            np.save(os.path.join(self.save_folder, "returns_eval.npy"), self.returns_train)
            np.save(os.path.join(self.save_folder, "returns_eval.npy"), self.critic_loss_avg)
            np.save(os.path.join(self.save_folder, "returns_eval.npy"), self.actor_loss_avg)
            np.save(os.path.join(self.save_folder, "returns_eval.npy"), self.critic_loss)
            np.save(os.path.join(self.save_folder, "returns_eval.npy"), self.actor_loss)
