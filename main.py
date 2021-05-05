import torch
import gym
import numpy as np
from random import random
import argparse
from tqdm import tqdm
from collections import OrderedDict
import visdom
import time

from utils.common_methods import *
from utils.deep_mind_wrapper import *
from model import DQNet
from replay_buffer import ReplayBuffer


class DQN_trainer:
    def __init__(self, config):
        # env info
        self.env = wrap_deepmind(gym.make('Breakout-v0'), skip=config.action_repeat, no_op_max=config.no_op_max)
        if config.is_monitor:
            self.env = gym.wrappers.Monitor(self.env, 'recording')
        self.action_num = self.env.action_space.n
        self.obs_shape = self.env.observation_space.shape  # [h, w, c]
        self.last_obs = self.env.reset()

        # reply buffer
        self.reply_buffer = ReplayBuffer(config.replay_memory_size, config.agent_history_length)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # initial model [batch_size, h, w, m*c]
        self.eval_model = DQNet(self.obs_shape[2] * config.agent_history_length, self.action_num).to(self.device)
        self.target_model = DQNet(self.obs_shape[2] * config.agent_history_length, self.action_num).to(self.device)

        # train param
        self.exploration = np.linspace(config.initial_exploration, config.final_exploration,
                                       config.final_exploration_frame)
        self.final_exploration_frame = config.final_exploration_frame
        self.discount_factor = config.discount_factor
        self.max_epoch = config.max_epoch
        self.learning_starts = config.learning_starts
        self.update_freq = config.update_freq
        self.target_update_freq = config.target_update_freq
        self.batch_size = config.batch_size

        self.model_path = config.model_path
        self.load_model_freq = config.load_model_freq

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.eval_model.parameters(), lr=config.learning_rate)

        self.viz = visdom.Visdom(env="DQN_train", log_to_filename="./logs/viz_dqn_train.log")
        self.log_freq = config.log_freq

    def collect_memories(self):
        """
        before DQN begins to learn, collect adequate memories.
        :return:
        """
        print("-------------------collect memories------------------------")
        for step in tqdm(range(self.learning_starts)):
            # store observation
            cur_index = self.reply_buffer.store_memory_obs(self.last_obs)
            # encoded_obs = self.reply_buffer.encoder_recent_observation()  # numpy: [m*c, h, w]
            #
            # image_num = int(encoded_obs.shape[0] / self.obs_shape[2])
            # images_numpy = np.array([[encoded_obs[i]] for i in range(image_num)])
            # self.viz.images(torch.from_numpy(images_numpy), win="observations")
            #
            # time.sleep(0.5)

            # choose action randomly
            action = self.env.action_space.sample()
            # interact with env
            obs, reward, done, info = self.env.step(action)
            # clip reward
            reward = np.clip(reward, -1.0, 1.0)
            # store other info
            self.reply_buffer.store_memory_effect(cur_index, action, reward, done)

            if done:
                obs = self.env.reset()

            self.last_obs = obs
        print("---------------------------end-----------------------------")

    def train(self):
        """
        train DQN agent
        :return:
        """
        total_reward = 0
        total_step = 0
        total_ave100_reward = 0
        total_ave100_step = 0
        episode = 0
        episode_100 = 0

        train_ave_loss = 0
        log_step = 0

        self.last_obs = self.env.reset()
        print("-------------------train DQN agent------------------------")
        for step in tqdm(range(1, self.max_epoch)):
            cur_index = self.reply_buffer.store_memory_obs(self.last_obs)
            encoded_obs = self.reply_buffer.encoder_recent_observation()  # numpy: [m*c, h, w]

            # visualize last k frames
            image_num = int(encoded_obs.shape[0] / self.obs_shape[2])
            images_numpy = np.array([[encoded_obs[i]] for i in range(image_num)])
            self.viz.images(torch.from_numpy(images_numpy), win="observations")

            sample = np.random.random()
            # change from 1.0 to 0.1 linearly
            epsilon = self.exploration[min([step, self.final_exploration_frame])]
            if sample > epsilon:
                # numpy: [m*c, h, w] => tensor: [1, m*c, h, w]
                encoded_obs = change_to_tensor(encoded_obs).unsqueeze(0)
                pred_action_values = self.eval_model(encoded_obs)  # [1, 4]
                _, action = pred_action_values.max(dim=1)
                action = action.item()
            else:
                action = self.env.action_space.sample()

            obs, reward, done, info = self.env.step(action)

            total_reward += reward
            total_step += 1

            # reward = np.clip(reward, -1.0, 1.0)

            self.reply_buffer.store_memory_effect(cur_index, action, reward, done)

            if done:
                obs = self.env.reset()
                episode += 1
                total_ave100_reward += total_reward
                total_ave100_step += total_step
                total_reward = 0
                total_step = 0

            self.last_obs = obs

            # train the model
            if step % self.update_freq == 0:
                obs_batch, next_obs_batch, action_batch, reward_batch, done_batch = self.reply_buffer.sample_memories(
                    self.batch_size)
                # numpy to tensor
                obs_batch, next_obs_batch = change_to_tensor(obs_batch), change_to_tensor(next_obs_batch)
                action_batch, reward_batch = change_to_tensor(action_batch, torch.int64), change_to_tensor(reward_batch)

                # estimate Q values
                q_values = self.eval_model(obs_batch)  # [b, action_num]
                q_pred = q_values.gather(dim=1, index=action_batch)  # [b, 1]

                # target Q values
                q_next = self.target_model(next_obs_batch).detach()
                # Bellman equation
                q_target = reward_batch + self.discount_factor * q_next.max(dim=1)[0].view(self.batch_size, -1)

                loss = self.criterion(q_pred, q_target)
                train_ave_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # update target net
            if step % self.target_update_freq == 0:
                self.target_model.load_state_dict(OrderedDict(self.eval_model.state_dict()))

            # save model
            if (step / self.update_freq) % self.load_model_freq == 0:
                torch.save(self.eval_model, self.model_path + '/step%d-trainLoss%.4f.pth' % (step, loss.item()))

            # visualize data
            if (step / self.update_freq) % self.log_freq == 0:
                log_step += 1
                train_ave_loss = train_ave_loss / self.log_freq
                self.viz.line([train_ave_loss], [log_step], win='train_average_loss', update='append', opts=dict(
                                title="train_average_loss",
                                xlabel="log_step",
                                ylabel="average_loss"
                            ))
                train_ave_loss = 0

            if episode % 100 == 0:
                episode_100 += 1
                total_ave100_reward = total_ave100_reward / 100
                total_ave100_step = total_ave100_step / 100
                self.viz.line([total_ave100_reward], [episode_100], win='average100_reward', update='append', opts=dict(
                    title="average100_reward",
                    xlabel="episode_100",
                    ylabel="average_reward"
                ))
                self.viz.line([total_ave100_step], [episode_100], win='average100_step', update='append', opts=dict(
                    title="average100_step",
                    xlabel="episode_100",
                    ylabel="average_step"
                ))

        print("---------------------------end-----------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # paper "Human-level control through deep reinforcement learning" argument
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--replay_memory_size', type=int, default=1000000)
    parser.add_argument('--agent_history_length', type=int, default=4)
    parser.add_argument('--target_update_freq', type=int, default=100000, help='target net update its param')
    parser.add_argument('--discount_factor', type=float, default=0.99, help='discount factor')
    parser.add_argument('--action_repeat', type=int, default=4, help='repeat same action in k frames')
    parser.add_argument('--update_freq', type=int, default=4, help='DQN learn once per learning freq')
    # RMSProp
    parser.add_argument('--learning_rate', type=float, default=0.00025)
    parser.add_argument('--gradient_momentum', type=float, default=0.95)
    parser.add_argument('--squared_gradient_momentum', type=float, default=0.95)
    parser.add_argument('--min_squared_gradient', type=float, default=0.01)
    # epsilon
    parser.add_argument('--initial_exploration', type=float, default=1.0)
    parser.add_argument('--final_exploration', type=float, default=0.1)
    parser.add_argument('--final_exploration_frame', type=int, default=1000000)

    parser.add_argument('--learning_starts', type=int, default=50000, help='after learning starts DQN begin to learn')
    parser.add_argument('--no_op_max', type=int, default=30, help='after reset taking random number of no-ops')

    parser.add_argument('--load_model_freq', type=int, default=10000)
    parser.add_argument('--model_path', type=str, default='./checkpoints/', help='path for saving trained models')

    # other argument
    parser.add_argument('--max_epoch', type=int, default=10000000)
    parser.add_argument('--is_monitor', type=bool, default=False, help='use monitor log the performance of the agent')
    parser.add_argument('--log_freq', type=int, default=1000, help='step size for updating visdom')

    args = parser.parse_args()
    print(args)

    dqn_agent_trainer = DQN_trainer(args)
    dqn_agent_trainer.collect_memories()
    dqn_agent_trainer.train()
