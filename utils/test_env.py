import gym
import time
import torch

from utils.common_methods import *
from utils.deep_mind_wrapper import *
from model import DQNet
from replay_buffer import ReplayBuffer

model_path = r"../checkpoints/state_dict_step900000_ave_reward_2.5945.pth"

if __name__ == "__main__":
    env = gym.make("Breakout-v0")
    # env = gym.wrappers.Monitor(env, "recording", force=True)
    env = wrap_deepmind(env)

    reply_buffer = ReplayBuffer(1000000, 4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # initial model [batch_size, h, w, m*c]
    eval_model = DQNet(1 * 4, 4).to(device)
    eval_model.load_state_dict(torch.load(model_path))

    starts = 10

    total_reward = 0.0
    total_steps = 0
    last_obs = env.reset()
    episode = 1

    while True:

        cur_index = reply_buffer.store_memory_obs(last_obs)
        encoded_obs = reply_buffer.encoder_recent_observation()  # numpy: [m*c, h, w]

        sample = np.random.random()
        if sample > 0.05:
            # numpy: [m*c, h, w] => tensor: [1, m*c, h, w]
            encoded_obs = change_to_tensor(encoded_obs).unsqueeze(0)
            pred_action_values = eval_model(encoded_obs)  # [1, 4]
            _, action = pred_action_values.max(dim=1)
            action = action.item()
        else:
            action = env.action_space.sample()

        # action = env.action_space.sample()
        last_obs, reward, done, _ = env.step(action)
        reply_buffer.store_memory_effect(cur_index, action, reward, done)

        total_reward += reward
        total_steps += 1
        env.render()
        if done:
            print("Episode %d done in %d steps, total reward %.2f" % (episode, total_steps, total_reward))
            time.sleep(1)
            env.reset()
            if episode > 100:
                break
            episode += 1
            total_reward = 0
