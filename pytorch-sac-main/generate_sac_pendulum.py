import time
import gym
from gym.wrappers import TimeLimit
from replay_buffer import ReplayBuffer, Transition
from params_pool import ParamsPool
from action_wrappers import ScalingActionWrapper
from make_video_of_saved_actor import make_video_of_saved_actor
from generate_trained_ql_pendulum import generate_trained

import wandb
import argparse

import logging
import numpy as np
import pickle
import gzip
import h5py
import argparse

def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': []
            }

def append_data(data, s, a, reward, done):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(reward)
    data['terminals'].append(done)
    
def npify(data):
    for k in data:
        if k == 'terminals':
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

# =================================================================================
# arguments

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=int)
args = parser.parse_args()

# =================================================================================
# logging

wandb.init(
    project='sa2c',
    entity='omarb',
    group=f'Pendulum-v0-sac',
    settings=wandb.Settings(_disable_stats=True),
    name=f'run_id={args.run_id}'
)

# =================================================================================

env_raw = gym.make('Pendulum-v0')
env = ScalingActionWrapper(env_raw, scaling_factors=env_raw.action_space.high)
buf = ReplayBuffer(capacity=int(1e6))
param = ParamsPool(
    input_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0]
)

batch_size = 64
num_episodes = 1000

start_time = time.perf_counter()

data = reset_data()

for e in range(num_episodes):

    obs = env.reset()

    total_reward = 0
    total_updates = 0

    while True:

        # ==================================================
        # getting the tuple (s, a, r, s', done)
        # ==================================================

        action = param.act(obs)
        next_obs, reward, done, _ = env.step(action)

        append_data(data, obs, action, reward, done)

        # no need to keep track of max time-steps, because the environment
        # is wrapped with TimeLimit automatically (timeout after 1000 steps)

        total_reward += reward

        # ==================================================
        # storing it to the buffer
        # ==================================================

        buf.push(Transition(obs, action, reward, next_obs, done))

        # ==================================================
        # update the parameters
        # ==================================================
 
        if buf.ready_for(batch_size):
            param.update_networks(buf.sample(batch_size))
            total_updates += 1

        # ==================================================
        # check done
        # ==================================================

        if done: break

        obs = next_obs

    # ==================================================
    # after each episode
    # ==================================================

    wandb.log({'return': total_reward})

    after_episode_time = time.perf_counter()
    time_elapsed = after_episode_time - start_time
    time_remaining = time_elapsed / (e + 1) * (num_episodes - (e + 1))

    print(f'Episode {e:4.0f} | Return {total_reward:9.3f} | Updates {total_updates:4.0f} | Remaining time {round(time_remaining / 3600, 2):5.2f} hours')

fname = 'results/datasets/pendulum_sac_1000_1000.hdf5' 
dataset = h5py.File(fname, 'w')
npify(data)
for k in data:
    dataset.create_dataset(k, data=data[k], compression='gzip')

param.save_actor(
    save_dir='results/trained_policies_pth/',
    filename=f'{args.run_id}.pth'
)

make_video_of_saved_actor(
    save_dir='results/trained_policies_pth/',
    filename=f'{args.run_id}.pth',
    run_id=args.run_id
)

#generate_trained()

env.close()
