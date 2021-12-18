import gym
from params_pool import ParamsPool
import argparse
from action_wrappers import ScalingActionWrapper

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


def generate_trained():
    save_dir='results/trained_policies_pth'
    filename='1000.pth'

    env_raw = gym.make('Pendulum-v0')
    env = ScalingActionWrapper(env_raw, scaling_factors=env_raw.action_space.high)

    num_episodes = 50000

    data = reset_data()

    print("About to load model")

    param = ParamsPool(
        input_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    param.load_actor(save_dir=save_dir, filename=filename)  # critics are not loaded and act does not depend on them

    print('Loaded. Vamos!')

    for e in range(num_episodes):
        obs = env.reset()
        print(e)

        while True:
            act = param.act(obs)
            next_obs, reward, done, _ = env.step(act)

            append_data(data, obs, act, reward, done)

            if done: break
            obs = next_obs

    env.close()

    fname = 'results/datasets/pendulum_medium_1000000.hdf5' 
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')

if __name__ == '__main__':
    generate_trained()