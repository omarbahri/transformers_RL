#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 16:01:44 2021

@author: omar
"""

import logging
#from offline_rl.gym_minigrid import fourroom_controller
#from offline_rl.gym_minigrid.envs import fourrooms
import numpy as np
import pickle
import gzip
import h5py
import argparse
import gym

from agent import *


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

def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--render', action='store_true', help='Render trajectories')
    #parser.add_argument('--random', action='store_true', help='Noisy actions')
    #parser.add_argument('--num_samples', type=int, default=int(1e5), help='Num samples to collect')
    #args = parser.parse_args()

    env = gym.make("Pendulum-v0")

    s = env.reset()
    #act = env.action_space.sample()
    done = False
    
    random = True
    
    if random:
        agent = RandomAgent(env)
    else:
        agent = TabularAgentOnPolicyTD(env)

    data = reset_data()
    ts = 0
    #reward = -1.0
    
    for i in range(200):
        print(ts)
        
        # act = np.array([agent.act(s)])

        act = agent.act(s) 

        ns, reward, done, _ = env.step(act)
        
        append_data(data, s, act, reward, done)

        if len(data['observations']) % 10000 == 0:
            print(len(data['observations']))

        ts += 1
        if done:
            s = env.reset()
            done = False
            ts = 0
        else:
            s = ns
    
    if random:
        fname = 'datasets/pendulum_random_50000.hdf5'
    else:
        fname = 'datasets/pendulum.hdf5' 
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')


if __name__ == "__main__":
    main()