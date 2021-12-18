#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 19:38:56 2021

@author: omar
"""

import numpy as np

import collections
import pickle
import h5py    

datasets = []

name = 'pendulum'
#dataset = h5py.File(name + '_random.hdf5','r+')    
#dataset = h5py.File(name + '.hdf5','r+')
dataset = h5py.File('datasets/' + name + '_medium_200000.hdf5','r+')


N = dataset['rewards'].shape[0]
data_ = collections.defaultdict(list)
  
use_timeouts = False
if 'timeouts' in dataset:
    use_timeouts = True
  
episode_step = 0
paths = []
for i in range(N):
    print(i)
    done_bool = bool(dataset['terminals'][i])
    if use_timeouts:
        final_timestep = dataset['timeouts'][i]
    else:
        final_timestep = (episode_step == 1000-1)
    for k in ['observations', 'actions', 'rewards', 'terminals']:
        data_[k].append(dataset[k][i])
    if done_bool or final_timestep:
        episode_step = 0
        episode_data = {}
        for k in data_:
            episode_data[k] = np.array(data_[k])
        paths.append(episode_data)
        data_ = collections.defaultdict(list)
    episode_step += 1
  
returns = np.array([np.sum(p['rewards']) for p in paths])
num_samples = np.sum([p['rewards'].shape[0] for p in paths])
print(f'Number of samples collected: {num_samples}')
#print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')
  
with open('datasets/' + name + '-medium_200000-v2.pkl', 'wb') as f:
    pickle.dump(paths, f)