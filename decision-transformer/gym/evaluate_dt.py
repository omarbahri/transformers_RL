import torch
import gym
import numpy as np

device = 'cpu'

env = gym.make("Pendulum-v0")

state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

num_episodes = 200

model_name = 'dt_1e-2_emb64_dt_200000_10iter'

model = torch.load('models/' + model_name + '.pkl')
model.eval()
model = model.to(device=device)

all_returns = 0

for e in range(num_episodes):

    #state_mean=[ 0.8124825,   0.08462816, -0.01141869]
    #state_std=[0.4824354,  0.31660303, 1.5608662 ]
    
    # state_mean=[-0.3723861  ,-0.0009759 , -0.08268486]
    # state_std=[0.62041616, 0.69022816, 3.5699291 ]

    state_mean=[0.8361047,  0.08713459, 0.02611632]
    state_std=[0.45545506, 0.31927997, 1.412226  ]


    target_return=0.

    state_mean = torch.from_numpy(np.asarray(state_mean)).to(device=device)
    state_std = torch.from_numpy(np.asarray(state_std)).to(device=device)

    state = env.reset()

    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0

    t=0

    while True:
    	# add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        pred_return = target_return[0,-1]

        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)



        episode_return += reward
        #episode_length += 1

        t+=1

        if done:
            break

    print(episode_return)
    #print(episode_length)
    all_returns+=episode_return

print("Average return: " + str(all_returns/200.))