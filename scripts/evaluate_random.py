import numpy as np
import gym

from agent import *

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

	all_returns = 0
	num_episodes = 200

	for e in range(num_episodes):
		print(e)
		state = env.reset()

		episode_return = 0

		while True:
			action = agent.act(s) 
			state, reward, done, _ = env.step(action)
			episode_return += reward

			if done:
				break

		all_returns += episode_return


	print(all_returns/200.)



if __name__ == "__main__":
    main()