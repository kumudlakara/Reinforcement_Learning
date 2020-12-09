import gym
import numpy as np
from collections import defaultdict

env = gym.make('Blackjack-v0')

def sample_policy(observation):
	score, dealer_card, usuable_ace = observation
	return 0 if score >= 20 else 1

def mc_prediction(env, policy, num_episodes, gamma = 1):
	r_sum = defaultdict(float)
	r_count = defaultdict(float)
	V = defaultdict(float)

	for i in range(1, num_episodes+1):
		if i % 1000 == 0:
			print("Episode {}/{}.".format(i, num_episodes), end="")

		episode = []
		obs = env.reset()
		for t in range(100):
			action = policy(obs)
			obs_, reward, done, _ = env.step(action)
			episode.append((obs, action, reward))
			if done:
				break
			obs = obs_

		states_in_episode = set([tuple(x[0]) for x in episode])
		for state in states_in_episode:
			first_occur_idx = next(i for i,x in enumerate(episode) if x[0] == state)
			G = sum([x[2]*(gamma**i) for i,x in enumerate(episode[first_occur_idx:])])

			r_sum[state] += G
			r_count[state] += 1
			V[state] = r_sum[state]/r_count[state]

	return V

V = mc_prediction(env, sample_policy, num_episodes = 10000)
print(V)






