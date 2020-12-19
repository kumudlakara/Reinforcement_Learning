import gym
import numpy as np
from env.windy_gridworld import WindyGridworldEnv
import itertools
from collections import defaultdict

env = WindyGridworldEnv()
print(env.nS)

def make_epsilon_greedy_policy(Q, epsilon, nA):
	def policy_func(observation):
		A = np.ones(nA, dtype=float)*epsilon/nA
		best_action = np.argmax(Q[observation])
		A[best_action] += (1-epsilon)
		return A
	return policy_func

def sarsa(env, num_episodes, gamma=1, alpha=0.5, epsilon=0.1):
	Q = defaultdict(lambda: np.zeros(env.action_space.n))

	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
	for episode in range(num_episodes):
		if (episode+1) % 100 == 0:
			print("Episode {}/{}.".format(i, num_episodes), end="")

		state = env.reset()
		action_prob = policy(state)
		action = np.random.choice(np.arange(len(action_prob)), p=action_prob)

		for t in itertools.count():
			next_state, reward, done, info = env.step(action)
			next_action_prob = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_prob)), p=next_action_prob)

			td_target = reward+gamma*Q[next_state][next_action]
			td_delta = td_target - Q[next_state][next_action]
			Q[state][action] += alpha * td_delta

			if done:
				break
			action = next_action
			state = next_state
	return Q

Q = sarsa(env, 200)
print(Q)
