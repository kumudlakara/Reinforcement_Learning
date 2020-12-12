import numpy as np
import gym
from collections import defaultdict

env = gym.make('Blackjack-v0')

def epsilon_greedy_policy(Q, epsilon, nA):
	def policy_func(obs):
		A = np.ones(nA, dtype=float)*epsilon/nA
		best_action = np.argmax(Q[obs])
		A[best_action] += (1.0 - epsilon)
		return A
	return policy_func

def mc_control_epsilon_greedy(env, num_episodes, gamma=1.0, epsilon=0.1):
	r_sum = defaultdict(float)
	r_count = defaultdict(float)

	Q = defaultdict(lambda: np.zeros(env.action_space.n))

	policy = epsilon_greedy_policy(Q, epsilon, env.action_space.n)

	for i in range(1, num_episodes+1):
		if i%1000 == 0:
			print("Episode {}/{}.".format(i, num_episodes))

		episode = []
		state = env.reset()
		for t in range(100):
			probs = policy(state)
			action = np.random.choice(np.arange(len(probs)), p=probs)
			next_state, reward, done, info = env.step(action)
			episode.append((state, action, reward))
			if done:
				break
			state = next_state

		sa = set([(tuple(x[0]), x[1]) for x in episode])
		for state, action in sa:
			sa_pair = (state, action)
			first_occurence_indx = next(i for i,x in enumerate(episode)
											if x[0]==state and x[1]==action)
			G = sum([x[2]*(gamma**i) for i,x in enumerate(episode[first_occurence_indx:])])
			r_sum[sa_pair] += G
			r_count[sa_pair] += 1
			Q[state][action] = r_sum[sa_pair]/r_count[sa_pair]
	return Q, policy

Q, policy = mc_control_epsilon_greedy(env, num_episodes = 500000, gamma=1.0, epsilon=0.1)

V = defaultdict(float)
for state, action in Q.items():
	action_value = np.max(action)
	V[state] = action_value

print(V)