import gym
import numpy as np

def custom_policy(observation):
	return 0 if observation[0] in [20,21] else 1

def generate_episode(policy, env):
	observation = env.reset()
	states = []
	rewards = []
	actions = []
	while True:
		states.append(observation)
		action = policy[observation]
		actions.append(action)
		observation, reward, done, info = env.step(action)
		rewards.append(reward)
		if done:
			break
	return states, rewards, actions

def first_visit_monte_carlo_pred(policy, env, n_episodes):
	V = np.zeros((3,1))
	N = np.zeros((3,1))
	gamma = 1.0

	for _ in range(n_episodes):
		G = 0
		states, rewards, actions = generate_episode(policy, env)
		for t in range(len(states)-1,-1,-1):
			R = rewards[t]
			S = states[t]
			G = gamma*G + R
			if S not in states[:t]:
				N[S] += 1
				V[S] += (G - V[S])/N[S]
	return V

if __name__ == "__main__":
	env = gym.make('Blackjack-v0')
	observation = env.reset()
	policy = custom_policy(observation)
	V = first_visit_monte_carlo_pred(policy, env, 100000)
	printf("optimum policy value function:{}".format(V))




