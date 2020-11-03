import gym
import numpy as np

n_states = 50
gamma = 1.0

def obs_to_state(env, obs):
	env_low = env.observation_space.low
	env_high = env.observation_space.high
	env_dx = (env_high - env.low)/n_states
	a = int((obs[0] - env_low[0])/env_dx[0])
	b = int((obs[1] - env_low[1])/env_dx[1])
	return a,b

def run_episode(env, policy = None, render = False):
	obs = env.reset()
	total_reward = 0
	step_indx = 0
	while True:
		if render:
			env.render()
		if policy is None:
			action = env.action_space.sample()
		else:
			a, b = obs_to_state(env, obs)
			action = policy[a][b]
		obs, reward, done, info = env.step(action)
		total_reward += (gamma**step_indx)*reward
		step_indx += 1
		if done:
			break
	return total_reward

if __name__ == '__main__':
	env = gym.make('MountainCar-v0')
	max_iter = 10000
	lr = 1.0
	min_lr = 0.003
	epsilon = 0.03
	Q_sa = np.zeros((n_states, n_states, 3))
	for i in range(max_iter):
		obs = env.reset()
		total_reward = 0
		step_indx = 0
		alpha = max(min_lr, lr*(0.83**(i//100)))
		while True:
			a,b = obs_to_state(env, obs)
			if np.random.uniform(0,1) < epsilon:
				action = np.random.choice(env.action_space.n)
			else:
				logits = Q_sa[a][b]
				prob = (np.exp(logits))/np.sum(np.exp(logits))
				action = np.random.choice(env.action_space.n, p = prob)
			obs, reward, done, info = env.step(action)
			total_reward += (gamma ** step_indx)*reward
			step_indx += 1

			a_, b_ = obs_to_state(env, obs)
			Q_sa[a][b][action] = Q_sa[a][b][action] + alpha*(reward + gamma*(np.max(Q_sa[a_][b_]) - Q_sa[a][b][action]))
			if done:
				break
		if i%100 == 0:
			print('Iteration: {}   Total Reward = {}'.format(i+1, total_reward))

	optimal_policy = np.argmax(Q_sa, axis = 2)
	optimal_policy_scores = [run_episode(env, optimal_policy, render = False) for _ in range(100)]
	print("Average score of solution = {}".format(np.mean(optimal_policy_scores)))
	run_episode(env, optimal_policy, render = True)


