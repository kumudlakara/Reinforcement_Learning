import numpy as np
import gym
import time

def run_episode(env, policy, t_steps = 100, render = False):
	obs = env. reset()
	total_reward = 0
	for t in range(t_steps):
		if render:
			env.render()
		action = policy[obs]

		obs, reward, done, info = env.step(action)
		total_reward += reward
		if done:
			break
	return total_reward

def gen_random_policy():
	return np.random.choice(4, size = 16)

def evaluate_policy(env, policy, episodes_n = 100):
	total_rewards = 0.0
	for _ in range (episodes_n):
		total_rewards += run_episode(env, policy)
	return  total_rewards/episodes_n

if __name__ == '__main__':
	env = gym.make('FrozenLake-v0')

	#no of random policies
	n_policy = 2000
	start = time.time()
	policy_list = [gen_random_policy() for _ in  range(n_policy)]
	score_list = [evaluate_policy(env, policy) for policy in policy_list]

	end = time.time()

	print("Best score = {}. Time taken = {} seconds".format(max(score_list), end - start))




