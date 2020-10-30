import gym
import numpy as np
import time
from gym import wrappers

def run_episode(env, policy, t_steps = 100, render = False):
	obs = env.reset()
	total_reward = 0
	for t in range(t_steps):
		if render:
			env.render()
		action = policy[obs]
		obs, reward, done, info = env.step(action)
		total_reward += reward
		if done:
			break;
	return total_reward

def gen_random_policy():
	return(np.random.choice(4, size = (16)))

def evaluate_policy(env, policy, n_episodes = 100):
	total_rewards = 0.0
	for _ in range(n_episodes):
		total_rewards += run_episode(env, policy)
	return total_rewards/n_episodes

def crossover(policy1, policy2):
	new_policy = policy1.copy()
	for i in range(len(policy1)):
		rand = np.random.uniform()
		if rand > 0.5:
			new_policy[i] = policy2[i]
	return new_policy

def mutation(policy, p = 0.05):
	new_policy = policy.copy()
	for i in range(len(policy)):
		rand = np.random.uniform()
		if rand < p:
			new_policy[i] = np.random.choice(4)
	return new_policy

if __name__ == '__main__':
	env = gym.make('FrozenLake-v0')

	n_policy = 100
	n_steps = 20
	start = time.time()
	policy_population = [gen_random_policy() for _ in range(n_policy)]

	for i in range(n_steps):
		scores_list = [evaluate_policy(env, policy) for policy in policy_population]
		print("Generation {}: max score = {}".format(i + 1, max(scores_list)))
		policy_ranks = list(reversed(np.argsort(scores_list)))
		elite_set = [policy_population[x] for x in policy_ranks[:5]]
		selection_probab = np.array(scores_list)/np.sum(scores_list)
		child_set = [crossover(policy_population[np.random.choice(range(n_policy), p = selection_probab)],
					 policy_population[np.random.choice(range(n_policy), p = selection_probab)]	)
					for _ in range(n_policy - 5)]
		mutated_list = [mutation(policy) for policy in child_set]
		policy_population = elite_set
		policy_population += mutated_list
	policy_score = [evaluate_policy(env, policy) for policy in policy_population]
	best_policy = policy_population[np.argmax(policy_score)]

	end = time.time()

	print("Best policy score = {}. Time taken : {} seconds".format(max(policy_score), end - start))

	for _ in range(200):
		run_episode(env, best_policy)
	env.close()





