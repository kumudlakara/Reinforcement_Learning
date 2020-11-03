import gym
import numpy as np

def run_episode(env, policy, gamma = 1.0, render = False):
	obs = env.reset()
	total_reward = 0
	step_indx = 0
	while True:
		if render:
			env.render()
		obs, reward, done, info = env.step(int(policy[obs]))
		total_reward += (gamma**step_indx) * reward
		step_indx += 1
		if done:
			break
	return total_reward

def evaluate_policy(env, policy, gamma = 1.0, n = 100):
	scores = [run_episode(env, policy, gamma) for _ in range(n)]
	return np.mean(scores)


def extract_policy_from_v(v, gamma = 1.0):
	policy = np.zeros(env.nS)
	for s in range(env.nS):
		Q_sa = np.zeros(env.action_space.n)
		for a in range(env.action_space.n):
			for next_sr in env.P[s][a]:
				p, s_, r, done = next_sr
				Q_sa[a] += (p*(r + gamma*v[s_]))
		policy[s] = np.argmax(Q_sa)
	return policy

def value_iteration(env, gamma = 1.0):
	v = np.zeros(env.nS)
	max_iterations = 100000
	epsilon = 1e-17
	for i in range(max_iterations):
		prev_v = np.copy(v)
		for s in range(env.nS):
			Q_sa = [sum([p*(r + prev_v[s_]) for p,s_,r, _ in env.P[s][a]]) for a in range(env.nA)]
			v[s] = max(Q_sa)
		if(np.sum(np.fabs(prev_v - v)) <= epsilon):
			print('Value iteration converged at iteration {}'.format(i+1))
			break
	return v

if __name__ == '__main__':
	env = gym.make('FrozenLake8x8-v0')
	gamma = 1.0
	optimal_v = value_iteration(env, gamma)
	policy = extract_policy_from_v(optimal_v)
	policy_score = evaluate_policy(env, policy, gamma, n = 1000)
	print('Policy average score : {}'.format(policy_score))



