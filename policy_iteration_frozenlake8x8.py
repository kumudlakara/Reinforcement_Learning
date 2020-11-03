import gym
import numpy as np

def run_episode(env, policy, gamma = 1.0, render = False):
	obs = env.reset()
	total_reward = 0
	step_indx = 0 #for calculating discounted reward
	while True:
		if render:
			env.render()
		obs, reward, done, info = env.step(int(policy[obs]))
		total_reward += (gamma**step_indx)*reward
		step_indx += 1
		if done:
			break
	return total_reward

def evaluate_policy(env, policy, gamma = 1.0, n = 100):
	scores = [run_episode(env, policy, gamma) for _ in range(n)]
	return np.mean(scores)

def extract_policy(v, gamma = 1.0):
	policy = np.zeros(env.nS)
	for s in range(env.nS):
		Q_sa = np.zeros(env.nA)
		for a in range(env.nA):
			Q_sa[a] = sum([p*(r + gamma*v[s_]) for p,s_,r,_ in env.P[s][a]])
		policy[s] = np.argmax(Q_sa)
	return policy

def compute_policy_v(env, policy, gamma = 1.0):
	v = np.zeros(env.nS)
	epsilon = 1e-10
	while True:
		prev_v = np.copy(v)
		for s in range(env.nS):
			v[s] = sum([p*(r + gamma*v[s_]) for p,s_,r,_ in env.P[s][policy[s]]])
		if (np.sum(np.fabs(prev_v - v))) <= epsilon:
			break
	return v	

def policy_iteration(env, gamma = 0.1):
	policy = np.random.choice(env.nA, size = (env.nS))
	max_iter = 200000
	gamma = 1.0
	for i in range(max_iter):
		old_policy_v = compute_policy_v(env, policy, gamma)
		new_policy = extract_policy(old_policy_v)
		if (np.all(policy == new_policy)):
			print('Policy iteration converged at iteration: {}'.format(i+1))
			break
		policy = new_policy
	return policy

if __name__ == '__main__':
	env = gym.make('FrozenLake8x8-v0')
	gamma = 1.0
	optimal_policy = policy_iteration(env, gamma)
	policy_score = evaluate_policy(env, optimal_policy, gamma, n = 1000)
	print('Average score = {}'.format(policy_score))
