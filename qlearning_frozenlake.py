import gym
import numpy as np
import random

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.9
gamma = 0.9
epsilon = 0.3

for i_episode in range(2000):
	s = env.reset()
	done = False
	t = 0
	while not done:
		#env.render()
		if random.uniform(0,1) < epsilon:
			a = env.action_space.sample()
		else:
			a = int(np.max(Q[s]) + np.random.randint(4))
			
		next_s, reward, done, info = env.step(a)
		Q[s, a] = ((1 - alpha)*Q[s, a] + alpha*(reward + gamma*np.max(Q[next_s])))
		s = next_s
		t = t+1

	print("Episode-{} ended after {} timesteps".format(i_episode, t+1))
	print(Q)




