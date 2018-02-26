import numpy as np

"""
Contains the definition of the agent that will run in an
environment.
"""

class Actor_Critic_PG:
    def __init__(self):
        # parameters
        self.gamma = 0.1 # # reward discount in TD error
        self.lr_a = 0.001 # learning rate for actor
        self.lr_c = 0.001  # learning rate for critic

        # features
        self.p = 20  # coordinate division
        self.k = 10  #  velocity division
        self.Sij = np.zeros([self.p + 1, self.k + 1, 2])
        self.PHI = np.zeros([self.p + 1, self.k + 1])  # features

        # matrices
        self.mu = 0 # mean of policy Normal distribution
        self.sigma = 10  # stdev. of Normal distribution
        self.count = 0
        # self.speed = 100 # variance decay speed

        self.error = 0  # rewards
        self.weights = np.random.random([self.p + 1, self.k + 1]) # action-value function parameters
        self.theta = np.random.random([self.p + 1, self.k + 1]) # policy parameters

        # Last state history
        self.Q = [0]
        self.Phi_memory = [np.zeros([self.p + 1, self.k + 1])]
        pass



    def reset(self, x_range):
        (self.x_min, self.x_max)= x_range

        for i in range(self.p + 1):
            for j in range(self.k + 1):
                self.Sij[i, j, 0] = self.x_min + i * (self.x_max-self.x_min) / self.p
                self.Sij[i, j, 1] = -20 + j * 40 / self.k
        pass

    def act(self, observation):
        (self.x, self.v) = observation
        # updata PHI
        for i in range(self.p + 1):
            for j in range(self.k + 1):
                self.PHI[i,j] = np.exp (-(self.x - self.Sij[i, j, 0]) ** 2) * \
                                np.exp(-(self.v - self.Sij[i, j, 1]) ** 2)

        # policy distribution
        self.count += 1
        self.mu = sum(sum(np.multiply(self.PHI, self.theta)))
        self.sigma = 10 / np.log(self.count + 1)

        return np.random.normal(self.mu, self.sigma)

    def reward(self, observation, action, reward):
        self.action = action
        self.reward = reward

        Q = sum(sum(np.multiply(self.PHI, self.weights)))
        self.Q.append(Q)
        # TD error:
        self.error = reward + self.gamma * self.Q[-1] - self.Q[-2]
        # actor: update theta
        self.theta += self.lr_a * (self.action - self.mu) * self.PHI / (self.sigma**2) * self.Q[-2]
        # Critic: update weights
        self.weights += self.lr_c * self.error *self.PHI





Agent = Actor_Critic_PG


# python main.py --ngames 1000 --niter 100 --batch 200 --verbose
# python main.py --ngames 200 --niter 400 --verbose