import numpy as np
import random
import math

"""
Contains the definition of the agent that will run in an
environment.
"""


# AGENT 1  epsilon_greedy
epsilon = 0.10

class epsilon_greedy:
    def __init__(self):
        self.K_counts = np.zeros(10)
        self.K_values = np.zeros(10)

    def act(self, observation):
        random_arm = np.random.random()
        if random_arm > epsilon:
            return np.argmax(self.K_values)
        else:
            return np.random.randint(0,9)

    def reward(self, observation, action, reward):
        self.K_counts[action] = self.K_counts[action] + 1
        n = self.K_counts[action]
        k_value = self.K_values[action]
        new_k_value = ((n - 1) / float(n)) * k_value + (1 / float(n)) * reward
        self.K_values[action] = new_k_value
        return



# AGENT 2 epsilon_decay_greedy
class e_decay_greedy:
    def __init__(self):
        self.K_counts = np.zeros(10)
        self.K_values = np.zeros(10)

    def act(self, observation):
        random_arm = np.random.random()
        if random_arm > 1/(1 + sum(self.K_counts) / 20):
            return np.argmax(self.K_values)
        else:
            return np.random.randint(0,9)

    def reward(self, observation, action, reward):
        self.K_counts[action] = self.K_counts[action] + 1
        n = self.K_counts[action]
        k_value = self.K_values[action]
        new_k_value = ((n - 1) / float(n)) * k_value + (1 / float(n)) * reward
        self.K_values[action] = new_k_value
        return


# AGENT 2' optimistic_epsilon_greedy
op_epsilon = 0
class op_e_greedy:
    def __init__(self):
        self.K_counts = np.repeat(1., 10)   #np.ones(10)
        self.K_values = np.repeat(15., 10)   #8

    def act(self, observation):
        random_arm = np.random.random()
        if random_arm > op_epsilon:
            return np.argmax(self.K_values)
        else:
            return np.random.randint(0,9)

    def reward(self, observation, action, reward):
        self.K_counts[action] = self.K_counts[action] + 1
        n = self.K_counts[action]
        k_value = self.K_values[action]
        new_k_value = ((n - 1) / float(n)) * k_value + (1 / float(n)) * reward
        self.K_values[action] = new_k_value
        return



# AGENT 3  SoftMax Agent
t = 0.1
class softmax:
    def __init__(self): # initialize the time arm K=0,1,,,,9 was picked and average reward of each arm
        self.K_weights = np.zeros(10)
        self.K_counts = np.zeros(10)
        self.K_values = np.zeros(10)

        self.K_e = np.zeros(10)
        #self.K_probs = np.zeros(10)

    def act(self, observation):  # arm choosing method
        for k in range(len(self.K_values)):
            self.K_e[k] = np.exp((np.array(self.K_values[k]) - np.max(self.K_values)) / t )
        for k in range(len(self.K_weights)):
            self.K_weights[k] = self.K_e[k] / np.sum(self.K_e)

        # stratified sampling
        start = 0
        random_num = random.randint(0, 1000)/100    # (0, 10)

        for k_arm, item in enumerate(self.K_weights*10):
            start += item
            if random_num <= start:
                break
        return k_arm


    def reward(self, observation, action, reward):  # action = arm chosen
        self.K_counts[action] = self.K_counts[action] + 1  # number of times that an arm i was picked
        n = self.K_counts[action]
        k_value = self.K_values[action]  # the reward this time
        new_k_value = ((n - 1) / float(n)) * k_value + (1 / float(n)) * reward  # updated average reward of that arm chosen
        self.K_values[action] = new_k_value
        return



# AGENT 4 UCB Agent
class UCB:
    def __init__(self):
        self.K_counts = np.zeros(10)
        self.K_values = np.zeros(10)

    def act(self, observation):
        K = len(self.K_counts)
        for k_arm in range(K):
            if self.K_counts[k_arm]== 0:
                return k_arm
        UCB_values = [0.0 for k_arm in range(K)]
        total_counts = sum(self.K_counts)
        for k in range(K):
            bonus = math.sqrt((2*math.log(total_counts))/float(self.K_counts[k]))
            UCB_values[k] = self.K_values[k]+bonus
        k_arm = UCB_values.index(max(UCB_values))
        return k_arm

    def reward(self,observation, action, reward):
        self.K_counts[action] = self.K_counts[action] + 1
        n = self.K_counts[action]
        k_value = self.K_values[action]
        new_k_value = ((n - 1) / float(n)) * k_value + (1 / float(n)) * reward
        self.K_values[action] = new_k_value
        return


# Choose which Agent is run for scoring
Agent = op_e_greedy

    # epsilon_greedy
    # e_decay_gree
    # op_e_greedy
    # softmax
    # UCB



# RUN in terminal:
# python main.py --niter 1000 --batch 2000 --verbose