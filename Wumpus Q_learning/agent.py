
import numpy as np

# import itertools  # not useful!!!!
"""
Contains the definition of the agent that will run in an
environment.
"""

ACT_UP    = 1
ACT_DOWN  = 2
ACT_LEFT  = 3
ACT_RIGHT = 4
ACT_TORCH_UP    = 5
ACT_TORCH_DOWN  = 6
ACT_TORCH_LEFT  = 7
ACT_TORCH_RIGHT = 8

alpha = 0.01 #learning rate     0.01 slow to converge
l= 0.8 # discount
epsilon = 0.001  # explore
speed = 150 #50   better to use this method, to explore more new situations to enrich the Q,
                 # bigger speed to encourage more exploration during study phase

class Q_learning:
    def __init__(self):
        self.Q = np.zeros([7,7,2,2,4,8])
        self.X=0
        self.Y=0
        self.S=0
        self.B=0
        self.F=0
        self.a=0  #np.random.randint(1, 8)
        self.Record =[]
        self.Last_pos = []

        self.steps=0
        # self.reset()

    def reset(self):
        pass

    def act(self, observation):
        # observation = return (self.agent, self.is_near_wumpus(), self.is_near_hole(), self.rem_charges)

        self.X = observation[0][0]
        self.Y = observation[0][1]
        if observation[1] == True:
            self.S = 1
        if observation[2] == True:
            self.B = 1
        self.F = observation[3]

        self.max_Q = []
        for i in range(1, 9):
            self.max_Q.append(self.Q[self.X, self.Y, self.S, self.B, self.F, i - 1])  # action = i + 1

        random_arm = np.random.random()
        # if random_arm > epsilon:
        if random_arm > 1/(1 + self.steps / speed):  #20
            return  np.argmax(self.max_Q)+1
        else: return np.random.randint(1,9)


    def reward(self, observation, action, reward):
        self.a = action  # ( 1 to 8)

        self.Record.append(reward)
        self.Record = self.Record[-2:]
        self.Last_pos.append((self.X, self.Y, self.S, self.B, self.F, self.a-1))
        self.Last_pos = self.Last_pos[-2:]

        self.steps += 1

        # Q-Learning

        if len(self.Last_pos) == 2:
            self.Q[self.Last_pos[-2]] = self.Q[self.Last_pos[-2]] +  \
                                   alpha * (self.Record[-2] + l * max(self.max_Q) - self.Q[self.Last_pos[-2]])
        else: self.Q[self.X, self.Y, self.S, self.B, self.F, self.a - 1] = reward

       #
        # SARSA  --> epsilon = 0

        # if len(self.Last_pos) >= 2:
        #     self.Q[self.Last_pos[-2]] = self.Q[self.Last_pos[-2]] +  \
        #                            alpha * (self.Record[-2] + l * self.Q[ self.Last_pos[-1]] - self.Q[self.Last_pos[-2]])
        # else: self.Q[self.X, self.Y, self.S, self.B, self.F, self.a - 1] = reward



Agent = Q_learning


# python main.py --ngames 1000 --niter 100 --batch 200 --verbose
# python main.py --ngames 1000 --niter 100 --batch 200
