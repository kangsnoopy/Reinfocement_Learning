
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


l = 0.8 # optimal
class Q_learning:
    def __init__(self):

        self.gridsize = (7,7)
        self.X= np.arange( self.gridsize[0] ) # [0,1,2,3,4,5,6]
        self.Y = np.arange(self.gridsize[1])
        # self.cases = list(itertools.product(self.X, self.Y))  # will not be useful!!!!

        self.cases = []
        for i in self.X:
            for j in self.Y:
                self.cases.append((i, j))


        self.Q = np.zeros((7,7))  # reward -1 marks " I've been there"  Value matrix
        self.B = np.zeros((7,7))  # Breeze memory  boolean
        self.S = np.zeros((7,7))  # Smell memory   boolean

        # self.position = self.environment
        self.Action = []   # record the action of eac step
        self.counts = 0

        self.reset()

    def reset(self):

        self.Action = []


    def act(self, observation):
        # observation = return (self.agent, self.is_near_wumpus(), self.is_near_hole(), self.rem_charges)
        # Flash = observation[3]
        (x,y) = observation[0]  # agent current location , before action
        if observation[1] == True:
            self.S[x,y] = 1
        if observation[2] == True:
            self.B[x,y] = 1


        nbs = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        index = np.argmax([self.Q[min(x + 1,self.X[-1]), y], self.Q[max(x - 1,0), y], self.Q[x, min(self.Y[-1],y + 1)], self.Q[x, max(0,y - 1)]])
        (nx, ny) = nbs[index]

        if observation[1]==True and observation[3]>0:
            return np.random.randint(5, 9)
        elif ny-y == 1:
            return 1
        elif ny-y == -1:
            return 2
        elif nx-x == -1:
            return 3
        elif nx-x == 1:
            return 4
        else:
            return np.random.randint(1,9)


    def reward(self, observation, action, reward):  # blue reward is score and updated agent position

        self.Action.append(action)
        self.counts += 1
        # self.gridsize = (max(4,observation[0][0]), max(4,observation[0][1]))
        (x, y) = observation[0]
        if action == 1:
            (x,y)= (x,y+1)
        elif action == 2:
            (x,y)= (x,y-1)
        elif action == 3:
            (x,y)= (x-1,y)
        elif action == 4:
            (x,y)= (x+1,y)

        x = max(0, min(self.gridsize[0]-1, x))
        y = max(0, min(self.gridsize[1]-1, y))
        # now (x,y) is updated


        nbs = [ (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        next_Q = max([self.Q[min(x + 1,self.X[-1]), y], self.Q[max(x - 1,0), y], self.Q[x, min(self.Y[-1],y + 1)], self.Q[x, max(0,y - 1)]])
        self.Q[x,y] = reward + l * next_Q


        # # ------------------------------
        # S=[]
        # B=[]
        #
        # # nb = np.zeros((self.e_x,self.e_y))
        # for [a,b] in self.cases:
        #     ele = [[max(a-1,0),b],[min(6,a+1),b],[a,max(b-1,0)],[a,min(6,b+1)]]
        #     ele = [ele[i] for i in range(len(ele)) if ele[i] not in ele[:i]]
        #     ele = np.array(ele)    # distinct elements
        #     # ss = sum([self.S[ele[i]] for i in range(len(ele))])
        #     bb = sum([self.B[ele[i]] for i in range(len(ele))])
        #     # S.append(ss[0,0])
        #     B.append(bb[0, 0])
        # # S = np.reshape(S, (self.e_x,self.e_y))
        # B = np.reshape(B, (7,7))
        #
        # for [a, b] in self.cases:
        #     if B[a,b] >= 3:
        #         self.Q[a, b] = -10
        #     elif (a,b) == (0,0) or (0,-1) or (-1,0) or (-1,-1):
        #         if B[a,b] >= 2:
        #             self.Q[a, b] = -10
        #
        #

Agent = Q_learning


# python main.py --ngames 1000 --niter 100 --batch 200 --verbose
# python main.py --ngames 1000 --niter 100 --batch 200
