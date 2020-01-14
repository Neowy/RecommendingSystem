import logging
import random
import gym
import numpy as np

logger = logging.getLogger(__name__)
class Recommending(gym.Env):
    # metadata = {
    #     'render.modes':['human','rgb_array']
    # }
    def __init__(self):
        self.episode_limit = 1000
        self.alpha = np.zeros(12)
        self.ALPHA = 0.5
        self.label = 1
        self.AllMaterial = np.arange(1,12)
        self.time = 0
        self.EPSILON = 0.3
        self.T = 6
        self.D = np.zeros(12*12).reshape(12,12)
        self.MS = np.zeros(12*12).reshape(12,12)
        self.D[1][2] = 1
        self.D[1][3] = 1
        self.D[1][4] = 1
        self.D[1][5] = 1
        self.D[1][6] = 1
        self.D[3][7] = 1
        self.D[4][8] = 1
        self.D[4][9] = 1
        self.D[5][11] = 1
        self.D[7][10] = 1
        self.D[8][10] = 1
        self.D[8][11] = 1
        self.D[9][10] = 1
        self.MS[1][2] = 1
        self.MS[1][1] = 1
        self.MS[1][6] = 1
        self.MS[2][2] = 1
        self.MS[3][3] = 1
        self.MS[4][4] = 1
        self.MS[5][5] = 1
        self.MS[6][6] = 1
        self.MS[7][7] = 1
        self.MS[8][8] = 1
        self.MS[9][9] = 1
        self.MS[10][7] = 1
        self.MS[10][8] = 1
        self.MS[10][9] = 1
        self.MS[10][10] = 19
        self.MS[11][11] = 1
    # def hash(self,alpha):
    #     num = 0
    #     for i in range(1,12):
    #         if alpha[i] == 1:
    #             num += 2**(11-1-i)
    #     return int(num)
    def check(self,d):
        self.label = 1
        for i in range(1,12):
            if self.D[i][d] == 1 and self.alpha[i] == 0:
                #print('修的课所对应的技能为',d,'要求的预备技能为',i,'掌握情况为',self.alpha[i])
                self.label = 0
    def step(self,d):
        # alpha = np.array(alpha)
        reward = 0
        # print(alpha)
        next_alpha = self.alpha.copy()
        for i in range(1,12):
            if self.MS[d][i] == 1:
                self.check(i)
                if self.label == 1:#说明预备技能都已经掌握
                    #s = np.random.uniform(0.1,0.3)
                    #if random.random() <= 1-s:
                    #    next_alpha[i] = 1
                    if self.alpha[i] == 0: #说明这个技能还没有学会
                        next_alpha[i] = 1
                        if reward != 0:
                            reward *= 1.5
                        else:
                            reward = 1
        if (self.alpha == next_alpha).all():
            reward = 0
        self.time += 1
        if(self.time == self.T):
            is_terminal = True
        else:
            is_terminal = False
        self.alpha = next_alpha
        return next_alpha,reward,is_terminal,{}
    def reset(self):
        self.alpha = np.zeros(12)
        self.time = 0
        return self.alpha
    def episode(self,q_value):
        self.reset()
        is_end = False
        while is_end!=True:
            if np.random.binomial(1,self.EPSILON)== 1:
                action = np.random.choice(self.AllMaterial)
            else:
                values_ = q_value[self.hash(self.alpha),:]
                action = np.random.choice([action_ for action_,value_ in enumerate(values_) if value_ == np.max(values_)])
            next_alpha,reward,is_end,_ = self.step(action,self.alpha)
            values_ = q_value[self.hash(next_alpha),:]
            next_action = np.random.choice([action_ for action_,value_ in enumerate(values_) if value_ == np.max(values_)])
            q_value[self.hash(self.alpha),action] += \
                self.ALPHA * (reward + q_value[self.hash(next_alpha),int(next_action)] -
                              q_value[self.hash(self.alpha),action])
            self.alpha = next_alpha
    def render(self,mode = 'human'):
        self.alpha = self.reset()
        self.q_value = np.zeros(4096*12).reshape(4096,12)
        ep = 0
        while ep < self.episode_limit:
            # print('episode: ',ep)
            self.episode(self.q_value)
            ep += 1
        self.alpha = np.zeros(12)
        print("optimal policy is:")
        is_end = False
        self.reset()
        while is_end != True:
            values_ = self.q_value[self.hash(self.alpha), :]
            action = np.random.choice(
                [action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
            self.alpha,reward,is_end,_ = self.step(action,self.alpha)
            # print("step:",action,'q_value=',self.q_value[self.hash(self.alpha),action])
            print("step:", action)
        print(self.alpha)