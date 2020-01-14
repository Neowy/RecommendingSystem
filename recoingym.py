import gym
import numpy as np
episode_limit = 1000
class Q_learning():
    def __init__(self,env):
        self.ALPHA = 0.5
        self.EPSILON = 0.3
        self.q_value = np.zeros(4096*12).reshape(4096,12)
        self.ALL = env.AllMaterial
        self.alpha = np.zeros(12)
    def hash(self, alpha):
        num = 0
        for i in range(1, 12):
            if alpha[i] == 1:
                num += 2 ** (11 - 1 - i)
        return int(num)
    def egreedy_action(self,alpha):
        if np.random.binomial(1, self.EPSILON) == 1:
            action = np.random.choice(self.ALL)
        else:
            values_ = self.q_value[self.hash(alpha), :]
            action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
        return action
    def reset(self):
        self.alpha = np.zeros(12)
    def perceive(self,alpha,reward,next_alpha,action):
        values_ = self.q_value[self.hash(next_alpha),:]
        next_action = np.random.choice([action_ for action_,value_ in enumerate(values_) if value_ == np.max(values_)])
        self.q_value[self.hash(alpha),action] += \
            self.ALPHA * (reward + self.q_value[self.hash(next_alpha),int(next_action)] -
                          self.q_value[self.hash(alpha),action])
def main():
    env = gym.make('RecommendingSystem-v1')
    agent = Q_learning(env)
    for episode in range(episode_limit):
        # print('episode: ',episode+1)
        agent.reset()
        env.reset()
        is_end = False
        while is_end!=True:
            action = agent.egreedy_action(env.alpha)
            alpha = env.alpha
            next_alpha,reward,is_end,_ = env.step(action)
            agent.perceive(alpha,reward,next_alpha,action)
    print('optimal policy is: ')
    is_end = False
    env.reset()
    while is_end!=True:
        values_ = agent.q_value[agent.hash(env.alpha),:]
        action = np.random.choice(
            [action_ for action_,value_ in enumerate(values_) if value_ == np.max(values_)])
        env.alpha,reward,is_end,_ = env.step(action)
        print('step:',action)
    print(env.alpha)
main()