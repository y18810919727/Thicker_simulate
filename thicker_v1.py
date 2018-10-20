import numpy as np
import sys
import pandas
import random
import gym
from io import StringIO
from gym import spaces, logger
from gym.utils import seeding
import math
class Thicker(gym.Env):
    """
    Description:
        A thicker receives low concentration paste(about 20-30%), and the sand volume
        will be accumulated with under feeding. The under concentration is equal to
        2*(sand/(sand+water)), the goal of under concentration  is changed.

    Observation:
    Type Box(5)
    Num     Observation     Min     Max
    0       in_size         1.8     2.2
    1       in_con          0.25    0.35
    2       under_size      0.5     1.5
    3       under_con       -Inf    Inf
    4       sand            -Inf    Inf



    Actions:
        Type: Discrete(5)
        Num     Action
        0       nothing
        1       add under_size -0.1
        2       add under_size 0.1
        3       add under_size -0.2
        4       add under_size 0.2

    Done:
        if abs(under_con-60) < 0.005:
            Done = True

    Reward:
        -1


    """


    def __init__(self):

        self.sand = random.randint(26,34)
        self.volume = 100
        self.under_size = random.uniform(0.5, 1.5)
        self.actions = [0,-0.1,0.1,-0.2,0.2]
        self.get_noise = self.noise_in()
        self.in_size,self.in_con = next(self.get_noise)
        self.goal = 0.6
        self.under_con = self.get_under_con()
        self.act_size = len(self.actions)
        self.observe_size = self.get_state().shape[0]
        self.time = 0
        self.log = []
        high = np.array([
            2.2,
            0.35,
            1.5,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])
        low = np.array([
            1.8,
            0.25,
            0.5,
            np.finfo(np.float32).min,
            np.finfo(np.float32).min])
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low,high,dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.state = np.array([self.in_size,self.in_con, \
                               self.under_size,self.under_con,self.sand])
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



    def noise_in(self):
        while True:
            random.seed()
            in_size,in_con = random.uniform(1.8,2.2),random.uniform(0.25,0.35)
            for _ in range(0,random.randint(200,300)):
                yield in_size,in_con

    def get_under_con(self):
        return float(self.sand)/self.volume*2

    def cal_reward(self):
        #return -math.fabs(self.get_under_con()-self.goal)*100
        return -1


    def new_noise(self):
        self.in_size,self.in_con = next(self.get_noise)

    def get_state(self):
        state = np.array([self.in_size,self.in_con,self.volume,self.sand,self.under_size,self.get_under_con(),self.goal],dtype=float)
        return state

    def reset(self):
        self.__init__()
        self.state = np.array([self.in_size,self.in_con, \
                               self.under_size,self.under_con,self.sand])
        return np.array(self.state)

    def write_log(self):
        tmp = np.concatenate(self.log)
        return pandas.DataFrame(tmp)

    def step(self,action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.in_size, self.in_con, self.under_size, _, self.sand = self.state
        self.time += 1
        self.under_size += self.actions[action]
        self.under_size = np.clip(self.under_size, 0.5, 1.5)
        t_len = 100
        for i in range(t_len):
            dt = (1.0/t_len)
            self.sand -= self.under_size*self.get_under_con()*dt
            self.sand += self.in_size*self.in_con*dt
            self.volume -= self.under_size*dt
            self.volume += self.in_size*dt
            self.under_con = self.get_under_con()
            if self.volume > 100:
                self.volume = 100
        reward = self.cal_reward()
        done = False
        if math.fabs(self.under_con-self.goal)<0.005:
            done = True
        self.state = np.array([self.in_size,self.in_con, \
                               self.under_size,self.under_con,self.sand])
        return np.array(self.state),reward,done,{}

        #log_state = np.array(self.get_state().tolist()+[reward,done])
        #self.log.append(log_state[np.newaxis,:])
        #self.new_noise()
        #return self.get_state(),reward,done,None
    def render(self, mode='human'):
        outfile = StringIO if mode == 'ansi' else sys.stdout

        outfile.writelines('In_size\tIn_con\tun_size\tun_con\tsand\n')
        outfile.writelines('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % (tuple(self.state)))
        outfile.write('\n')


if __name__ == '__main__':
    env = Thicker()
    env.render()
    env.reset()
    env.render()
    env.step(0)
    env.render()
    env.step(1)
    env.render()
    env.step(2)
    env.render()
    env.step(3)
    env.render()
    env.step(4)
    env.render()


