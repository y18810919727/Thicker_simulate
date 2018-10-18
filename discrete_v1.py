import numpy as np
import pdb
import sys

from io import StringIO
from gym.envs.toy_text import discrete
legal_range = 4


# make A closer to B by A + [1 or 2 or -1 or -2]
#   state    reward.   dowe
#    A=B        -1       True
#   |A-B|>=5  -100    True
#    else       -1         False
class Thicker(discrete.DiscreteEnv):
    metadata = {'render.models':['human','ansi']}

    def _limit_state(self,state):
        state = (state[0],min(state[1],self.shape[1]-1))
        state = (state[0],max(state[1],0))
        return state

    def _calculate_transition_prob(self,state,action):
        state = (state[0],state[1]+action)
        state = self._limit_state(state)
        new_state = np.ravel_multi_index(state,self.shape)
        reward = -100.0 if self._ilegal[state] else -1.0
        is_done = self._ilegal[state] or state[0] == state[1]
        return [(1.0, new_state, reward, is_done)]

    def __init__(self):
        self.shape = (20,20)
        nS = np.prod(self.shape)
        nA = 5
        actions = [-2,-1,0,1,2]
        self._ilegal = np.zeros(self.shape,dtype=bool)
        for goal in range(self.shape[0]):
            self._ilegal[goal,0:max(0,goal-legal_range)] = True
            self._ilegal[goal,min(goal+legal_range+1,self.shape[1]):self.shape[1]] = True
        # calculate transition probabilities
        P = {}
        for s in range(nS):
            P[s] = {a: [] for a in range(nA)}
            state = np.unravel_index(s,self.shape)
            for id,action in enumerate(actions):
                P[s][id] = self._calculate_transition_prob(state,action)

        isd = np.zeros(nS)
        legal_cnt = np.sum(self._ilegal==False)
        #np.ravel_multi_index()

        #        pdb.set_trace()
        position_legal = [self._ilegal == False]
        #print(position_legal)
        # isd[self._ilegal==False] = 1.0/legal_cnt
        for s in range(nS):
            #print(np.unravel_index(s, self.shape))
            if not self._ilegal[np.unravel_index(s, self.shape)]:
                #print(np.unravel_index(s, self.shape))
                isd[s] = 1.0/legal_cnt
        #print(np.sum(isd))
        super(Thicker,self).__init__(nS, nA, P, isd)

    def render(self, mode='human',close=False):
        self._render(mode,close)

    def _render(self, mode='human',close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        #print(self.s)
        outfile.write('(%i,%i)'%(np.unravel_index(self.s, self.shape)))
        outfile.write('\n')



def test():
    env = Thicker()
    print(env)
    print('reset')
    for x in range(100):
        env.reset()
        env.render()
    print('step')
    env.render()
    for t in range(20):
        act =np.random.randint(5)
        print(act)
        print(env.step(act))
        env.render()

if __name__ == '__main__':
    test()

