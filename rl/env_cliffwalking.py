'''
Environment for Example 6.6 in Sutton and Barto 2018
'''

import numpy as np
import sys
from gym.envs.toy_text import discrete
import matplotlib.pyplot as plt
import seaborn as sns


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class CliffWalkingEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, wind):
        new_position = np.array(current) + np.array(delta) + np.array(wind)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        reward = self.penalty_walk # -5.0
        if self._cliff[tuple(new_position)]:
            reward = self.penalty_cliff # -200.0
        elif self._goal[tuple(new_position)]:
            reward = self.penalty_goal # 100.0
        else:
            reward = self.penalty_walk #-5.0
        # reward = -100.0 if self._cliff[tuple(new_position)] else -10.0
        is_done = self._cliff[tuple(new_position)] or (tuple(new_position) == (self.length-1,self.width-1))
        if is_done:
            new_state = np.ravel_multi_index((self.length-1,0), self.shape)
        nS = np.prod(self.shape)
        next_state_prob = np.zeros(nS)
        next_state_prob[new_state] = 1.0
        next_state_reward = {}
        next_state_reward[new_state] = reward
        return next_state_prob, next_state_reward, [(1.0, new_state, reward, is_done)]

    def _concat_transition(self, position, transitions):
        assert len(transitions)>0, "No transitions to concatenate"

        delta_position, wind, prob =  transitions[0]
        c_next_state_prob, c_next_state_reward, c_next_states = self._calculate_transition_prob(position, delta_position, wind)
        c_next_state_prob = prob * c_next_state_prob

        for (delta_position, wind, prob) in transitions[1:]:
            next_state_prob, next_state_reward, next_states = self._calculate_transition_prob(position, delta_position, wind)
            c_next_state_prob += prob * next_state_prob
            c_next_state_reward.update(next_state_reward)
            c_next_states += next_states
        
        nS = np.prod(self.shape)
        _c_next_state_reward = np.zeros(nS)
        for i in c_next_state_reward:
            _c_next_state_reward[i] = c_next_state_reward[i]
        c_next_state_reward = _c_next_state_reward

        return c_next_state_prob, c_next_state_reward, c_next_states

    def _random_transition_prob(self):
        nS_2 = np.prod(self.shape_2)
        next_state_prob_2 = np.ones(nS_2) / nS_2 # transition to any state with equal probability
        new_state_2 = np.random.choice(range(nS_2), p=next_state_prob_2)
        # next_state_reward = np.ones(nS_2) * reward # zero reward for all states
        next_state_reward_2 = self.next_state_reward_2
        reward_2 = next_state_reward_2[new_state_2]
        is_done = False
        return next_state_prob_2, next_state_reward_2, [(1.0, new_state_2, reward_2, is_done)]

    def __init__(self, slip, slip_prob, seed=0):
        self.length = 6
        self.width = 6
        self.shape = (self.length, self.width)
        self.shape_2 = (self.length, self.width)

        nS = np.prod(self.shape)
        nS_2 = np.prod(self.shape_2)
        nA = 4

        self.slip = slip
        self.slip_prob = slip_prob

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=np.bool)
        self._cliff[self.length-1, 1:-1] = True

        # Goal Location
        self._goal = np.zeros(self.shape, dtype=np.bool)
        self._goal[self.length-1,-1] = True
        
        # Wind
        self._wind = np.zeros(self.shape)
        self._wind[:, 1:-1] = self.slip

        self.penalty_walk, self.penalty_cliff, self.penalty_goal = -1.0, -100.0, 0.0 # -50.0, -500.0, 100.0
        self.penalty_walk_2 = -1.0 # uniform random number from -1,0 # -50

        np.random.seed(seed)
        self.next_state_reward_2 = np.random.uniform(self.penalty_walk_2, 0, nS_2)

        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = { a : [] for a in range(nA) }
            # slip in direction of current action
            # P[s][UP] = self._concat_transition(position, [([-1, 0], 1-self.slip_prob), ([-1*self.slip,0], self.slip_prob)])
            # P[s][RIGHT] = self._concat_transition(position, [([0, 1], 1-self.slip_prob), ([0, 1*self.slip], self.slip_prob)])
            # P[s][DOWN] = self._concat_transition(position, [([1, 0], 1-self.slip_prob), ([1*self.slip, 0], self.slip_prob)])
            # P[s][LEFT] = self._concat_transition(position, [([0, -1], 1-self.slip_prob), ([0, -1*self.slip], self.slip_prob)])
            # slip downward with wind
            wind = np.array([1, 0])*self._wind[tuple(position)]
            P[s][UP] = self._concat_transition(position, [([-1, 0], [0, 0], 1-self.slip_prob), ([-1, 0], wind, self.slip_prob)])
            P[s][RIGHT] = self._concat_transition(position, [([0, 1], [0, 0], 1-self.slip_prob), ([0, 1], wind, self.slip_prob)])
            P[s][DOWN] = self._concat_transition(position, [([1, 0], [0, 0], 1-self.slip_prob), ([1, 0], wind, self.slip_prob)])
            P[s][LEFT] = self._concat_transition(position, [([0, -1], [0, 0], 1-self.slip_prob), ([0, -1], wind, self.slip_prob)])

        P_2 = {}
        for s in range(nS_2):
            position = np.unravel_index(s, self.shape)
            P_2[s] = { a : [] for a in range(nA) }
            P_2[s][UP] = self._random_transition_prob()
            P_2[s][RIGHT] = self._random_transition_prob()
            P_2[s][DOWN] = self._random_transition_prob()
            P_2[s][LEFT] = self._random_transition_prob()
        
        # We always start in state (3, 0), (0, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((self.length-1,0), self.shape)] = 1.0
        # Index into initial state in vector form
        self.state = np.ravel_multi_index((self.length-1,0), self.shape)
        self.state_2 = np.ravel_multi_index((0,0), self.shape_2)

        self.nS_2 = nS_2
        self.P_2 = P_2
        super(CliffWalkingEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human', close=False):
        self._render(mode, close)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if self.s == s:
                output = " x "
            elif position == (self.length-1,self.width-1):
                output = " T "
            elif self._cliff[position]:
                output = " C "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip() 
            if position[1] == self.shape[1] - 1:
                output = output.rstrip() 
                output += "\n"

            outfile.write(output)
        outfile.write("\n")

    def reset(self):
        # We always start in state (3, 0), (0, 0)
        # self.isd = np.zeros(self.nS)
        # self.isd[np.ravel_multi_index((self.length-1,0), self.shape)] = 1.0
        # start at bottom left at (5,0)
        self.state = np.ravel_multi_index((self.length-1,0), self.shape)
        self.state_2 = np.ravel_multi_index((0,0), self.shape_2)

        return [self.state, self.state_2]
    
    def step(self, action):
        
        next_state_prob, next_state_reward, _ = self.P[self.state][action]
        next_state = np.random.choice(np.arange(len(next_state_prob)), 1, p=next_state_prob).item()

        next_state_prob_2, next_state_reward_2, _ = self.P_2[self.state_2][action]
        next_state_2 = np.random.choice(np.arange(len(next_state_prob_2)), 1, p=next_state_prob_2).item()

        reward = next_state_reward[next_state] + next_state_reward_2[next_state_2]

        next_position = np.unravel_index(next_state, self.shape)
        is_done = False
        # is_done = self._cliff[tuple(next_position)] or (tuple(next_position) == (self.length-1,self.width-1))
        restart = self._cliff[tuple(next_position)] or (tuple(next_position) == (self.length-1,self.width-1))
        if restart:
            next_state = np.ravel_multi_index((self.length-1,0), self.shape)

        self.state = next_state
        self.state_2 = next_state_2

        return [self.state, self.state_2], reward, is_done, {}

    def plot(self, v_value, filename):
        plt.figure()
        ax = sns.heatmap(v_value, annot=True, fmt=".2f", linewidth=0.5)
        ax.figure.savefig(filename)

        return ax

    def plot_actions(self, v_value, actions, filename):
        plt.figure()
        text, data = actions, v_value
        labels = (np.asarray(["{0}\n{1:.2f}".format(t, d) for t, d in zip(text.flatten(), data.flatten())])).reshape(text.shape[0],text.shape[1])
        ax = sns.heatmap(v_value, annot=labels, fmt="", linewidth=0.5)
        ax.figure.savefig(filename)

        return ax