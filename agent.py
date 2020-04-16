import numpy as np
from collections import defaultdict

class Agent:
    
    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 1.0
        self.alpha = .05
        self.gamma = 1.0
        
    def __update_Q(self, current_Q, next_Q, reward):
        """ updates the action-value function estimate using the most recent time step """
        return current_Q + (self.alpha * (reward + (self.gamma * next_Q) - current_Q))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy_probabilities = np.ones(self.nA) * self.epsilon / self.nA
        policy_probabilities[np.argmax(self.Q[state])] = 1 - self.epsilon + (self.epsilon / self.nA)

        return np.random.choice(np.arange(self.nA), p=policy_probabilities)

    def step(self, state, action, reward, next_state, episode, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.epsilon = 1.0 / episode
        
        self.Q[state][action] = self.__update_Q(self.Q[state][action], np.max(self.Q[next_state]), reward)      