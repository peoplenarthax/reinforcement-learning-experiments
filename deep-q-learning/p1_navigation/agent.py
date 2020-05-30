import numpy as np
import random
from collections import namedtuple, deque, defaultdict
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from memory import Memory
from network_model import QNetwork


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed = 0, buffer_size = int(1e4), batch_size = 64, gamma = 0.99, tau = 1e-3, lr = 7e-4, update_every = 4):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed,  fc1_units=32, fc2_units=8).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed,  fc1_units=32, fc2_units=8).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = Memory(buffer_size, state_size, alpha = 0.6) # replay buffer size

        # Parameters
        self.batch_size = batch_size # minibatch size
        self.gamma = gamma # discount factor
        self.tau = tau # for soft update of target parameters
        self.update_every = update_every # how often to update the network

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, i_episode):
        # Save experience in replay memory
        self.memory.store(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if (len(self.memory) >= self.batch_size):
            # If enough samples are available in memory, get radom subset and learn
                self.learn(self.gamma, i_episode)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, gamma, episode_n):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        tree_id, states, actions, rewards, next_states, dones, ISWeights = self.memory.sample(self.batch_size)
        # Double DQN
        # Use local network to select max Q for actions in every experience
        Q_expected_next_max = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        # Use gather to get the same actions but from the Q on target network
        Q_targets_next = self.qnetwork_target(next_states).gather(1, Q_expected_next_max) 
        
        # Normal DQN
        # use target network for selecting next Q value
        # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
    
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        dt_errors = Q_targets - Q_expected
        
        self.memory.batch_update(tree_id, (abs(dt_errors) + 1e-5).cpu().detach().numpy().flatten())

        # Compute loss
        loss =  torch.mul(dt_errors.pow(2), ISWeights)
        loss = torch.mean(loss)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)                     

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)