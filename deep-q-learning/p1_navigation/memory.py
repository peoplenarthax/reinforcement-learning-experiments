from sum_tree import SumTree
import numpy as np

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
    Prioritized Experience Replay
    Most of the theory for  PER is implemented in memory, here the key
    point is the way we sample the store based on priority (hence the name).
    Everything is heavy relying on the Sum Tree data structure to save a lot of time on keeping the priority sorted.
"""
class Memory(object):  
    def __init__(self, capacity, state_size = 37, epsilon = 0.001, alpha = 0.4, beta = 0.3, beta_increment_per_sampling = 0.001, abs_err_upper = 1):
        self.tree = SumTree(capacity)
        self.epsilon = epsilon  # Avoid 0 priority and hence a do not give a chance for the priority to be selected stochastically 
        self.alpha = alpha  # Vary priority vs randomness. alpha = 0 pure uniform randomnes. Alpha = 1, pure priority
        self.beta = beta # importance-weight-sampling, from small to big to give more importance to corrections done towards the end of the training
        self.beta_increment_per_sampling = 0.001 
        self.abs_err_upper = 1 # clipped abs error
        self.state_size = state_size

    # Save experience in memory
    def store(self, state, action, reward, next_state, done):
        transition = [state, action, reward, next_state, done]
        max_p = np.max(self.tree.tree[-self.tree.capacity:])

        # In case of no priority, we set abs error to 1
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    # Sample n amount of experiences using prioritized experience replay
    def sample(self, n):
        b_idx = np.empty((n,), dtype=np.int32)
        states = np.empty((n, self.state_size))
        actions = np.empty((n,))
        rewards = np.empty((n,))
        next_states = np.empty((n,self.state_size))
        dones = np.empty((n,))
        ISWeights = np.empty((n,)) # IS -> Importance Sampling

        pri_seg = self.tree.total_p / n # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # Increase the importance of the sampling for ISWeights

        # min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight

        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i] = np.power(prob, -self.beta)
            b_idx[i]= idx
            states[i, :] = data[0]
            actions[i] = data[1]
            rewards[i] = data[2]
            next_states[i, :] = data[3]
            dones[i] = data[4]
        
        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)
        ISWeights = torch.from_numpy(np.vstack(ISWeights)).float().to(device)

        return b_idx, states, actions, rewards, next_states, dones, ISWeights

    # Update the priorities according to the new errors
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
            
            
    def __len__(self):
        return self.tree.length()