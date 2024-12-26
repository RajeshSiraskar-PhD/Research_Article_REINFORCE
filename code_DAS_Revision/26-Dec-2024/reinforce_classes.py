# ----------------------------------------------------------------------------------------------------
# 25-Dec-2024: Sensitivity Analysis
#
# ----------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical, Normal
import torch.nn as nn
import torch.optim as optim

### Network class - function approximator
# - Simple one layer MLP

class PolicyNetwork(nn.Module):
    
    # Step 1: Define the network architecture
    def __init__(self, lr, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        
        # 1.1. Define network architecture
        layers = [
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        ]
            
        # layers = [
        #     nn.Linear(input_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, output_dim),
        # ]
        
        # 1.3. Assemble the network and this becomes our "model" i.e function approximation (i.e. "model") 
        self.model = nn.Sequential(*layers)
        
        # 1.2. Adam optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    # Step 2: Feed-forward algo. PyTorch handles back-prop for us, but feed-forward we must provide
    def feed_forward(self, state):
        # Probability distribution (pd) parameters
        pdparam = self.model(state)
        return (pdparam)

### Agent class
# - This will also "hold" the policy network, defined above, by class PolicyNetwork
# - Other agent specific activities:
#     - Learn the "policy" using the "policy network" above
#     - Decide what action to take
#     - On-policy type - so discard previous experiences 

class Agent():
    def __init__(self, input_dim, n_actions, alpha, gamma):
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        
        self.log_probs = []
        self.rewards = []
        self.pd = None
        # Create the policy network
        self.policy_network = PolicyNetwork(self.alpha, self.input_dim, self.n_actions)
                
        # On-policy, so discard previous experiences. Empty buffer
        self.onpolicy_reset()
        
        # Call training loop
        # self.learn()
        
    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []
    
    def act(self, state):
        ## Continous action - use Normal pd
        ## pd = Normal(loc=pdparams[0], scale=pdparams[1]) # probability distribution
        
        x = torch.from_numpy(state.astype(np.float32)) # Convert to tensor
        pdparam = self.policy_network.feed_forward(x) # forward pass
        pd = Categorical(logits=pdparam) # probability distribution
        ## Note: 
        # 1. self.pd.probs = prob. distribution of all actions
        # 2. Sum of all possible action probabilities = 1.0
        action = pd.sample() # pi(a|s) in action via pd
        log_prob = pd.log_prob(action) # log prob of pi(a|s)
        self.log_probs.append(log_prob) # store for training
        return (action.item())
    
    ## predict is the function used by Stable-Baselines
    ## We simply call act()
    def predict(self, state):
        predicted_action = self.act(state)
        next_state = None
        return predicted_action, next_state
    
    def learn(self):
        # Inner gradient-ascent loop
        T = len(self.rewards) # Length of a trajectory
        returns = np.empty(T, dtype=np.float32)
        future_returns = 0.0

        # Compute returns
        for t in reversed(range(T)):
            future_returns = self.rewards[t] + self.gamma*future_returns
            returns[t] = future_returns
            
        returns = torch.tensor(returns)
        log_probs = torch.stack(self.log_probs)
        
        loss = torch.sum(- log_probs*returns) # Compute gradient term. Negative for maximizing
        self.policy_network.optimizer.zero_grad()
        loss.backward() # backpropogate and compute gradients
        self.policy_network.optimizer.step() # gradient ascent, update the weights
        return (loss)