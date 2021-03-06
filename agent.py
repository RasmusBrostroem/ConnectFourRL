#Links for implementing agents
'''
Tic-tac-toe with policy gradient desent
https://medium.com/@carsten.friedrich/part-8-tic-tac-toe-with-policy-gradient-descent-da2496defc45
https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_4_policy_gradient.pdf
'''

'''
Reward function is built based on these point systems:
- Winning move: 1
- losing action: -1
- draw action: 0
- illegal move: -10
- non ending action: l^t * Final_Reward
    Where l is a constant between 0 and 1, t is the number of moved the action happened
    before the end and Final_reward being one the the four rewards above
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random


class DirectPolicyAgent(nn.Module):
    def __init__(self, device, gamma = 0.99):
        super(DirectPolicyAgent, self).__init__()
        self.L1 = nn.Linear(42, 200)
        self.L2 = nn.Linear(200, 300)
        self.L3 = nn.Linear(300, 100)
        self.L4 = nn.Linear(100, 100)
        self.final = nn.Linear(100, 7)

        self.device = device
        self.gamma = gamma

        self.saved_log_probs = []
        self.game_succes = [] # True if win or tie, false if lose or illegal
        self.probs = []
        self.rewards = []
    
    def forward(self, x):
        x = self.L1(x)
        x = F.relu(x)
        x = self.L2(x)
        x = F.relu(x)
        x = self.L3(x)
        x = F.relu(x)
        x = self.L4(x)
        x = F.relu(x)
        x = self.final(x)
        return F.softmax(x, dim=0)

    def select_action(self, x, legal_moves):
        x = x.copy()
        x = torch.from_numpy(x).float().flatten()
        x = x.to(self.device)
        probs = self.forward(x)
        m = Categorical(probs.to("cpu"))
        action = m.sample()
        if legal_moves and action not in legal_moves:
            action = torch.tensor(random.choice(legal_moves))

        self.saved_log_probs.append(m.log_prob(action))
        self.probs.append(probs[action])
        return action.to("cpu")

    def calculate_rewards(self, env):
        final_reward = self.rewards[-1]
        for i, val in enumerate(reversed(self.rewards)):
            if val != 0 and i != 0:
                break
            
            weighted_reward = self.gamma**i * final_reward
            self.rewards[len(self.rewards)-(i+1)] = weighted_reward
            if final_reward == env.game.loss or final_reward == env.game.illegal:
                self.game_succes[len(self.game_succes)-(i+1)] = False
            else:
                self.game_succes[len(self.game_succes)-(i+1)] = True

class DirectPolicyAgent_large(DirectPolicyAgent):
    def __init__(self, device, gamma=0.99):
        super().__init__(device, gamma=gamma)
        self.L1 = nn.Linear(42, 300)
        self.L2 = nn.Linear(300, 500)
        self.L3 = nn.Linear(500, 1000)
        self.L4 = nn.Linear(1000, 600)
        self.L5 = nn.Linear(600, 200)
        self.L6 = nn.Linear(200, 100)
    
    def forward(self, x):
        x = self.L1(x)
        x = F.relu(x)
        x = self.L2(x)
        x = F.relu(x)
        x = self.L3(x)
        x = F.relu(x)
        x = self.L4(x)
        x = F.relu(x)
        x = self.L5(x)
        x = F.relu(x)
        x = self.L6(x)
        x = F.relu(x)
        x = self.final(x)
        return F.softmax(x, dim=0)

class DirectPolicyAgent_mini(DirectPolicyAgent):
    def __init__(self, device, gamma=0.99):
        super().__init__(device, gamma=gamma)
        self.L1 = nn.Linear(42, 300)
        self.final = nn.Linear(300, 7)
    
    def forward(self, x):
        x = self.L1(x)
        x = F.relu(x)
        x = self.final(x)
        return F.softmax(x, dim=0)