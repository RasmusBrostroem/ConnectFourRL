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
import torch.optim as optim
import numpy as np
import random
from types import SimpleNamespace

class Player():
    '''
    A player class that will act like a template for the rest of the players/agents.
    This player is a random player that just chooses a random valid column to place its piece in
    '''
    def __init__(self, player_piece: int, **kwargs) -> None:
        self.params = {
            "win_reward": 1,
            "loss_reward": -1,
            "tie_reward": 0.5,
            "illegal_reward": -5,
            "not_ended_reward": 0,
            "reward_decay": 0.8,
            "device": "cpu"
        }
        self.params.update(kwargs)  

        self.playerPiece = player_piece
        self.device = self.params["device"]
        self.gamma = self.params["reward_decay"]

        self.saved_log_probs = []
        self.game_succes = [] # True if win or tie, false if lose or illegal
        self.probs = []
        self.rewards = []
    
    def select_action(self, board: np.matrix, legal_moves: list = []) -> int:
        '''
        Return a random valid column from the board to place the piece in
        '''
        return random.choice([col for col, val in enumerate(board[0]) if val == 0])

    def calculate_rewards(self) -> None:
        final_reward = self.rewards[-1]
        for i, val in enumerate(reversed(self.rewards)):
            if val != 0 and i != 0:
                break
            
            weighted_reward = self.gamma**i * final_reward
            self.rewards[len(self.rewards)-(i+1)] = weighted_reward
            
            # Assigns the game_success (false if loss or illegal, true if tie or win) for all moves played in a game
            if final_reward == self.params["loss_reward"] or final_reward == self.params["illegal_reward"]:
                self.game_succes[len(self.game_succes)-(i+1)] = False
            else:
                self.game_succes[len(self.game_succes)-(i+1)] = True
    
    def reset_rewards(self) -> None:
        """NOTE: Consider how this functions in regard to logging.
        The challenge is that we don't necessarily want to update and log
        at the same time. When is it okay to delete the different lists?
        """
        del self.game_succes[:]
        del self.probs[:]
        del self.rewards[:]
        del self.saved_log_probs[:]

    def update_agent(self, optimizer = None) -> None:
        #Delete lists after use
        del self.rewards[:]
        del self.saved_log_probs[:]

    def load_params(self, path: str) -> None:
        pass

    def save_params(self, path: str) -> None:
        pass




class DirectPolicyAgent(nn.Module, Player):
    '''
    NOTE: This class initialises with the same keyword arguments as the Player
    class.
    '''
    def __init__(self, **kwargs):
        Player.__init__(self, **kwargs)
        nn.Module.__init__(self)
        self.L1 = nn.Linear(42, 200)
        self.L2 = nn.Linear(200, 300)
        self.L3 = nn.Linear(300, 100)
        self.L4 = nn.Linear(100, 100)
        self.final = nn.Linear(100, 7)

        
    
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

    def select_action(self, board, legal_moves):
        board = board * self.playerPiece
        board_vector = torch.from_numpy(board).float().flatten()
        board_vector = board_vector.to(self.device)
        probs = self.forward(board_vector)
        move = Categorical(probs.to("cpu"))
        action = move.sample()
        if legal_moves and action not in legal_moves:
            # Re-scale probabilities for the legal columns and draw one of the
            #   legal columns
            legal_probs = [probs[col] for col in legal_moves]
            legal_probs = np.divide(legal_probs, sum(legal_probs))
            action = torch.tensor(random.choices(legal_moves, legal_probs)[0])
            # action = torch.tensor(random.choice(legal_moves)) # old approach

        self.saved_log_probs.append(move.log_prob(action))
        self.probs.append(probs[action])
        return action.to("cpu")

    def update_agent(self, optimizer) -> None:
        """NOTE: consider how and when the loss should be logged.

        Args:
            optimizer (torch.optim.Optimizer): _description_
        """
        loss = [-log_p * r for log_p, r in zip(self.saved_log_probs,
                                               self.rewards)]

        loss = torch.stack(loss).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del self.rewards[:]
        del self.saved_log_probs[:]
        # save loss to agent for logging

        # return loss.detach().numpy()

    def save_agent():
        # Figure out what is necessary to save.
        # Do we need to save the entire agent?
        #   (and will it then be possible to load only parameters)
        # TODO: check up on torch docs
        pass
    def load_agent():
        pass
        

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