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
import neptune.new as neptune

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
            "gamma": 0.8,
            "device": "cpu"
        }
        self.params.update(kwargs)  

        self.playerPiece = player_piece
        self.device = self.params["device"]

        # Parameters used for logging to neptune
        self.neptune_id = ""
        self.stats = {
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "illegals": 0,
            "games": 0,
            "probs_succes_sum": 0,
            "moves_succes_total": 0,
            "probs_failure_sum": 0,
            "moves_failure_total": 0,
            "loss_sum": 0
        }
        self.total_games = 0
        self.probs = []

        # Parameters used for updating agent
        self.saved_log_probs = []
        self.rewards = []
        self.gamma = self.params["gamma"]
    
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
    
    # def reset_rewards(self) -> None:
    #     """NOTE: Consider how this functions in regard to logging.
    #     The challenge is that we don't necessarily want to update and log
    #     at the same time. When is it okay to delete the different lists?
    #     """
    #     del self.rewards[:]
    #     del self.saved_log_probs[:]

    def update_agent(self, optimizer = None) -> None:
        #Delete lists after use
        del self.rewards[:]
        del self.saved_log_probs[:]

    def load_params(self, path: str) -> None:
        pass

    def save_params(self, path: str) -> None:
        pass

    def update_stats(self) -> None:
        final_reward = self.rewards[-1]

        self.stats["games"] += 1
        self.total_games += 1
        if final_reward == self.params["loss_reward"]:
            self.stats["losses"] += 1
        elif final_reward == self.params["win_reward"]:
            self.stats["wins"] += 1
        elif final_reward == self.params["tie_reward"]:
            self.stats["ties"] += 1
        elif final_reward == self.params["illegal_reward"]:
            self.stats["illegals"] += 1
        
        if final_reward == self.params["loss_reward"] or final_reward == self.params["illegal_reward"]:
            self.stats["probs_failure_sum"] += sum(self.probs)
            self.stats["moves_failure_total"] += len(self.probs)
        else:
            self.stats["probs_succes_sum"] += sum(self.probs)
            self.stats["moves_succes_total"] += len(self.probs)
        
        del self.probs[:]
    
    def log_params(self, neptune_run: neptune.Run) -> None:
        self.neptune_id = neptune_run._short_id
        neptune_run[f"player{self.playerPiece}/params"] = self.params

    def log_stats(self, neptune_run: neptune.Run) -> None:
        folder_name = f"player{self.playerPiece}/metrics"
        neptune_run[folder_name + "/winrate"].log(self.stats["wins"]/self.stats["games"])
        neptune_run[folder_name + "/lossrate"].log(self.stats["losses"]/self.stats["games"])
        neptune_run[folder_name + "/tierate"].log(self.stats["ties"]/self.stats["games"])
        neptune_run[folder_name + "/illegalrate"].log(self.stats["illegals"]/self.stats["games"])

        neptune_run[folder_name + "/loss_sum"].log(self.stats["loss_sum"])
        try:
            neptune_run[folder_name + "/averagePropSucces"].log(self.stats["probs_succes_sum"]/self.stats["moves_succes_total"])
            neptune_run[folder_name + "/averagePropFailure"].log(self.stats["probs_failure_sum"]/self.stats["moves_failure_total"])
        except ZeroDivisionError:
            pass

        self.stats = dict.fromkeys(self.stats, 0) # Sets all values back to zero

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
            # legal columns
            # legal_probs = [probs[col].detach().numpy() for col in legal_moves]
            # print(legal_probs)
            # legal_probs = np.divide(legal_probs, sum(legal_probs))
            # action = torch.tensor(random.choices(legal_moves, legal_probs)[0])
            action = torch.tensor(random.choice(legal_moves)) # old approach

        self.saved_log_probs.append(move.log_prob(action))
        self.probs.append(probs[action].detach().numpy())
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
        self.stats["loss_sum"] += loss.detach().numpy()

        # return loss.detach().numpy()

    def save_agent(self):
        # Figure out what is necessary to save.
        # Do we need to save the entire agent?
        #   (and will it then be possible to load only parameters)
        # TODO: check up on torch docs
        pass
    def load_agent(self):
        pass

class DirectPolicyAgent_large(DirectPolicyAgent):
    def __init__(self, **kwargs):
        DirectPolicyAgent.__init__(self, **kwargs)
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
    def __init__(self, **kwargs):
        DirectPolicyAgent.__init__(self, **kwargs)
        self.L1 = nn.Linear(42, 300)
        self.final = nn.Linear(300, 7)
    
    def forward(self, x):
        x = self.L1(x)
        x = F.relu(x)
        x = self.final(x)
        return F.softmax(x, dim=0)

class HumanPlayer(Player):
    '''Let user play the game using console input.
    NOTE: This class initialises with the same keyword arguments as the Player
    class.
    '''
    def __init__(self, **kwargs):
        Player.__init__(self, **kwargs)

    def select_action(self, board: np.matrix, legal_moves: list = []) -> int:
        """Ask for user input to choose a column.

        Args:
            board (np.matrix): The current game board
            legal_moves (list, optional): List of legal moves. Defaults to [].
                This argument is not used by the function, but is included
                since every select_action method needs to have the argument.

        Returns:
            int: The column to place the piece in, 0-indexed.
        """
        # Calculating legal_cols since legal_moves may be an empty list
        legal_cols = [col for col, val in enumerate(board[0]) if val == 0]
        chosen_col = int(input("Choose column: ")) - 1
        while chosen_col not in legal_cols:
            printable_legals = [col+1 for col in legal_cols] # 1-indexed
            print(f"Illegal column. Choose between {printable_legals}.")
            chosen_col = int(input("Choose column: ")) - 1
        return chosen_col
