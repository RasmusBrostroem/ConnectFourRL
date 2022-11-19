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
        self.gamma = self.params["gamma"]

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

class MinimaxAgent(Player):
    def __init__(self, max_depth=2, **kwargs):
        Player.__init__(self, **kwargs)
        self.max_depth = max_depth

    def select_action(self, board: np.matrix, legal_moves: list = []) -> int:
        """Selects a column that the agent is going to place their piece in

        Args:
            board (np.matrix): the current state of the board
            legal_moves (list, optional): the columns that we can legally choose from. Defaults to [].

        Returns:
            int: the column that is choosen to place a piece in
        """
        best_score = -10
        best_col = None
        possible_ties = []

        # Make sure the MinimaxAgent always have legal_moves to choose from
        # The reason for this is that this agent can not make illegal moves
        if not legal_moves:
            legal_moves = [col for col, val in enumerate(board[0]) if val == 0]
        
        for col in legal_moves:
            board = self.update_board(current_state=board, choice_col=col, player_piece=1)
            score = self.minimax(board=board, depth=0, maximizing=False)
            board = self.remove_piece(board=board, column=col)
            if score is not None:
                if score == self.params["not_ended_reward"] and score >= best_score:
                    possible_ties.append(col)
                    best_score = score
                    best_col = col
                elif score > best_score:
                    best_score = score
                    best_col = col

        if best_score == self.params["not_ended_reward"]:
            return random.choice(possible_ties)
        return best_col

    def minimax(self, board: np.matrix, depth: int, maximizing: bool) -> float:
        """Runs the minimax algorithm on the board

        Args:
            board (np.matrix): the current state of the board
            depth (int): the current depth of the minimax algorithm
            maximizing (bool): true if it is the maximizing player, and false if minimizing player

        Returns:
            float: the best score optained within before reacing the 'max_depth'
        """
        if self.winning_move(board=board):
            if not maximizing:
                return self.params["win_reward"]/(depth+1)
            else:
                return self.params["loss_reward"]/(depth+1)
        if self.is_tie(board=board):
            return self.params["tie_reward"]
        if depth > self.max_depth:
            return self.params["not_ended_reward"]

        if maximizing:
            best_score = None
            for col in range(board.shape[1]):
                if board[0][col] == 0: # Checks if the column is not filled
                    board = self.update_board(current_state=board, choice_col=col, player_piece=self.playerPiece)
                    score = self.minimax(board=board, depth=depth+1,maximizing=False)
                    board = self.remove_piece(board=board, column=col)
                    if score is not None:
                        if best_score is None:
                            best_score = score
                        elif score > best_score:
                            best_score = score
            return best_score
        else:
            best_score = None
            for col in range(board.shape[1]):
                if board[0][col] == 0: # Checks if the column is not filled
                    board = self.update_board(current_state=board, choice_col=col, player_piece=self.playerPiece*-1)
                    score = self.minimax(board=board, depth=depth+1, maximizing=True)
                    board = self.remove_piece(board=board, column=col)
                    if score is not None:
                        if best_score is None:
                            best_score = score
                        elif score < best_score:
                            best_score = score
            return best_score

    @staticmethod
    def update_board(current_state: np.matrix, choice_col: int, player_piece: int) -> np.matrix:
        """Places the piece on the board, such that the piece is on top in the given column.

        Args:
            current_state (np.matrix): the current state of the board
            choice_col (int): the column that we want to place a piece in
            player_piece (int): the piece that we want to place

        Returns:
            np.matrix: the board with the piece placed in the given column
        """
        board = np.flip(current_state, 0)
        for i, row in enumerate(board):
            if row[choice_col] == 0:
                board[i][choice_col] = player_piece
                break
        return np.flip(board, 0)

    @staticmethod
    def winning_move(board: np.matrix) -> bool:
        """Checks if there is a player with four connected pieces

        Args:
            board (np.matrix): the board that we want to check if there is a winner in

        Returns:
            bool: true if there is a player with four connected pieces, and false if not
        """
        rows, columns = board.shape
        #Check horizontal locations for win
        for c in range(columns-3):
            for r in range(rows):
                winning_sum = np.sum(board[r,c:c+4])
                if winning_sum == 4 or winning_sum == -4:
                    return True
        
        #Check vertical locations for win
        for c in range(columns):
            for r in range(rows-3):
                winning_sum = np.sum(board[r:r+4,c]) 
                if winning_sum == 4 or winning_sum == -4:
                    return True
        
        #Check diagonals for win
        for c in range(columns-3):
            for r in range(rows-3):
                sub_matrix = board[r:r+4,c:c+4]
                #diag1 is the negative slope diag
                diag1 = sub_matrix.trace()
                #diag2 is the positive slope diag
                diag2 = np.fliplr(sub_matrix).trace()
                
                if diag1 == 4 or diag1 == -4 or diag2 == 4 or diag2 == -4:
                    return True
        
        return False

    @staticmethod
    def is_tie(board: np.matrix) -> bool:
        """Checks if the board is filled, which results in a tie

        Args:
            board (np.matrix): the board that we want to check for a tie in

        Returns:
            bool: true if the board is filled, and false if not
        """
        return all([val != 0 for val in board[0]])

    @staticmethod
    def remove_piece(board: np.matrix, column: int) -> np.matrix:
        """Removes the top piece of the board from the choosen column

        Args:
            board (np.matrix): the board that we want to remove a peice from
            column (int): the column that the piece we want to remove is in

        Returns:
            np.matrix: the board with a piece removed from the given column
        """
        for i, row in enumerate(board):
            if row[column] != 0:
                board[i][column] = 0
                break
        return board
