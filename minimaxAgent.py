# import gym
# noinspection PyUnresolvedReferences
# import gym_game
import random
import numpy as np

from connectFourTest import connect_four


class MinimaxAgent:
    def __init__(self, max_depth=0):
        self.game = connect_four(win=1, loss=-1, tie=0)
        self.max_depth = max_depth
        # self.game = gym.make('ConnectFour-v0')

    def select_action(self, board, legal_choices):
        board = np.flip(np.copy(board), 0)
        best_score = -10
        best_col = None
        possible_ties = []
        for col in legal_choices:
            current_state = self.update_board(board, col, 1)
            score = self.minimax(current_state, 0, False)
            board = self.game.remove_piece(col)
            if score is not None:
                if score == 0 and score >= best_score:
                    possible_ties.append(col)
                    best_score = score
                    best_col = col
                elif score > best_score:
                    best_score = score
                    best_col = col

        if best_score == 0:
            return random.choice(possible_ties)
        return best_col

    def minimax(self, board, depth, maximizing):
        self.game.board = board

        if self.game.winning_move():
            if not maximizing:
                return self.game.win/(depth+1)
            else:
                return self.game.loss/(depth+1)
        if self.game.is_tie():
            return self.game.tie
        if depth > self.max_depth:
            return 0

        if maximizing:
            best_score = None
            for col in range(self.game.columns):
                if board[self.game.rows-1][col] == 0:
                    board = self.update_board(board, col, 1)
                    score = self.minimax(board, depth+1, False)
                    board = self.game.remove_piece(col)
                    if score is not None:
                        if best_score is None:
                            best_score = score
                        elif score > best_score:
                            best_score = score
            return best_score
        else:
            best_score = None
            for col in range(self.game.columns):
                if board[self.game.rows - 1][col] == 0:
                    board = self.update_board(board, col, -1)
                    score = self.minimax(board, depth+1, True)
                    board = self.game.remove_piece(col)
                    if score is not None:
                        if best_score is None:
                            best_score = score
                        elif score < best_score:
                            best_score = score
            return best_score

    @staticmethod
    def update_board(current_state, choice_col, player):
        for i, row in enumerate(current_state):
            if row[choice_col] == 0:
                current_state[i][choice_col] = player
                break
        return current_state
