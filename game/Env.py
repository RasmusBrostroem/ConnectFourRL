import numpy as np
from game.connectFour import connect_four
import pygame as pg
import random

pg.init()


class Env():
    def __init__(self, _player1, _player2) -> None:
        self.game = connect_four()
        self.player1 = _player1
        self.player2 = _player2
        #currentPlayer controls whether it is player1 or player2 playing
        self.currentPlayer = None

    # def reset(self) -> None:
    #     '''
    #     Resets the game in the environment
    #     '''
    #     self.game.restart()

    def game_state(self, col_choice: int) -> str:
        '''
        Returns the game state seen for the perspective for self.currentPlayer.
        Game state is defined as one of the following:
        "win" - if col_choice lead to a win
        "tie" - if col_choice lead to a tie
        "illegal" - if col_choice is an illegal move
        "notEnded" - if col_choice didn't lead to the game ending
        '''
        if self.game.winning_move():
            return "win"
        elif self.game.is_tie():
            return "tie"
        elif not self.game.is_legal(column=col_choice):
            return "illegal"
        else:
            return "notEnded"
    
    def step(self):
        '''
        Takes a step in the environment by a player choosing an action

        Input
            - action (int): the column that the player wants to place a piece in
        Returns
            - Obs: the state of the game after the action
            - reward: the reward for the player to choose the given action
            - done: boolean that tells if the game is over or not
        '''
        col_choice = self.currentPlayer.select_action(self.game.return_board())

        # Place piece in column for current player
        self.game.place_piece(col_choice, self.currentPlayer.playerPiece)

        # Check state of game and give state to agent for him to assign reward
        game_state = self.game_state(col_choice=col_choice)
        self.currentPlayer.assign_reward(game_state)
        #Handle the situation where  
        if game_state == "win":
            if self.currentPlayer is self.player1:
                self.player2.assign_reward("loss")
            else:
                self.player1.assign_reward("loss")
        elif game_state == "tie": ###################################################### HERE TODO
            pass
        # Give reward to the agent that made the choise

        # Check state of game
        reward = self.game.evaluate(self.player, legal)
        done = self.game.is_done(legal)
        return reward, done, {}

    def render(self, reward = 0, mode="human", close=False):
        '''
        Renders the current state of the game, so the viewer can watch the game play out
        '''
        self.game.draw_board(reward)
        pg.time.wait(500)
    
    def change_player(self) -> None:
        if self.currentPlayer is self.player1:
            self.currentPlayer = self.player2
        else:
            self.currentPlayer = self.player1

    def play_game(self):
        self.currentPlayer = random.choice([self.player1, self.player2])


    # def configureRewards(self, win, loss, tie, illegal):
    #     '''
    #     Function to change the standard rewards in the game to something new
    #     '''
    #     self.win = win
    #     self.loss = loss
    #     self.tie = tie
    #     self.illegal = illegal

