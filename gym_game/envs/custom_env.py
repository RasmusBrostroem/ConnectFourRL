import gym
from gym import spaces
import numpy as np
from gym_game.envs.connectFour import connect_four
import pygame as pg

pg.init()

class CustomEnv(gym.Env):
    def __init__(self) -> None:
        self.game = connect_four()
        self.player = 1
        self.action_space = spaces.Discrete(self.game.columns)
        self.observation_space = spaces.Box(low = -1, high = 1, shape = (self.game.rows,self.game.columns), dtype = np.int0)

    def reset(self):
        '''
        Resets the game in the environment and returns the observed game state
        '''
        del self.game
        self.game = connect_four()
        obs = self.game.return_board()
        return obs
    
    def step(self, action: int):
        '''
        Takes a step in the environment by a player choosing an action

        Input
            - action (int): the column that the player wants to place a piece in
        Returns
            - Obs: the state of the game after the action
            - reward: the reward for the player to choose the given action
            - done: boolean that tells if the game is over or not
        '''
        legal = self.game.is_legal(action)
        self.game.place_piece(action, self.player)
        reward = self.game.evaluate(self.player, legal)
        done = self.game.is_done(legal)
        obs = self.game.return_board()
        return obs, reward, done, {}

    def render(self, mode="human", close=False):
        '''
        Renders the current state of the game, so the viewer can watch the game play out
        '''
        self.game.draw_board()
    
    def configurePlayer(self, newPlayer):
        self.player = newPlayer
