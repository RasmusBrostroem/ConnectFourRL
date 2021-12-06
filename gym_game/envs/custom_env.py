import gym
from gym import spaces
import numpy as np
from numpy.core.fromnumeric import shape
from gym_game.envs.connectFour import connect_four

class CustomEnv(gym.Env):
    def __init__(self) -> None:
        self.game = connect_four()
        self.action_space = spaces.Discrete(self.game.columns)
        self.observation_space = spaces.Box(low = -1,high = 1,shape = (self.game.rows,self.game.columns), dtype = np.int0)
