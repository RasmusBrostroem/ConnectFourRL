import numpy as np
import gym
import gym_game
import pygame as pg

import torch
import torch.optim as optim

import random
from agent import DirectPolicyAgent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

agent = DirectPolicyAgent(device)

pg.init()

# Parameters
episodes_per_gen = 10000 # Episodes before new generation
batch_size = 100 #Episodes before param update
learning_rate = 0.0001 # Learning rate
weight_decay = 0.99 # Weight decay for Adam optimizer

# Environment
env = gym.make('ConnectFour-v0')

s = env.reset()
env.configurePlayer(random.choice([-1,1]))

print(type(s))

for move in range(20):
    #env.render()
    #pg.time.wait(2000)

    if env.player == -1:
        choices = env.game.legal_cols()
        action = random.choice(choices)
    else:
        action = agent.select_action(s)

    s, r, done, _ = env.step(action)
    
    if env.player == 1:
        agent.rewards.append(r)
    elif done:
        break
    
    env.configurePlayer(env.player * -1)




