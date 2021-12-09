import numpy as np
import gym
from pygame.constants import K_ESCAPE, KEYDOWN
import gym_game
import pygame as pg
import os
import sys
import keyboard

import torch
import torch.optim as optim

import random
from agent import DirectPolicyAgent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
agent = DirectPolicyAgent(device)
agent.to(device)

pg.init()

# Parameters
generations = 10
episodes_per_gen = 10000 # Episodes before new generation
batch_size = 10 #Episodes before param update
learning_rate = 0.0001 # Learning rate
decay_rate = 0.99 # Weight decay for Adam optimizer

# Optimizer
optimizer = optim.Adam(agent.parameters(), lr=learning_rate, weight_decay=decay_rate)

# Environment
env = gym.make('ConnectFour-v0')

s = env.reset()
env.configurePlayer(random.choice([-1,1]))

# Main training loop
# for gen in range(generations):
#     model_save_path = 'AgentParameters/agent_params'+gen+'.pkl'
#     if os.path.isfile(model_save_path):
#         print('Loading model parameters')

win_rate = []
losses = []
games_final_rewards = []

episode = 0

while True:
    if keyboard.is_pressed("Esc"):
        sys.exit()

    s = env.reset()
    env.configurePlayer(random.choice([-1,1]))
    
    # Playing a game
    while True:
        # if episode % 1000 == 0:
        #     env.render()
        #     pg.time.wait(1000)
    
        if env.player == -1:
            choices = env.game.legal_cols()
            action = random.choice(choices)
        else:
            action = agent.select_action(s)

        s, r, done, _ = env.step(action)

        if done:
            agent.rewards.append(r)
            agent.calculate_rewards()
            games_final_rewards.append(r)
            #print(r)
            break
        elif env.player == 1:
            agent.rewards.append(r)

        env.configurePlayer(env.player * -1)
    
    if episode % batch_size == 0:
        loss = []
        for log_prob, reward in zip(agent.saved_log_probs, agent.rewards):
            loss.append(-log_prob * reward)
        
        loss = torch.stack(loss).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().numpy())

        del agent.rewards[:]
        del agent.saved_log_probs[:]

        # Calculating win percentage
        # wins = [game_r == 1 for game_r in games_final_rewards]
        # win_rate.append(np.mean(wins))

        # del games_final_rewards[:]

        #print(f'Reinforce ep {episode} done. Winrate: {np.mean(wins)}. Loss: {loss.detach().numpy()}')

    if episode % 20000 == 0:
        # Calculating win percentage
        wins = [game_r == 1 for game_r in games_final_rewards]
        win_rate.append(np.mean(wins))

        del games_final_rewards[:]

        print(f'Reinforce ep {episode} done. Winrate: {np.mean(wins)}.')
    
    episode += 1

