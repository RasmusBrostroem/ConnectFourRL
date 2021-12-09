import numpy as np
import gym
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
batch_size = 1 #Episodes before param update
learning_rate = 0.001 # Learning rate
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
show = False

def play_game(env, agent, opponent = None, show_game = False):
    pass

while True:
    s = env.reset()
    env.configurePlayer(random.choice([-1,1]))

    if keyboard.is_pressed("Esc"):
            sys.exit()

    if keyboard.is_pressed("t"):
        show = True
        
    # Playing a game
    while True:
        if show:
            env.render()
    
        if keyboard.is_pressed("n"):
            show = False
            pg.display.quit()
    
        if env.player == -1:
            choices = env.game.legal_cols()
            action = random.choice(choices)
        else:
            action = agent.select_action(s)

        s, r, done, _ = env.step(action)

        if done:
            if show:
                env.render(r)
                
            agent.rewards.append(r)
            agent.calculate_rewards()
            games_final_rewards.append(r)
            break
        elif env.player == 1:
            agent.rewards.append(r)

        env.configurePlayer(env.player * -1)
    
    if episode % batch_size == 0:
        # # Output agent parameters
        # for param in agent.parameters():
        #     print(param.data)
        #     break

        loss = []
        for log_prob, reward in zip(agent.saved_log_probs, agent.rewards):
            loss.append((-log_prob+1) * reward)
        
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

    if episode % 1000 == 0:
        # Calculating win percentage
        wins = [game_r == env.game.win for game_r in games_final_rewards]
        win_rate.append(np.mean(wins))
        illegals = [game_r == env.game.illegal for game_r in games_final_rewards]

        print(f'Reinforce ep {episode} done. Winrate: {np.mean(wins)}. Illegal move rate: {np.mean(illegals)}. Average Loss: {np.mean(losses)}')

        del games_final_rewards[:]
        del losses[:]

    episode += 1
