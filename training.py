import numpy as np
import gym
import pygame as pg
import gym_game
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
episodes_per_gen = 500 # Episodes before new generation
batch_size = 5 #Episodes before param update
learning_rate = 0.001 # Learning rate
decay_rate = 0.99 # Weight decay for Adam optimizer

# Optimizer
optimizer = optim.Adam(agent.parameters(), lr=learning_rate, weight_decay=decay_rate)

# Environment
env = gym.make('ConnectFour-v0')

# Main training loop
# for gen in range(generations):
#     model_save_path = 'AgentParameters/agent_params'+gen+'.pkl'
#     if os.path.isfile(model_save_path):
#         print('Loading model parameters')


def play_game(env, agent, opponent = None, show_game = False):
    s = env.reset()
    env.configurePlayer(random.choice([-1,1]))

    while True:
        if show_game:
            env.render()

        if env.player == -1 and opponent is None:
            choices = env.game.legal_cols()
            action = random.choice(choices)
        elif env.player == -1 and opponent is not None:
            choices = env.game.legal_cols()
            with torch.no_grad():
                for i in range(10):
                    action = opponent.select_action(s)
                    if action in choices:
                        break
                    elif i == 9:
                        action = random.choice(choices)
        else:
            action = agent.select_action(s)

        s, r, done, _ = env.step(action)

        if done:
            if show_game:
                env.render(r)
                pg.display.quit()
                
            agent.rewards.append(r)
            agent.calculate_rewards()
            return r
        elif env.player == 1:
            agent.rewards.append(r)

        env.configurePlayer(env.player * -1)

def update_agent(agent, optimizer):
    loss = []
    for log_prob, reward in zip(agent.saved_log_probs, agent.rewards):
        loss.append(-log_prob * reward)
    
    loss = torch.stack(loss).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    del agent.rewards[:]
    del agent.saved_log_probs[:]

    return loss.detach().numpy()

def train_agent(env, agent, optimizer, generations, episodes_per_gen, batchsize, path_name = ["",""], print_every = 1000, show_every = 1000):
    losses = []
    games_final_rewards = []

    path, name = path_name

    for gen in range(1,generations+1):
        opponent_name = name + f"_gen_{gen-1}.pth"
        opponent_path = os.path.join(path, opponent_name)
        if os.path.isfile(opponent_path):
            print(f"Loading generation {gen-1}")
            opponent = DirectPolicyAgent(device)
            opponent.train(False)
            opponent = torch.load(opponent_path)
        else:
            opponent = None

        for ep in range(1,episodes_per_gen+1):
            if keyboard.is_pressed("Esc"):
                sys.exit()

            if ep % show_every == 0:
                final_reward = play_game(env, agent, opponent, True)
            else:
                final_reward = play_game(env, agent, opponent)
            
            games_final_rewards.append(final_reward)

            if ep*gen % batchsize == 0:
                loss = update_agent(agent, optimizer)
                losses.append(loss)
            
            if ep % print_every == 0:
                wins = [game_r == env.game.win for game_r in games_final_rewards]
                illegals = [game_r == env.game.illegal for game_r in games_final_rewards]

                print(f'Reinforce ep {ep} in gen {gen} done. Winrate: {np.mean(wins)}. Illegal move rate: {np.mean(illegals)}. Average Loss: {np.mean(losses)}')

                del games_final_rewards[:]
                del losses[:]
        
        # Saving the model as a new generation is beginning
        agent_name = name + f"_gen_{gen}.pth"
        agent_path = os.path.join(path, agent_name)
        torch.save(agent, agent_path)

if __name__ == "__main__":
    train_agent(env, agent, optimizer, generations, episodes_per_gen, batch_size, ["C:\Projects\ConnectFourRL\AgentParameters", "StackerBoi"], print_every=500, show_every=500)