import numpy as np
import gym
import pygame as pg
import gym_game
import os
import sys
import keyboard
from numba import jit, cuda

# Neptune
import neptune.new as neptune

import torch
import torch.optim as optim

import random
from agent import DirectPolicyAgent

run = neptune.init(
    project="DLProject/ConnectFour"
) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
agent = DirectPolicyAgent(device)
agent.to(device)

pg.init()

# Parameters
generations = 100
episodes_per_gen = 100000 # Episodes before new generation
batch_size = 100 #Episodes before param update
learning_rate = 0.001 # Learning rate
decay_rate = 0 # Weight decay for Adam optimizer
illegal_move_possible = False

# Optimizer
optimizer = optim.Adam(agent.parameters(), lr=learning_rate, weight_decay=decay_rate)

# Environment
env = gym.make('ConnectFour-v0')

# Neptune params
params = {"generations": generations,
          "episodes_per_gen": episodes_per_gen,
          "batch_size": batch_size,
          "optimizer": "Adam",
          "learning_rate": learning_rate,
          "weight_decay": decay_rate,
          "reward_decay": agent.gamma,
          "win_reward": env.game.win,
          "loss_reward": env.game.lose,
          "tie_reward": env.game.tie,
          "illegal_reward": env.game.illegal,
          "illegal_move_possible": illegal_move_possible}

run["parameters"] = params

for name, param in agent.named_parameters():
    run["model/summary"].log(f"name: {name} with size: {param.size()}")


def play_game(env, agent, opponent = None, show_game = False):
    s = env.reset()
    env.configurePlayer(random.choice([-1,1]))

    while True:
        if show_game:
            env.render()
        choices = env.game.legal_cols()
        if env.player == -1 and opponent is None:
            action = random.choice(choices)
        elif env.player == -1 and opponent is not None:
            with torch.no_grad():
                action = opponent.select_action(s, choices)
        else:
            if illegal_move_possible:
                action = agent.select_action(s, None)
            else:
                action = agent.select_action(s, choices)

        s, r, done, _ = env.step(action)

        if done:
            if show_game:
                env.render(r)
                pg.display.quit()
                
            agent.rewards.append(r)
            agent.game_succes.append(None)
            agent.calculate_rewards(env)

            return r
        elif env.player == 1:
            agent.rewards.append(r)
            agent.game_succes.append(None)

        env.configurePlayer(env.player * -1)

def update_agent(agent, optimizer):
    loss = []
    for log_prob, reward in zip(agent.saved_log_probs, agent.rewards):
        loss.append(-log_prob * reward)

    # loss = [-log_prob*reward if succes else -torch.log(1-prob)*reward
    #         for log_prob, reward, prob, succes 
    #         in zip(agent.saved_log_probs, agent.rewards, agent.probs, agent.game_succes)]
    
    loss = torch.stack(loss).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    run["metrics/AverageProbWins"].log(torch.mean(torch.tensor([prob for prob, succes in zip(agent.probs, agent.game_succes) if succes])))
    run["metrics/AverageProbLoss"].log(torch.mean(torch.tensor([prob for prob, succes in zip(agent.probs, agent.game_succes) if not succes])))

    del agent.rewards[:]
    del agent.saved_log_probs[:]
    del agent.game_succes[:]
    del agent.probs[:]

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

            if ep % batchsize == 0:
                loss = update_agent(agent, optimizer)
                losses.append(loss)
            
            if ep % print_every == 0:
                wins = [game_r == env.game.win for game_r in games_final_rewards]
                illegals = [game_r == env.game.illegal for game_r in games_final_rewards]

                #print(f'Reinforce ep {ep} in gen {gen} done. Winrate: {np.mean(wins)}. Illegal move rate: {np.mean(illegals)}. Average Loss: {np.mean(losses)}')
                run["metrics/Winrate"].log(np.mean(wins))
                run["metrics/Illegal_rate"].log(np.mean(illegals))
                run["metrics/Average_loss"].log(np.mean(losses))

                del games_final_rewards[:]
                del losses[:]
        
        # Saving the model as a new generation is beginning
        agent_name = name + f"_gen_{gen}.pth"
        agent_path = os.path.join(path, agent_name)
        torch.save(agent, agent_path)

if __name__ == "__main__":
    train_agent(env, agent, optimizer, generations, episodes_per_gen, batch_size, ["C:\Projects\ConnectFourRL\AgentParameters", "StackerBoi"], print_every=1000, show_every=100000000)
    run.stop()