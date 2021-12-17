import pygame as pg
import os
import torch
import random
import itertools
from agent import DirectPolicyAgent, DirectPolicyAgent_large, DirectPolicyAgent_mini
import numpy as np
from minimaxAgent import MinimaxAgent

def play_game(env, agent, illegal_move_possible, opponent = None, show_game = False):
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
                action = opponent.select_action(s*-1, choices)
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

    del agent.rewards[:]
    del agent.saved_log_probs[:]

    return loss.detach().numpy()

def load_agent(path, name, gen, size, device):
    '''
    Loads one of the agents (small or large) if the generation (gen) exists with the giving name and path.
    If this agent doesn't exist, then return None
    '''
    opponent_name = name + f"_gen_{gen}.pth"
    opponent_path = os.path.join(path, opponent_name)
    if os.path.isfile(opponent_path):
        if size == "Small":
            opponent = DirectPolicyAgent(device)
            opponent.train(False)
            opponent = torch.load(opponent_path)
            opponent.to(device)
        elif size == "Mini":
            opponent = DirectPolicyAgent_mini(device)
            opponent.train(False)
            opponent = torch.load(opponent_path)
            opponent.to(device)
        else:
            opponent = DirectPolicyAgent_large(device)
            opponent.train(False)
            opponent = torch.load(opponent_path)
            opponent.to(device)
    else:
        opponent = None
    
    return opponent

def train_agent(env, agent, optimizer, neptune_run, generations, episodes_per_gen, batchsize, minimax, agent_size, illegal_move_possible, device, path_name = ["",""], print_every = 1000, show_every = 1000):
    losses = []
    games_final_rewards = []

    path, name = path_name

    minimax_agent = MinimaxAgent(max_depth=0)

    for gen in range(generations):
        opponents = None
        if not minimax:
            opponents = [load_agent(path, name, gen-i, agent_size, device) for i in range(3,0,-1)]
        else:
            opponents = [minimax_agent]
        opponent_iter = itertools.cycle(opponents)
        for ep in range(episodes_per_gen):
            opponent = next(opponent_iter)

            if (ep+1) % show_every == 0:
                final_reward = play_game(env, agent, illegal_move_possible, opponent, True)
            else:
                final_reward = play_game(env, agent, illegal_move_possible, opponent)
            
            games_final_rewards.append(final_reward)

            if (ep+1) % batchsize == 0:
                loss = update_agent(agent, optimizer)
                losses.append(loss)
                neptune_run["metrics/Batch_loss"].log(np.mean(loss))
            
            if (ep+1) % print_every == 0:
                wins = [game_r == env.game.win for game_r in games_final_rewards]
                illegals = [game_r == env.game.illegal for game_r in games_final_rewards]
                defeats = [game_r == env.game.loss for game_r in games_final_rewards]
                ties = [game_r == env.game.tie for game_r in games_final_rewards]

                neptune_run["metrics/Winrate"].log(np.mean(wins))
                neptune_run["metrics/Illegal_rate"].log(np.mean(illegals))
                neptune_run["metrics/Loss_rate"].log(np.mean(defeats))
                neptune_run["metrics/Tie_rate"].log(np.mean(ties))

                neptune_run["metrics/Average_loss"].log(np.mean(losses))
                neptune_run["metrics/AverageProbWins"].log(np.mean([prob.cpu().detach().numpy() for prob, succes in zip(agent.probs, agent.game_succes) if succes]))
                neptune_run["metrics/AverageProbLoss"].log(np.mean([prob.cpu().detach().numpy()  for prob, succes in zip(agent.probs, agent.game_succes) if not succes]))

                test_rewards = []
                for i in range(100):
                    re = play_game(env, agent, illegal_move_possible, minimax_agent)
                    test_rewards.append(re)
                
                test_wins = [game_r == env.game.win for game_r in test_rewards]
                neptune_run["metrics/test_winrate"].log(np.mean(test_wins))

                del games_final_rewards[:]
                del losses[:]
                del agent.game_succes[:]
                del agent.probs[:]
                del agent.rewards[:]
                del agent.saved_log_probs[:]
                del test_rewards[:]

        
        # Saving the model as a new generation is beginning
        agent_name = name + f"_gen_{gen}.pth"
        agent_path = os.path.join(path, agent_name)
        torch.save(agent, agent_path)

        # Deleting the model parameters five generations back
        agent_gen5_name = name + f"_gen_{gen-5}.pth"
        agent_gen5_path = os.path.join(path, agent_gen5_name)
        if os.path.isfile(agent_gen5_path):
            os.remove(agent_gen5_path)