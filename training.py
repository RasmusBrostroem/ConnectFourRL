import pygame as pg
import keyboard
import os
from pygame.display import update
import torch
import random
import itertools
from game.players import DirectPolicyAgent, DirectPolicyAgent_large, DirectPolicyAgent_mini, MinimaxAgent
import numpy as np

pg.init()

def play_game(env, agent, illegal_move_possible, opponent = None, show_game = False):
    s = env.reset()
    env.configurePlayer(random.choice([-1,1]))

    while True:
        if keyboard.is_pressed("z"):
            show_game = False
            pg.display.quit()
        
        if keyboard.is_pressed("x"):
            show_game = True
        
        if show_game:
            env.render()

        choices = env.game.legal_cols()
        if env.player == -1 and opponent is None:
            action = random.choice(choices)
        elif env.player == -1 and opponent is not None:
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
            if not isinstance(opponent, MinimaxAgent):
                if r == env.game.tie:
                    opponent.rewards.append(r)
                elif r == env.game.loss:
                    opponent.rewards.append(env.game.win)
                elif r == env.game.win:
                    opponent.rewards.append(env.game.loss)
                opponent.game_succes.append(None)
                opponent.calculate_rewards(env)

            return r
        elif env.player == 1:
            agent.rewards.append(r)
            agent.game_succes.append(None)
        elif env.player == -1:
            if not isinstance(opponent, MinimaxAgent):
                opponent.rewards.append(r)
                opponent.game_succes.append(None)

        env.configurePlayer(env.player * -1)

def update_agent(agent, optimizer):
    loss = []
    # rewards = torch.tensor(agent.rewards)
    # norm_rewards = (rewards - rewards.mean())/(rewards.std()+1e-6)
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

    # if not minimax:
    #     #opponents = [load_agent(path, name, gen-1, agent_size, device)]
    #     if agent_size == "Mini":
    #         opponents = [DirectPolicyAgent_mini(device)]
    #     elif agent_size == "Small":
    #         opponents = [DirectPolicyAgent(device)]
    #     else:
    #         opponents = [DirectPolicyAgent_large(device)]
    # else:
    #     opponents = [minimax_agent]

    # opponent_iter = itertools.cycle(opponents)

    opponent = DirectPolicyAgent(device)

    for gen in range(generations):
        for ep in range(episodes_per_gen):
            #opponent = next(opponent_iter)

            if (ep+1) % show_every == 0:
                final_reward = play_game(env, agent, illegal_move_possible, opponent, True)
            else:
                final_reward = play_game(env, agent, illegal_move_possible, opponent)
            
            games_final_rewards.append(final_reward)

            if (ep+1) % batchsize == 0:
                loss = update_agent(agent, optimizer)
                losses.append(loss)
                neptune_run["metrics/Batch_loss"].log(np.mean(loss))
                
                update_agent(opponent, optimizer)
                
            
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

                del opponent.game_succes[:]
                del opponent.probs[:]
                del opponent.rewards[:]
                del opponent.saved_log_probs[:]

        
        # Saving the model as a new generation is beginning
        agent_name = name + f"_gen_{gen}.pth"
        agent_path = os.path.join(path, agent_name)
        torch.save(agent, agent_path)

        # Deleting the model parameters five generations back
        agent_gen5_name = name + f"_gen_{gen-5}.pth"
        agent_gen5_path = os.path.join(path, agent_gen5_name)
        if os.path.isfile(agent_gen5_path):
            os.remove(agent_gen5_path)