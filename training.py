import pygame as pg
import os
import torch
import random
from agent import DirectPolicyAgent, DirectPolicyAgent_large

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

def load_agent(path, name, gen, size):
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
        else:
            opponent = DirectPolicyAgent_large(device)
            opponent.train(False)
            opponent = torch.load(opponent_path)
    else:
        opponent = None
    
    return opponent

def train_agent(env, agent, optimizer, generations, episodes_per_gen, batchsize, minimax, agent_size, illegal_move_possible, path_name = ["",""], print_every = 1000, show_every = 1000):
    losses = []
    games_final_rewards = []

    path, name = path_name

    for gen in range(generations):
        opponents = None
        if not minimax:
            opponents = [load_agent(path, name, gen-i, agent_size) for i in range(5,0,-1)]
        for ep in range(episodes_per_gen):
            if type(opponents) == list:
                opponent_id = ep % 5
                opponent = opponents[opponent_id]
            else:
                #opponent = MinMax()
                pass

            if (ep+1) % show_every == 0:
                final_reward = play_game(env, agent, illegal_move_possible, opponent, True)
            else:
                final_reward = play_game(env, agent, illegal_move_possible, opponent)
            
            games_final_rewards.append(final_reward)

            if (ep+1) % batchsize == 0:
                loss = update_agent(agent, optimizer)
                losses.append(loss)
            
            if (ep+1) % print_every == 0:
                #print(f'Reinforce ep {ep} in gen {gen} done. Winrate: {np.mean(wins)}. Illegal move rate: {np.mean(illegals)}. Average Loss: {np.mean(losses)}')
                run["metrics/Winrate"].log(torch.mean(torch.tensor([game_r == env.game.win for game_r in games_final_rewards])))
                run["metrics/Illegal_rate"].log(torch.mean(torch.tensor([game_r == env.game.illegal for game_r in games_final_rewards])))
                run["metrics/Loss_rate"].log(torch.mean(torch.tensor([game_r == env.game.loss for game_r in games_final_rewards])))
                run["metrics/Tie_rate"].log(torch.mean(torch.tensor([game_r == env.game.tie for game_r in games_final_rewards])))

                run["metrics/Average_loss"].log(torch.mean(losses))
                run["metrics/AverageProbWins"].log(torch.mean(torch.tensor([prob for prob, succes in zip(agent.probs, agent.game_succes) if succes])))
                run["metrics/AverageProbLoss"].log(torch.mean(torch.tensor([prob for prob, succes in zip(agent.probs, agent.game_succes) if not succes])))

                del games_final_rewards[:]
                del losses[:]
                del agent.game_succes[:]
                del agent.probs[:]
        
        # Saving the model as a new generation is beginning
        agent_name = name + f"_gen_{gen}.pth"
        agent_path = os.path.join(path, agent_name)
        torch.save(agent, agent_path)