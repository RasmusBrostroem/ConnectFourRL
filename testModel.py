import torch
import pygame as pg
import gym
import gym_game
from agent import DirectPolicyAgent
import random

model = DirectPolicyAgent("cpu")
model.train(False)
model = torch.load("AgentParameters/StackerBoi_gen_24.pth")
# opponent = DirectPolicyAgent("cpu")
# opponent.train(False)
# opponent = torch.load("AgentParameters/StackerBoi_gen_25.pth")

env = gym.make('ConnectFour-v0')

pg.init()

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
                for i in range(10):
                    action = opponent.select_action(s*-1, choices)
                    if action in choices:
                        break
                    elif i == 9:
                        action = random.choice(choices)
        else:
            with torch.no_grad():
                action = agent.select_action(s, choices)

        s, r, done, _ = env.step(action)

        if done:
            if show_game:
                env.render(r)
                pg.display.quit()
                
            #agent.rewards.append(r)
            #agent.calculate_rewards(env)
            return r
        #elif env.player == 1:
            #agent.rewards.append(r)

        env.configurePlayer(env.player * -1)

wins = 0
ties = 0
illegal = 0
n_games = 1000
for i in range(n_games):
    re = play_game(env, model, show_game=False)
    if re == env.game.win:
        wins += 1
    elif re == env.game.tie:
        ties += 1
    elif re == env.game.illegal:
        illegal += 1

print(f"Winrate: {wins/n_games}")
print(f"tierate: {ties/n_games}")
print(f"illegal_rate: {illegal/n_games}")

