import torch
import pygame as pg
import gym
import gym_game
from agent import DirectPolicyAgent
import random

model = DirectPolicyAgent("cpu")
model.train(False)
model = torch.load("AgentParameters/StackerBoi_gen_10.pth")

env = gym.make('ConnectFour-v0')

pg.init()

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

        if done and show_game:
            env.render(r)
            pg.display.quit()
            
            #agent.rewards.append(r)
            #agent.calculate_rewards(env)
            return r
        #elif env.player == 1:
            #agent.rewards.append(r)

        env.configurePlayer(env.player * -1)

for i in range(10):
    play_game(env, model, show_game=True)

