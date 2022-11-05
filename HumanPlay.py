import torch
import pygame as pg
import gym
import game
import minimaxAgent
from game.players import DirectPolicyAgent
import random
from minimaxAgent import MinimaxAgent
# model = DirectPolicyAgent("cpu")
# model.train(False)
# model = torch.load("AgentParameters/StackerBoi_gen_24.pth")
model = MinimaxAgent(max_depth=2)
env = gym.make('ConnectFour-v0')
env.configureRewards(win=1, loss=-1, tie=0.1, illegal=-10)

pg.init()

def play_game(env, agent, opponent = None, show_game = False):
    s = env.reset()
    env.configurePlayer(random.choice([-1,1]))

    while True:
        if show_game:
            env.render()

        choices = env.game.legal_cols()
        if env.player == -1 and opponent is None:
            action = int(input("Vælg kolonne: "))-1
        elif env.player == -1 and opponent is not None:
            with torch.no_grad():
                for i in range(10):
                    action = opponent.select_action(s*-1)
                    if action in choices:
                        break
                    elif i == 9:
                        action = random.choice(choices)
        else:
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

for i in range(1):
    re = play_game(env, model, show_game=True)