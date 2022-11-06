from testModel import matchup
from minimaxAgent import MinimaxAgent
from game.players import DirectPolicyAgent, Player
from game.Env import Env
import torch
import random
import numpy as np

# model = DirectPolicyAgent("cpu")
# model.train(False)
# model = torch.load("AgentParameters/LastHopeBoi_gen_4.pth")
# # opponent = DirectPolicyAgent("cpu")
# # opponent.train(False)
# # opponent = torch.load("AgentParameters/AverageJoe_gen_14.pth")
# opponent = MinimaxAgent(max_depth=0)

player1 = Player(_playerPiece = 1, _winReward = 1, _lossReward = -1, _tieReward = 0.5, _illegalReward = -5, _device = "cpu")
player2 = Player(_playerPiece = -1, _winReward = 1, _lossReward = -1, _tieReward = 0.5, _illegalReward = -5, _device = "cpu")

environment = Env(player1, player2, False)
for i in range(1000):
    environment.play_game()

# print("player1 rewards")
# print(player1.rewards)
# print("player2 rewards")
# print(player2.rewards)