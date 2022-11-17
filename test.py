from testModel import matchup
from minimaxAgent import MinimaxAgent
from game.players import DirectPolicyAgent, Player
from game.Env import Env
from torch.distributions import Categorical
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

#player1 = DirectPolicyAgent(_playerPiece = 21, _winReward = 31, _lossReward = -11, _tieReward = 0.25, _illegalReward = -5, _device = "cpu")# #_winReward = 1, _lossReward = -1, _tieReward = 0.5, _illegalReward = -5, _device = "cpu")
player2 = Player(player_piece = -1)
player1 = Player(player_piece= 1)


#print(player2.params)
environment = Env(player1, player2)
for i in range(1):
    environment.play_game()


#print(torch.tensor(3) == torch.tensor([3]))



# print("player1 rewards")
# print(player1.rewards)
# print("player2 rewards")
# print(player2.rewards)