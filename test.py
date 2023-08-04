from game.players import DirectPolicyAgent, Player, MinimaxAgent
from game.Env import Env
from game.connectFour import connect_four
from torch.distributions import Categorical
import torch
import torch.optim as optim
import random
import numpy as np
import neptune
from time import sleep

# model = DirectPolicyAgent("cpu")
# model.train(False)
# model = torch.load("AgentParameters/LastHopeBoi_gen_4.pth")
# # opponent = DirectPolicyAgent("cpu")
# # opponent.train(False)
# # opponent = torch.load("AgentParameters/AverageJoe_gen_14.pth")
# opponent = MinimaxAgent(max_depth=0)

#player1 = DirectPolicyAgent(_playerPiece = 21, _winReward = 31, _lossReward = -11, _tieReward = 0.25, _illegalReward = -5, _device = "cpu")# #_winReward = 1, _lossReward = -1, _tieReward = 0.5, _illegalReward = -5, _device = "cpu")
player2 = MinimaxAgent(player_piece = -1, win_reward = 10, loss_reward = -20)
player1 = MinimaxAgent(player_piece=1)

# run = neptune.init_run(project="DLProject/ConnectFour")
# player1.log_params(neptune_run=run)
# player2.log_params(neptune_run=run)

# optimizerPL2 = optim.RMSprop(player2.parameters(), lr=0.1, weight_decay=0.95)

environment = Env(player1, player2, allow_illegal_moves = False)
# game = connect_four()
# game.place_piece(1, 1)
# game.draw_board()
# sleep(1)
# game.remove_piece(1)
# game.draw_board()
# sleep(1)

for i in range(1,2):
    environment.play_game()
    # if i % 20 == 0:
    #     player2.update_agent(optimizer=optimizerPL2)
    #     player1.update_agent()
    #     player1.log_stats(neptune_run=run)
    #     player2.log_stats(neptune_run=run)



#print(torch.tensor(3) == torch.tensor([3]))



# print("player1 rewards")
# print(player1.rewards)
# print("player2 rewards")
# print(player2.rewards)