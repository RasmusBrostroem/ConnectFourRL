from game.players import DirectPolicyAgent, Player, MinimaxAgent, TDAgent
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
#player2 = MinimaxAgent(player_piece = -1, win_reward = 10, loss_reward = -20)
#player1 = MinimaxAgent(player_piece=1)
# run = neptune.init_run(project="DLProject/ConnectFour")
# player1.log_params(neptune_run=run)
# player2.log_params(neptune_run=run)

# optimizerPL2 = optim.RMSprop(player2.parameters(), lr=0.1, weight_decay=0.95)


player1 = TDAgent(player_piece=1)
player1.load_network_weights("learned_weights/player1_selftrain.pt")
player1.eval()
player2 = TDAgent(player_piece=-1)
player2.load_network_weights("learned_weights/player1_selftrain.pt")
player2.eval()
for p1p,p2p in zip(player1.parameters(), player2.parameters()):
    print(p1p == p2p)

#player2 = MinimaxAgent(player_piece=-1)
#player2 = Player(player_piece=-1)
environment = Env(player1, player2, allow_illegal_moves=False)

for i in range(1, 10):
    #environment.self_play()
    environment.play_game()
    # wins = player1.stats["wins"]
    # games = player1.stats["games"]
    # print(f"Winrate, TDtraining against  : {wins}/{games}({wins/games})")
    # if i % 20 == 0:
    #     player2.update_agent(optimizer=optimizerPL2)
    #     player1.update_agent()
    #     player1.log_stats(neptune_run=run)
    #     player2.log_stats(neptune_run=run)
print("played")
for p1p,p2p in zip(player1.parameters(), player2.parameters()):
    print(p1p == p2p)