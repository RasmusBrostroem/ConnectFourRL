from testModel import matchup
from minimaxAgent import MinimaxAgent
from game.players import DirectPolicyAgent, Player
import torch
import random

# model = DirectPolicyAgent("cpu")
# model.train(False)
# model = torch.load("AgentParameters/LastHopeBoi_gen_4.pth")
# # opponent = DirectPolicyAgent("cpu")
# # opponent.train(False)
# # opponent = torch.load("AgentParameters/AverageJoe_gen_14.pth")
# opponent = MinimaxAgent(max_depth=0)

# matchup(model, opponent, 4, True)
a = DirectPolicyAgent("gpu")
b = Player(-1, 10, 10, 10, 10, "grp")


test = [a,b]

t = random.choice(test)

if t is a or t is b:
    print("Noiuce")
else:
    print("Fuck")