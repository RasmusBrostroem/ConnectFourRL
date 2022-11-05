from testModel import matchup
from minimaxAgent import MinimaxAgent
from agent import DirectPolicyAgent
import torch

# model = DirectPolicyAgent("cpu")
# model.train(False)
# model = torch.load("AgentParameters/LastHopeBoi_gen_4.pth")
# # opponent = DirectPolicyAgent("cpu")
# # opponent.train(False)
# # opponent = torch.load("AgentParameters/AverageJoe_gen_14.pth")
# opponent = MinimaxAgent(max_depth=0)

# matchup(model, opponent, 4, True)


test = DirectPolicyAgent("cpu")

print(isinstance(test, MinimaxAgent))