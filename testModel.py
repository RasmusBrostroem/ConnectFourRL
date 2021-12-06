import torch
import numpy as np

from agent import DirectPolicyAgent



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DirectPolicyAgent(device)
model.to(device)

test = np.zeros((6,7))

action = model.select_action(test)

print(action)



