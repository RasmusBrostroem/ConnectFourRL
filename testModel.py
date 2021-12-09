import numpy as np
import torch
from torch.distributions import Categorical
# probs = torch.Tensor([0.4,0.2,0.3,0.1])

# m = Categorical(probs)
# action = m.sample()
# print(action)

# log_prob = m.log_prob(action)
# print(log_prob)

# print(np.log(probs[action]))

prob = 0.9

log_prob = np.log(1-prob)
print(log_prob)

