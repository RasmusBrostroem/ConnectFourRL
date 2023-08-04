import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from game import connectFour
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self):
        """Construct TDAgent object.

        Does not call the init of DirectPolicyAgent to avoid copying its
        network architecture.
        """
        nn.Module.__init__(self)
        self.L1 = nn.Linear(42, 50)
        self.L2 = nn.Linear(50, 1)
        # TODO: eligibility traces (torch tensor with correct dims)
        # TODO: Do we need two traces? one for each player piece?
        # TODO: Lambda value (as kwarg)
        # TODO: alpha value

    def forward(self, x):
        """Pass a game state through the network to estimate its value.

        Args:
            x (Tensor): Flattened binary representation of game state.

        Returns:
            Tensor: Probability for each column. The final layer is softmax,
                so the output tensor sums to 1.
        """
        x = self.L1(x)
        x = F.sigmoid(x)
        x = self.L2(x)
        return F.sigmoid(x)


model = NeuralNetwork()

board = np.zeros((6, 7))
board += np.random.choice(a=[0, 1, -1], size=(6, 7))
x = torch.from_numpy(board).float().flatten()
with torch.no_grad():
    v_hat = model.forward(x)
print("v_hat", v_hat)
print("grad before", model.L1.weight.grad)
v_hat = model.forward(x)
print("v_hat", v_hat)
v_hat.backward()
print("grad after", model.L1.weight.grad)
# torch.set_grad_enabled(False)
# v_hat = model.forward(x)
# print("v hat", v_hat)
# torch.set_grad_enabled(True)
# v_hat = model.forward(x)
# print(v_hat)
# v_hat.backward()
# print("grad after", model.L1.weight.grad)
# model.zero_grad()
# print("grad after zero", model.L1.weight.grad)
#print("dims of grad weight", model.L1.weight.grad.shape)
#print("dims of grad bias", model.L1.bias.grad.shape)
