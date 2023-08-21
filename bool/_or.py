import torch
from torch import nn
import numpy as np
from _not import Not
from _and import And
from common import fit, load

"""
OR gate
Requires: <NOT, AND>
Returns: 
- (0, 0) -> 0
- (0, 1) -> 1
- (1, 0) -> 1
- (1, 1) -> 1 
"""
class Or(nn.Module):
    def __init__(self):
        super(Or, self).__init__()
        self.h1 = nn.Linear(4, 2)
        self.f = nn.Sigmoid()
        self.not_model = load(Not)
        self.and_model = load(And)
        
    def forward(self, x):
        x_pos = x.view(-1, 1)
        x_neg = self.not_model(x.view(-1, 1))
        x = torch.cat((x_pos, x_neg), 1).view(-1)
        x = self.h1(x)
        x = self.f(x)
        x = self.and_model(x)
        x = self.not_model(x)
        return x

    @staticmethod
    def name():
        return "or"

if __name__ == "__main__":
    fit(
        model = Or(), 
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), 
        y = np.array([0, 1, 1, 1]), 
        epochs=1000
    )
