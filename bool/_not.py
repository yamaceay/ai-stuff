from torch import nn
import numpy as np
from common import fit

"""
NOT gate
Requires: <>
Returns: 
- (0) -> 1
- (1) -> 0
"""
class Not(nn.Module):
    def __init__(self):
        super(Not, self).__init__()
        self.h1 = nn.Linear(1, 1)
        self.f = nn.ReLU()
        
    def forward(self, x):
        x = self.h1(x)
        x = self.f(x)
        return x
        
    @staticmethod
    def name():
        return "not"

if __name__ == "__main__":
    fit(
        model = Not(),
        X = np.array([[0], [1]]), 
        y = np.array([1, 0]), 
        epochs=100,
    )