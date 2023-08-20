import torch
from torch import nn
import numpy as np
from common import fit

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

class And(nn.Module):
    def __init__(self):
        super(And, self).__init__()
        self.h1 = nn.Linear(2, 1)
        self.f = nn.ReLU()
        
    def forward(self, x):
        x = self.h1(x)
        x = self.f(x)
        return x

if __name__ == "__main__":
    model = And()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    fit(model, X, y, optimizer, criterion, epochs=100, name="and")