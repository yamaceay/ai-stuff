import torch
from torch import nn
import numpy as np
from _not import Not
from _and import And
from _or import Or
from common import fit, load
    
class Xor(nn.Module):
    def __init__(self):
        super(Xor, self).__init__()
        self.h1 = nn.Linear(4, 4)
        self.f = nn.Sigmoid()
        self.not_model = load(Not)
        self.and_model = load(And)
        self.or_model = load(Or)
        
    def forward(self, x):
        x_pos = x.view(-1, 1)
        x_neg = self.not_model(x.view(-1, 1))
        x = torch.cat((x_pos, x_neg), 1).view(-1)
        x = self.h1(x)
        x = self.f(x)
        x_first = self.or_model(x[:2])
        x_second = self.or_model(x[2:])
        x = torch.cat((x_first, x_second), 0).view(-1)
        x = self.and_model(x)
        return x
    
    @staticmethod
    def name():
        return "xor"

if __name__ == "__main__":    
    fit(
        model = Xor(), 
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        y = np.array([0, 1, 1, 0]),
        epochs=10000
    )
