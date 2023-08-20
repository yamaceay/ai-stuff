import torch
from torch import nn
import numpy as np
from _not import Not
from _and import And
from _or import Or
from common import fit

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

not_model = Not()
not_model_state_dict = torch.load("pickles/not.pickle").state_dict()
not_model.load_state_dict(not_model_state_dict)
for param in not_model.parameters():
    param.requires_grad = False

and_model = And()
and_model_state_dict = torch.load("pickles/and.pickle").state_dict()
and_model.load_state_dict(and_model_state_dict)
for param in and_model.parameters():
    param.requires_grad = False
    
or_model = Or()
or_model_state_dict = torch.load("pickles/or.pickle").state_dict()
or_model.load_state_dict(or_model_state_dict)
for param in or_model.parameters():
    param.requires_grad = False
    
class Xor(nn.Module):
    def __init__(self):
        super(Xor, self).__init__()
        self.h1 = nn.Linear(4, 4)
        self.f = nn.Sigmoid()
        
    def forward(self, x):
        x_pos = x.view(-1, 1)
        x_neg = not_model(x.view(-1, 1))
        x = torch.cat((x_pos, x_neg), 1).view(-1)
        x = self.h1(x)
        x = self.f(x)
        x_first = or_model(x[:2])
        x_second = or_model(x[2:])
        x = torch.cat((x_first, x_second), 0).view(-1)
        x = and_model(x)
        return x

if __name__ == "__main__":
    model = Xor()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    
    fit(model, X, y, optimizer, criterion, epochs=10000, name="xor")
