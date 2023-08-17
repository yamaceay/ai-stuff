import torch
from torch import nn
import numpy as np
from _not import Not
from _and import And

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

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
    
class Or(nn.Module):
    def __init__(self):
        super(Or, self).__init__()
        self.h1 = nn.Linear(4, 2)
        self.f = nn.Sigmoid()
        
    def forward(self, x):
        x_pos = x.view(-1, 1)
        x_neg = not_model(x.view(-1, 1))
        x = torch.cat((x_pos, x_neg), 1).view(-1)
        x = self.h1(x)
        x = self.f(x)
        x = and_model(x)
        x = not_model(x)
        return x

if __name__ == "__main__":
    model = Or()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    
    for epoch in range(1000):
        losses = 0
        for data, target in zip(X, y):
            data = torch.Tensor(data)
            target = torch.Tensor([target])

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            losses += loss.item()
         
        print('epoch {}, loss {}'.format(epoch, losses))
    print('w', model.state_dict().items())

    torch.save(model, "pickles/or.pickle")
