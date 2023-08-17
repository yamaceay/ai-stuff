import torch
from torch import nn
import numpy as np

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
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(100):
        for data, target in zip(X, y):
            data = torch.Tensor(data)
            target = torch.Tensor([target])

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            print('epoch {}, loss {}'.format(epoch, loss.item()))
            
    print('w', model.state_dict().items())

    torch.save(model, "pickles/and.pickle")