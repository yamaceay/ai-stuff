import torch

def fit(model, X, y, optimizer, criterion, epochs, name):
    
    for epoch in range(epochs):
        losses = .0
        for data, target in zip(X, y):
            data = torch.Tensor(data)
            target = torch.Tensor([target])

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            losses += loss.item()
            optimizer.step()
            
        print('epoch {}, loss {}'.format(epoch, losses))
            
    print('w', model.state_dict().items())

    torch.save(model, f"pickles/{name}.pickle")