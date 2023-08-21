import torch

def fit(model, X, y, epochs, name = None, optimizer = None, criterion = None):
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    if criterion is None:
        criterion = torch.nn.MSELoss()
    if name is None:
        name = model.name()
    
    for epoch in range(epochs):
        losses = .0
        for data, target in zip(X, y):
            data = torch.Tensor(data)
            target = torch.Tensor([target])

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            losses += loss.item()
            
            loss.backward()
            optimizer.step()
            
        print('epoch {}, loss {}'.format(epoch, losses))
            
    print('w', model.state_dict().items())

    torch.save(model, f"pickles/{name}.pickle")
    

def load(model_class):
    pre_trained_model = torch.load(f"pickles/{model_class.name()}.pickle")
    
    new_model = model_class()
    new_model.load_state_dict(pre_trained_model.state_dict())
    
    for param in new_model.parameters():
        param.requires_grad = False
        
    return new_model