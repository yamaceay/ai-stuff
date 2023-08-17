import torch
import numpy as np
from _not import Not
from _and import And
from _or import Or
from _xor import Xor

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
    
xor_model = Xor()
xor_model_state_dict = torch.load("pickles/xor.pickle").state_dict()
xor_model.load_state_dict(xor_model_state_dict)
for param in xor_model.parameters():
    param.requires_grad = False
  
if __name__ == "__main__":
    X_not = np.array([[0], [1]])
    y_not = np.array([1, 0])
    
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([0, 1, 1, 1])
    
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    print("--------------------------------------------")
    for data, target in zip(X_not, y_not):
        data = torch.Tensor(data)
        target = torch.Tensor([target])
        output = not_model(data)
        
        data = [str(int(x)) for x in data.numpy()]
        pred = [str(int(x)) for x in output.round().numpy()]
        truth = [str(int(x)) for x in target.numpy()]
        print(f"not({', '.join(data)}) = {', '.join(pred)} (expected: {', '.join(truth)})")
        
    print("--------------------------------------------")
    for data, target in zip(X_and, y_and):
        data = torch.Tensor(data)
        target = torch.Tensor([target])
        output = and_model(data)
          
        data = [str(int(x)) for x in data.numpy()]
        pred = [str(int(x)) for x in output.round().numpy()]
        truth = [str(int(x)) for x in target.numpy()]
        print(f"and({', '.join(data)}) = {', '.join(pred)} (expected: {', '.join(truth)})")
        
    print("--------------------------------------------")
    for data, target in zip(X_or, y_or):
        data = torch.Tensor(data)
        target = torch.Tensor([target])
        output = or_model(data)
            
        data = [str(int(x)) for x in data.numpy()]
        pred = [str(int(x)) for x in output.round().numpy()]
        truth = [str(int(x)) for x in target.numpy()]
        print(f"or({', '.join(data)}) = {', '.join(pred)} (expected: {', '.join(truth)})")
        
    print("--------------------------------------------")
    for data, target in zip(X_xor, y_xor):
        data = torch.Tensor(data)
        target = torch.Tensor([target])
        output = xor_model(data)
           
        data = [str(int(x)) for x in data.numpy()]
        pred = [str(int(x)) for x in output.round().numpy()]
        truth = [str(int(x)) for x in target.numpy()]
        print(f"xor({', '.join(data)}) = {', '.join(pred)} (expected: {', '.join(truth)})")
    
