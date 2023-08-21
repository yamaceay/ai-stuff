import torch
import numpy as np
from _not import Not
from _and import And
from _or import Or
from _xor import Xor
from common import load
  
if __name__ == "__main__":
    print("--------------------------------------------")
    not_model = load(Not)
    X_not = np.array([[0], [1]])
    y_not = np.array([1, 0])
    
    for data, target in zip(X_not, y_not):
        data = torch.Tensor(data)
        target = torch.Tensor([target])
        output = not_model(data)
        
        data = [str(int(x)) for x in data.numpy()]
        pred = [str(int(x)) for x in output.round().numpy()]
        truth = [str(int(x)) for x in target.numpy()]
        print(f"not({', '.join(data)}) = {', '.join(pred)} (expected: {', '.join(truth)})")
        
    print("--------------------------------------------")
    and_model = load(And)
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    
    for data, target in zip(X_and, y_and):
        data = torch.Tensor(data)
        target = torch.Tensor([target])
        output = and_model(data)
          
        data = [str(int(x)) for x in data.numpy()]
        pred = [str(int(x)) for x in output.round().numpy()]
        truth = [str(int(x)) for x in target.numpy()]
        print(f"and({', '.join(data)}) = {', '.join(pred)} (expected: {', '.join(truth)})")
        
    print("--------------------------------------------")
    or_model = load(Or)
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([0, 1, 1, 1])
    
    for data, target in zip(X_or, y_or):
        data = torch.Tensor(data)
        target = torch.Tensor([target])
        output = or_model(data)
            
        data = [str(int(x)) for x in data.numpy()]
        pred = [str(int(x)) for x in output.round().numpy()]
        truth = [str(int(x)) for x in target.numpy()]
        print(f"or({', '.join(data)}) = {', '.join(pred)} (expected: {', '.join(truth)})")
        
    print("--------------------------------------------")
    xor_model = load(Xor)
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    for data, target in zip(X_xor, y_xor):
        data = torch.Tensor(data)
        target = torch.Tensor([target])
        output = xor_model(data)
           
        data = [str(int(x)) for x in data.numpy()]
        pred = [str(int(x)) for x in output.round().numpy()]
        truth = [str(int(x)) for x in target.numpy()]
        print(f"xor({', '.join(data)}) = {', '.join(pred)} (expected: {', '.join(truth)})")
    
