import torch
import numpy as np
from _not import Not
from _and import And
from _or import Or
from _xor import Xor
from common import load, test
  
if __name__ == "__main__":
    test(
        model = load(Not),
        X = np.array([[0], [1]]),
        y = np.array([1, 0]),
    )
        
    test(
        model = load(And),
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        y = np.array([0, 0, 0, 1]),
    )
        
    test(
        model = load(Or),
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        y = np.array([0, 1, 1, 1]),
    )
    
    test(
        model = load(Xor),
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        y = np.array([0, 1, 1, 0]),
    )    
