import numpy as np

def print_payoff(payoff):
    try:
        payoff_str = np.array([[
            f"({payoff[i, j, 0]}, {payoff[i, j, 1]})" 
            for j in range(payoff.shape[1])] 
            for i in range(payoff.shape[0])])
        print(payoff_str)
    except IndexError as e:
        print(e)
        print(payoff)

def is_dominant(a, b):
    return np.all([a[k] >= b[k] for k in range(len(a))])

def find_rational_decisions(matrix):
    n, m, _ = matrix.shape
    
    a_values = [matrix[i, :, 0] for i in range(n)]
    b_values = [matrix[:, i, 1] for i in range(m)]
    
    dominance_matrix_a, dominance_matrix_b = np.zeros((n, n), dtype=bool), np.zeros((m, m), dtype=bool)
    a_to_delete, b_to_delete = set(), set()
    
    for i in range(n):
        for j in range(n):
            if is_dominant(a_values[i], a_values[j]):
                dominance_matrix_a[i, j] = True
                if i != j:
                    # print(f"Player A: {i} dominates {j}")
                    a_to_delete.add(j)
                    
    for i in range(m):
        for j in range(m):
            if is_dominant(b_values[i], b_values[j]):
                dominance_matrix_b[i, j] = True
                if i != j:
                    # print(f"Player B: {i} dominates {j}")
                    b_to_delete.add(j)
                
    return dominance_matrix_a, dominance_matrix_b, a_to_delete, b_to_delete


def search_for_dominant_strategies(matrix, depth=0):
    print("-------------------------------------------"*2)

    print_payoff(matrix)
    
    n_strategies_prev = matrix.shape[0] + matrix.shape[1]
        
    *_, a_to_delete, b_to_delete = find_rational_decisions(matrix)
    
    a_matrix = np.delete(matrix, list(a_to_delete), axis=0)
    n_strategies_a = a_matrix.shape[0] + a_matrix.shape[1]
    
    if n_strategies_a < n_strategies_prev:
    
        print("-------------------------------------------"*2)
        print("--"*depth + f"Player A won't play: {a_to_delete}")
        search_for_dominant_strategies(a_matrix, depth+1)
    
    b_matrix = np.delete(matrix, list(b_to_delete), axis=1)
    n_strategies_b = b_matrix.shape[0] + b_matrix.shape[1]
    
    if n_strategies_b < n_strategies_prev:
        
        print("-------------------------------------------"*2)
        print("--"*depth + f"Player B won't play: {b_to_delete}")
        search_for_dominant_strategies(b_matrix, depth+1)
    
if __name__ == "__main__":
    
    l, u = -6, 6
    shape = [10, 6]

    payoff = np.random.randint(l, u, size=(*shape, len(shape)))
    
    search_for_dominant_strategies(payoff)