import numpy as np

FOUND_SOLUTIONS = None

def simple_slice(arr, inds, axis):
    sl = [slice(None)] * len(arr.shape)
    sl[axis] = inds
    return arr[tuple(sl)]

def ndindex(shape):
    total_size = np.prod(shape)
    unraveled_index = np.unravel_index(range(total_size), shape)
    ndlist = np.array(unraveled_index).T.tolist()
    ndlist = [tuple(x) for x in ndlist]
    return ndlist

def print_payoff(payoff):
    try:
        payoff_index = ndindex(payoff.shape[:-1])
        payoff_str = np.array([[
            f"{tuple(payoff[m])}" 
            for m in payoff_index]])
        print(payoff_str.reshape(payoff.shape[:-1]))
    except: 
        print(payoff)

def is_dominant(a, b):
    return np.all([a >= b])

def irrational_decisions(matrix):
    n_players = len(matrix.shape) - 1
    
    values_list = []
    for i in range(n_players):
        player_matrix = simple_slice(matrix, i, axis=n_players)
        values = np.moveaxis(player_matrix, source=i, destination=0)
        values_list.append(values)

    dominance_matrices = [
        np.zeros((matrix.shape[i], matrix.shape[i]), dtype=bool) 
        for i in range(n_players)]
    players_to_delete = [set() for _ in range(n_players)]
    
    for p in range(n_players):    
        for i in range(matrix.shape[p]):
            for j in range(matrix.shape[p]):
                if is_dominant(values_list[p][i], values_list[p][j]):
                    dominance_matrices[p][i, j] = True
                    if i != j:
                        # print(f"Player {p+1}: {i} dominates {j}")
                        players_to_delete[p].add(j)
                
    return players_to_delete


def dfs_dominant_strategies(matrix, history, depth=0):
    print("-------------------------------------------"*2)

    print_payoff(matrix)
    n_players = len(matrix.shape) - 1
    
    n_strategies_prev = sum(matrix.shape[:-1])
    
    if np.all(matrix.shape[:-1] == np.ones((n_players))):
        FOUND_SOLUTIONS.append(tuple(np.squeeze(history)))
        print("-------------------------------------------"*2)
        print("SUCCESS")
        return
        
    players_to_delete = irrational_decisions(matrix)
    for i in range(n_players):
        p_matrix = np.delete(matrix, list(players_to_delete[i]), axis=i)
    
        n_strategies_p = sum(p_matrix.shape[:-1])
        if n_strategies_p < n_strategies_prev:
            print("-------------------------------------------"*2)
            print("--"*depth + f"Player {i+1} won't play: {players_to_delete[i]}")
            history[i] = [x for x in history[i] if x not in players_to_delete[i]]
            dfs_dominant_strategies(p_matrix, history, depth+1)
    
if __name__ == "__main__":
    
    FOUND_SOLUTIONS = []
    
    l, u = -6, 6
    shape = [2, 2, 2]

    payoff = np.random.randint(l, u, size=(*shape, len(shape)))
    
    history = [list(range(shape[i])) for i in range(len(shape))]
    dfs_dominant_strategies(payoff, history=history)
    
    print(set(FOUND_SOLUTIONS))