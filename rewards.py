import torch

def is_invalid_action(ramsey_env, edge_idx):
    """Checks if the action is invalid (i.e., the edge is already colored)."""
    return ramsey_env.colored_edges[edge_idx].item() != -1

def invalid_action_reward(invalid_action_penalty: float = -0.1):
    reward = invalid_action_penalty
    done = True
    return reward, done, {"invalid": True}

def simple_reward(ramsey_env,
                  action_color,
                  terminal_reward_loss: float = -1.0,
                  terminal_reward_success: float = 1.0):
    """Computes simple reward.

    Rewarding scheme:
        - invalid action (edge already colored): invalid_action_penalty
        - creating monochromatic clique: terminal_reward_success and done
        - fully coloring without forbidden cliques: terminal_reward_loss and done
        - otherwise: 0 reward and continue
    """
    found_max_clique = ramsey_env.has_max_clique(action_color, ramsey_env.n_red_vertices)
    if found_max_clique:
        done = True
        reward = terminal_reward_loss
        return reward, done, {"violation_color": action_color}
    
    # check if all colored
    if (ramsey_env.colored_edges == -1).sum().item() == 0:
        done = True
        reward = terminal_reward_success
        return reward, done, {"All colored": action_color}
    
    # non-terminal
    return 0.0, False, {}

def hoffman_wip_reward(ramsey_env, action_color: int, regular_degree: int):
    """Computes Hoffman simple reward.
    
    Assuming that G is a d-regular graph, then the Hoffman bound states that for the independence number \alpha(G):
    \alpha(G) <= n * (-lambda_min) / (d - lambda_min)
    where lambda_min is the smallest eigenvalue of the adjacency matrix of G and \alpha is the independence number.

    The simple Hoffman reward function is defined as:
    f(G) = -avg_degree(G) + \beta * min(0, smallest_eigenvalue(G) - (n_vertives(G) * d / (n_vertices(G) - d)))
    
    In practice, we first create the adjacency matrix for the action_color, then we use torch to compute
    the smallest eigenvalue.
    """
    adj = torch.zeros((ramsey_env.n_vertices, ramsey_env.n_vertices), dtype=torch.float)
    edge_indices = (ramsey_env.colored_edges == action_color).nonzero(as_tuple=True)[0]
    for idx in edge_indices:
        u, v = ramsey_env.all_edges[idx]
        adj[u, v] = 1.0
        adj[v, u] = 1.0

    degrees = adj.sum(dim=1)
    
    avg_degree = degrees.mean().item()
    """if degrees.min().item() < regular_degree:
        # not a d-regular graph
        return -avg_degree, False, {"not_regular": True}"""
    
    # compute smallest eigenvalue
    eigenvalues = torch.linalg.eigvalsh(adj)
    smallest_eigenvalue = eigenvalues[0].real.item()
    
    beta = 1.0  # scaling factor for the eigenvalue term
    reward = -avg_degree + beta * min(0, smallest_eigenvalue - (ramsey_env.n_vertices * regular_degree / (ramsey_env.n_vertices - regular_degree)))
    #print('reward:', reward, 'avg_degree:', avg_degree, 'smallest_eigenvalue:', smallest_eigenvalue)
    
    # check terminal conditions
    found_max_clique = ramsey_env.has_max_clique(action_color, ramsey_env.n_red_vertices)
    if found_max_clique:
        done = True
        return reward, done, {"violation_color": action_color}

    return reward, False, {}

def hoffman_simple_reward(ramsey_env, action_color: int, regular_degree: int):
    """Computes the Hoffman simle reward.
    
    The Hoffman bound states that for a d-regular graph, the independence number \alpha(G) is bounded by:
    \alpha(G) <= n * (-lambda_min) / (d - lambda_min)
    where lambda_min is the smallest eigenvalue of the adjacency matrix of G.
    """
    adjacency_matrix = ramsey_env._edges_to_adjacency_tensor(action_color)
    eigenvalue_min = torch.linalg.eigvalsh(adjacency_matrix).min().item()
    hoffman = ramsey_env.n_vertices * (-eigenvalue_min) / (regular_degree - eigenvalue_min)
    reward = -hoffman

    # check terminal conditions
    found_max_clique = ramsey_env.has_max_clique(action_color, ramsey_env.n_red_vertices)
    if found_max_clique:
        done = True
        return reward, done, {"violation_color": action_color}
    return -hoffman, False, {}


    