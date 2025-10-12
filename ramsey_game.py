"""Ramsey Game Environment for RL using PyTorch Geometric

A Ramsey Game is inpired by the Ramsey's Theorem. Two players take turns
coloring a red and blue graph with n vertices. The winner is the first person to
color a clique of size 

This file provides RamseyEnv:
- State: A 2_coloring of a ramsey game in red and blue: 
    -1  => uncolored
     0  => color red
     1  => color blue

- Action: An action is the coloring of an edge. Since coloring can be either
  red or blue, the number of possible actions is two times the number of edges.
  In other to ensure compatibility with other python packages, a coloring of
  edge with index i is encoded into [0, 2*n_edges[, where the first n_edges
  are red and the rest blue.
"""

from typing import Optional, Tuple, List, Union
import itertools

import torch
from torch import Tensor
import networkx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gymnasium

import clique_algorithms
import rewards
import utils


class RamseyEnv():
    """Gym-like environment for Ramsey Games."""

    def __init__(
        self,
        n_vertices: int,
        n_red_edges: int,
        n_blue_edges: int,
        number_of_players: int = 2,
        device: Optional[Union[str, torch.device]] = None
    ) -> None:
        """Initialize Ramsey Environment."""
        if number_of_players not in (1, 2):
            raise ValueError("number_of_players must be either 1 or 2.")
        self.number_of_players = number_of_players

        assert n_vertices >= 1
        self.n_vertices = n_vertices
        self.n_red_vertices = n_red_edges
        self.n_blue_vertices = n_blue_edges
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.all_edges = list(itertools.combinations(range(n_vertices), 2))  # list of all edges
        self.n_edges = len(self.all_edges)

        edge_index = torch.tensor(self.all_edges, dtype=torch.long).t().contiguous()
        self.base_edge_index = edge_index.to(self.device)  # shape [2, n_edges]
        
        self.reset()

    def _init_edges(self) -> Tensor:
        """Initializes the graph edges."""
        if self.number_of_players == 2:
            return -1 * torch.ones(self.n_edges, dtype=torch.float, device=self.device)
        if self.number_of_players == 1:
            return utils.get_initial_random_coloring(self.n_edges, self.device)

    def reset(self):
        """Resets environment."""
        self.colored_edges = self._init_edges()
        self.done = False
        self.info = {}
        self.steps = 0
        return self.colored_edges

    def encode_action(self, edge_idx: int, color: int) -> int:
        """Encode (edge_idx, color) -> single integer in [0, E*2[."""
        assert 0 <= edge_idx < self.n_edges
        assert color in (0, 1)
        return edge_idx + color * self.n_edges

    def decode_action(self, action: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        """Decode action into (edge_idx, color). Accepts already a tuple."""
        if isinstance(action, tuple) or isinstance(action, list):
            edge_idx, color = action
            return int(edge_idx), int(color)
        edge_idx = action % self.n_edges
        color = action // self.n_edges
        return edge_idx, color

    def list_legal_actions(self) -> List[int]:
        """Return list of only uncolored edges."""
        uncolored = (self.colored_edges == -1).nonzero(as_tuple=False).view(-1).cpu().tolist()
        return sorted([self.encode_action(e, c) for e in uncolored for c in (0, 1)])

    def _edjes_to_adj_dict(self, color: int):
        """Convert colored edges to adjacency dict for given color."""
        colored_edges_idx = (self.colored_edges == color).nonzero(as_tuple=True)[0]
        adj_dict = {}
        for idx in colored_edges_idx:
            u, v = self.base_edge_index[:, idx].tolist()
            u, v = str(u), str(v)

            adj_dict.setdefault(u, set()).add(v)
            adj_dict.setdefault(v, set()).add(u)
        return adj_dict
    
    def has_max_clique(self, color: int, size: int) -> bool:
        """Uses the Bron-Kerbosch algorithm to check if the maximal clique size is in the graph."""
        adjacency_dict = self._edjes_to_adj_dict(color=color)
        cliques = clique_algorithms.bron_kerbosch(adjacency_dict)
        if cliques:
            len_cliques = [len(clique) for clique in cliques]
            if max(len_cliques) >= size:
                #print('found clique:', cliques)
                return True
        else:
            return False

    def step(self, action: Union[int, Tuple[int, int]]):
        """Apply action. Returns (state, reward, done, info).

        Rewarding scheme:
          - invalid action (edge already colored): invalid_action_penalty
          - creating monochromatic clique: terminal_reward_success and done
          - fully coloring without forbidden cliques: terminal_reward_loss and done
          - otherwise: 0 reward and continue
        """
        print("action", action)
        if self.done:
            raise RuntimeError("Episode has finished. Call reset().")
        action_edge_idx, color = self.decode_action(action)
        self.steps += 1

        if self.number_of_players == 2:
            if rewards.is_invalid_action(self, action_edge_idx):
                reward, done, info = rewards.invalid_action_reward(invalid_action_penalty=-1000)
                return self.colored_edges, reward, done, info

        # TODO: make reward type a parameter 
        reward_type = "hoffman"
        if reward_type == "simple":            
            #color the edge
            self.colored_edges[action_edge_idx] = int(color)
            reward, done, info = rewards.simple_reward(self, color, terminal_reward_loss=-1.0, terminal_reward_success=1.0)  #TODO: make params
        #print(color)
        if reward_type == "hoffman":
            self.colored_edges[action_edge_idx] = int(color)
            reward, done, info = rewards.hoffman_simple_reward(self, color, 8)  #TODO: make d-regular degree a parameter
        return self.colored_edges, reward, done, info


if __name__ == '__main__':
    # Small example: attempt to color edges of K_5 to avoid red K3 and blue K3 (classic Ramsey R(3,3)=6)
    env = RamseyEnv(n_vertices=5, n_red_edges=3, n_blue_edges=3, number_of_players=2)
    state = env.reset()

    # naive random rollouts
    import random

    done = False
    total_reward = 0.0
    while not done:
        legal = env.list_legal_actions()
        a = random.choice(legal)
        state, rwd, done, info = env.step(a)
        total_reward += rwd
        if rwd != 0:
            print("Reward", rwd, info)
    print("Episode done, total reward", total_reward)
