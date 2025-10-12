"""Generates a dataset of random graphs and saves them to disk."""

import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import operations


class NaiveDataGenerator:
    """Generates and saves a dataset of random graphs."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_random_graph(self, num_nodes: int, max_edge_prob: float = 0.25):
        """Generate a random undirected graph represented by an adjacency matrix.
        
        Each possible edge has a random probability less than `max_edge_prob`.
        """
        adjacency = torch.rand(num_nodes, num_nodes)
        adjacency = (adjacency < max_edge_prob).float()
        adjacency = torch.triu(adjacency, diagonal=1)
        adjacency = adjacency + adjacency.T
        
        # No self-loops
        adjacency.fill_diagonal_(0)
        return adjacency
    
    def _get_graph_complement(self, adjacency: torch.Tensor):
        """Returns the complement of the given graph adjacency matrix."""
        complement = 1 - adjacency
        complement.fill_diagonal_(0)
        return complement

    def generate_dataset(self, num_graphs: int, num_nodes: int, p_min: float = 0.1, p_max: float = 0.5, tensor_clique_size: int = 0):
        """Generates and saves a dataset of random graphs.

        Each graph will have a random edge probability between `p_min` and `p_max`.
        """
        if tensor_clique_size > 0:
            tensor_clique_name = f"tensor_clique_{tensor_clique_size}"
            complement_tensor_clique_name = f"complement_tensor_clique_{tensor_clique_size}"

        dataset = []
        for i in tqdm(range(num_graphs)):
            edge_prob = torch.rand(1).item() * (p_max - p_min) + p_min
            graph = self.generate_random_graph(num_nodes, edge_prob)
            graph_complement = self._get_graph_complement(graph)
            data = {
                "adjacency": graph,
                "complement": graph_complement,
                "edge_prob": edge_prob
            }

            if tensor_clique_size > 0:
                tensor_clique = operations.get_tensor_clique(graph, tensor_clique_size)
                complement_tensor_clique = operations.get_tensor_clique(graph_complement, tensor_clique_size)
                data[tensor_clique_name] = tensor_clique
                data[complement_tensor_clique_name] = complement_tensor_clique

            dataset.append(data)
            torch.save(data, os.path.join(self.output_dir, f"graph_{i}.pt"))
        
        # Optionally save all graphs together
        torch.save(dataset, os.path.join(self.output_dir, "graph_dataset.pt"))
        print(f"✅ Saved {num_graphs} graphs to '{self.output_dir}'.")


class RandomGraphDataset(Dataset):
    """A PyTorch Dataset for loading the generated random graphs."""

    def __init__(self, data_dir: str):
        self.data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt') and f != "graph_dataset.pt"]
        self.data_files.sort()

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = torch.load(self.data_files[idx])
        return data


if __name__ == "__main__":
    generator = NaiveDataGenerator(output_dir="random_graphs")
    generator.generate_dataset(num_graphs=100, num_nodes=10, p_min=0.1, p_max=0.5)