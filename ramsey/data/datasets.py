"""Generates a dataset of random graphs and saves them to disk."""

import os

from absl import logging
from absl import app
from absl import flags
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ramsey import env_utils
from ramsey import clique_algorithms

FLAGS = flags.FLAGS
flags.DEFINE_string("output_dir", "random_graphs", "Directory to save generated graphs.")
flags.DEFINE_integer("num_graphs", 1000, "Number of graphs to generate.")
flags.DEFINE_integer("num_nodes", 10, "Number of nodes in each graph.")
flags.DEFINE_float("p_min", 0.1, "Minimum edge probability.")
flags.DEFINE_float("p_max", 0.5, "Maximum edge probability.")
flags.DEFINE_integer("batch_size", 1, "Number of graphs per saved batch.")


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

    def generate_dataset(self, num_graphs: int, num_nodes: int, p_min: float = 0.1, p_max: float = 0.5, batch_size: int = 1):
        """Generates and saves a dataset of random graphs.

        Each graph will have a random edge probability between `p_min` and
        `p_max`. Data is saved to disk in batches.
        """
        if num_graphs // batch_size != 0:
            logging.warning("Number of graphs is not a multiple of batch size. "
                            "The last batch may be smaller.")
        
        num_batches = (num_graphs + batch_size - 1) // batch_size
            
        for batch_number in tqdm(range(num_batches), desc="Generating batches"):
            start = batch_number * batch_size
            end = min(start + batch_size, num_graphs)
            batch = []

            for _ in range(start, end):
                edge_prob = torch.rand(1).item() * (p_max - p_min) + p_min
                graph = self.generate_random_graph(num_nodes, edge_prob)
                graph_complement = self._get_graph_complement(graph)
                graph_dict = env_utils.adj_matrix_to_dict(graph, 0)
                complement_graph_dict = env_utils.adj_matrix_to_dict(graph, 1)
                cliques = clique_algorithms.bron_kerbosch(graph_dict)
                complement_cliques = clique_algorithms.bron_kerbosch(complement_graph_dict)

                data = {
                    "adjacency": graph,
                    "complement": graph_complement,
                    "edge_prob": edge_prob,
                    "cliques": cliques,
                    "complement_cliques": complement_cliques
                }
                batch.append(data)

            batch_path = os.path.join(self.output_dir, f"graphs_batch_{batch_number:04d}.pt")
            torch.save(batch, batch_path)


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


def main(_):
    generator = NaiveDataGenerator(output_dir="random_graphs")
    generator.generate_dataset(num_graphs=FLAGS.num_graphs,
                               num_nodes=FLAGS.num_nodes,
                               p_min=FLAGS.p_min,
                               p_max=FLAGS.p_max,
                               batch_size=FLAGS.batch_size)


if __name__ == "__main__":
    app.run(main)