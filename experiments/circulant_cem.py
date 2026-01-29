"""Neural network CEM for finding Ramsey graphs in the circulant graph space.

Circulant graphs are defined by chord lengths. Given numbered vertices, each
vertex i has an edge to vertices (i + k) mod n for each chord length k. This
significantly reduces the search space compared to arbitrary graphs.

This version uses a neural network policy trained via cross-entropy method,
as opposed to the direct probability-based CirculantCEM approach.
"""

from absl import app
from absl import flags
from absl import logging

import torch
import torch.nn as nn

from ramsey import cross_entropy_method
from ramsey import gym_ramsey_env
from ramsey import rendering
from ramsey import rewards

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_vertices", 17, "Number of vertices")
flags.DEFINE_list("clique_sizes", [4, 4], "Clique sizes for each color")
flags.DEFINE_integer("num_iterations", 1000, "Number of CEM iterations")
flags.DEFINE_integer("population_size", 128, "Population size for CEM")
flags.DEFINE_float("elite_fraction", 0.1, "Fraction of elite samples")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate for supervised update")
flags.DEFINE_integer("supervised_steps", 5,
                     "Gradient steps for supervised update")
flags.DEFINE_integer("hidden_size", 256, "Hidden layer size")
flags.DEFINE_string("device", "cpu", "Device: 'cpu' or 'cuda'")


def log_solution(adjacency_vec: torch.Tensor, n_vertices: int,
                 clique_sizes: list):
    """Log details about a graph solution.

    Args:
        adjacency_vec: Flattened upper triangular adjacency vector.
        n_vertices: Number of vertices.
        clique_sizes: Maximum clique sizes for each color.
    """
    logging.info(
        "Graph Solution for R(%d,%d) on %d vertices",
        clique_sizes[0],
        clique_sizes[1],
        n_vertices,
    )

    logging.info("Final adjacency vector: %s", adjacency_vec.tolist())


def main(_):
    """Main function to run neural network CEM on circulant graphs."""
    device = FLAGS.device

    n_vertices = FLAGS.n_vertices
    clique_sizes = [int(size) for size in FLAGS.clique_sizes]

    logging.info(
        "Searching for R(%d,%d) circulant graph on %d vertices",
        clique_sizes[0],
        clique_sizes[1],
        n_vertices,
    )
    logging.info("Population size: %d", FLAGS.population_size)
    logging.info("Elite fraction: %s", FLAGS.elite_fraction)
    logging.info("Iterations: %d", FLAGS.num_iterations)

    # Setup reward strategy
    reward_strategy = rewards.ColorSumRewardStrategy(
        cumulative=False,
        reward_colors=[0, 1],
        max_clique_sizes=clique_sizes,
        reward_loss=0.0,
        reward_success=1.0,
    )

    # Create circulant environment
    env = gym_ramsey_env.RamseyGymEnvV2(n_vertices=n_vertices,
                                        clique_sizes=clique_sizes,
                                        device=device,
                                        init_method_name="uncolored",
                                        reward_strategy=reward_strategy)

    # Observation: flattened adjacency vector
    # Action: chord_length * n_colors (select chord and color)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    logging.info("Observation dim: %d, Action dim: %d", obs_dim, action_dim)

    # Create policy network
    policy = cross_entropy_method.PolicyNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=FLAGS.hidden_size,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(policy.parameters(), lr=FLAGS.learning_rate)

    best_reward = float("-inf")
    best_trajectory = None

    # Main CEM loop
    for iteration in range(1, FLAGS.num_iterations + 1):
        # Collect population of trajectories
        trajectories = cross_entropy_method.collect_population(
            env=env,
            policy=policy,
            population_size=FLAGS.population_size,
            device=device,
        )

        # Track best trajectory
        current_best = max(trajectories, key=lambda t: t.rewards)
        current_best_reward = current_best.rewards[-1]
        logging.debug("Current best reward: %.4f", current_best_reward)
        logging.debug("Current best adjacency: %s",
                      current_best.observations[-1].tolist())
        logging.debug("Current best info: %s", current_best.info)

        if current_best_reward > best_reward:
            best_reward = current_best_reward
            best_trajectory = current_best
            logging.info(
                "Iteration %d: New best reward: %.4f",
                iteration,
                best_reward,
            )

        # Check if we found a counterexample
        if current_best.info["is_counterexample"]:
            logging.info("SUCCESS: Counterexample found at iteration %d!",
                         iteration)
            rendering.render_graph_from_adj_vec(
                current_best.observations[-1], n_vertices)
            break

        # Select elite trajectories
        elite_trajectories = cross_entropy_method.select_elite_by_fraction(
            trajectories=trajectories, elite_fraction=FLAGS.elite_fraction)

        # Train policy on elite trajectories
        avg_loss = cross_entropy_method.train_on_elite(
            policy=policy,
            elite_trajectories=elite_trajectories,
            optimizer=optimizer,
            criterion=criterion,
            supervised_steps=FLAGS.supervised_steps,
            device=device)

        # Log progress
        reward_list = [t.rewards[-1] for t in trajectories]
        elite_rewards = [t.rewards[-1] for t in elite_trajectories]
        if iteration % 10 == 0 or iteration == 1:
            logging.info("Iteration %d: Mean=%.4f, Elite mean=%.4f, Loss=%.4f",
                         iteration,
                         sum(reward_list) / len(reward_list),
                         sum(elite_rewards) / len(elite_rewards), avg_loss)

    # log final results
    if best_trajectory is not None:
        final_adjacency = best_trajectory.observations[-1]
        log_solution(final_adjacency, n_vertices, clique_sizes)

    if "is_counterexample" in best_trajectory.info:
        logging.info("SUCCESS: Found valid Ramsey coloring! Best score: %s",
                     best_reward)
    else:
        logging.info("No valid coloring found. Best score: %s", best_reward)


if __name__ == "__main__":
    app.run(main)
