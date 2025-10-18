# Ramsey Nummber's project

This repository is intended to explore the [Ramsey's Numbers problem](https://en.wikipedia.org/wiki/Ramsey%27s_theorem#Ramsey_numbers) and it is a work in progress.

# Reinforcement learning approach

[Wagner](https://arxiv.org/abs/2104.14516) showed that reinforcement learning can be used in combinatorics and graph theory.

Even more so, the approach has been used concretly to find new bounds for the Ramsey's Numbers problem in [Gheble et al.](https://arxiv.org/abs/2403.20055).

### RL approach tasks and TODO:
 - [x] Create a ramsey "game" environment. Where two players take turns to paint the edges of a graph with the goal of not creating cliques of a specific size.
 - [x] RL policy that learns how to color a graph with two colors without making invalid moves i.e. repainting a colored edge and painting the same color twice in a row.
 - [ ] Verify the agent's Cross-Entropy method.
 - [ ] Explore and document strategies for reward function shapping.


# Dataset Creation

One of the goals of this project is also to create a large dataset in order to infer relationships between ramsey's numbers graphs. This is also useful so others can do their own exploration of the problem.

### Dataset approach tasks and Todo:

 - [x] Create a naive implementation of random graph generation and wwriting to disk.
 - [ ] Explore the dataset and identify relationships.

 # Installation

 Creating a virtual environment is recommended. Install in development mode
 with:
 
 `pip install -e .`
