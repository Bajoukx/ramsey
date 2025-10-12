import itertools
import torch
from typing import Optional, Tuple, List, Union, Dict
import numpy as np
import networkx
from torch import Tensor
from torch_geometric.data import Data
import torch.nn as nn
import matplotlib.pyplot as plt
import collections
import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium
from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.cem import CEM, CEM_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.models.torch import Model, CategoricalMixin


import gym_ramsey_game

def test_stable_baselines():
    env = gym_ramsey_game.RamseyGymEnv(n_vertices=17,
                                       n_red_edges=4,
                                       n_blue_edges=4)
    model = stable_baselines3.A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

def test_skrl():
    class Policy(CategoricalMixin, Model):
        def __init__(self, observation_space, action_space, device="cpu", unnormalized_log_prob=True):
            Model.__init__(self, observation_space, action_space, device)
            CategoricalMixin.__init__(self, unnormalized_log_prob)

            self.net = nn.Sequential(
                nn.Linear(self.observation_space.shape[0], 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_actions)
            )

        def compute(self, inputs, role):
            mask_invalid = True
            if not mask_invalid:
                logits = self.net(inputs["states"])
            else:
                x = inputs["states"]
                if x.dtype != torch.float32:
                    x = x.float()
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                logits = self.net(x)

                # Mask invalid actions
                # x: (batch_size, n_edges), logits: (batch_size, 2*n_edges)
                batch_size, n_edges = x.shape
                for i in range(batch_size):
                    for edge in range(n_edges):
                        if x[i, edge] != -1:
                            # Mask both colors for this edge
                            logits[i, edge] = -1e9
                            logits[i, edge + n_edges] = -1e9
            return logits, {}

    env = gym_ramsey_game.RamseyGymEnv(
        n_vertices=17,
        n_red_edges=4,
        n_blue_edges=4,
        render_mode="animated",
        device="cpu"
    )
    env = wrap_env(env, wrapper="gymnasium")
    device = env.device

    memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=device, replacement=False)

    """policy = Policy(env.observation_space,
                    env.action_space, device=device,
                    unnormalized_log_prob=True)"""
    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, device)
    for model in models.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

    cfg = CEM_DEFAULT_CONFIG.copy()
    cfg["random_timesteps"] = 1000
    cfg["learning_starts"] = 100
    cfg["rollouts"] = 1000
    cfg["experiment"]["directory"] = "runs/ramsey_cem"  # logging

    cfg["experiment"]["write_interval"] = 1000
    cfg["experiment"]["checkpoint_interval"] = 5000
    cfg["experiment"]["directory"] = "runs/ramsey_cem"

    agent = CEM(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
        )
    agent.track_data("episode_rewards", env.episode_rewards)

    trainer = SequentialTrainer(
        cfg={"timesteps": 20000},
        env=env,
        agents=agent
    )

    trainer.train()

if __name__ == "__main__":
    test_skrl()