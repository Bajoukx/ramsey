"""Work in progress script to experiment with skrl library on Ramsey env."""

from absl import app
from absl import flags

import torch
import torch.nn as nn
from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.cem import CEM, CEM_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.models.torch import Model, CategoricalMixin

import gym_ramsey_game

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_vertices", 17, "Number of vertices in the complete graph K_n.")
flags.DEFINE_integer("n_red_edges", 4, "Number of red edges.")
flags.DEFINE_integer("n_blue_edges", 4, "Number of blue edges.")
flags.DEFINE_integer("number_of_players", 2, "Number of players.")
flags.DEFINE_string("render_mode", "animated", "Render mode: 'static', 'animated', or None.")
flags.DEFINE_string("device", "cpu", "Device to use: 'cpu' or 'cuda'.")


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

def main(argv):

    env = gym_ramsey_game.RamseyGymEnv(
        n_vertices=FLAGS.n_vertices,
        n_red_edges=FLAGS.n_red_edges,
        n_blue_edges=FLAGS.n_blue_edges,
        render_mode=FLAGS.render_mode,
        number_of_players=FLAGS.number_of_players,
        device=FLAGS.device
    )
    env = wrap_env(env, wrapper="gymnasium")
    device = env.device

    memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=device, replacement=False)

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
    app.run(main)