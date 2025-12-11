"""SKRL experiments for Ramsey problem."""

from absl import app
from absl import flags
import psutil

import torch
import torch.nn as nn
from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.cem import CEM, CEM_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.models.torch import Model, CategoricalMixin

from ramsey import gym_ramsey_env
from ramsey import rewards

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_vertices", 17,
                     "Number of vertices in the complete graph K_n.")
flags.DEFINE_integer("n_red_edges", 4, "Number of red edges.")
flags.DEFINE_integer("n_blue_edges", 4, "Number of blue edges.")
flags.DEFINE_string("render_mode", "animated",
                    "Render mode: 'static', 'animated', or None.")
flags.DEFINE_string("device", "cpu", "Device to use: 'cpu' or 'cuda'.")


class Policy(CategoricalMixin, Model):

    def __init__(self,
                 observation_space,
                 action_space,
                 device="cpu",
                 unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.net = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, self.num_actions))

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
        return logits, {}


def main(_):
    clique_sizes = [FLAGS.n_red_edges, FLAGS.n_blue_edges]
    reward_strategy = rewards.SimpleRewardStrategy(
        max_clique_size=max(clique_sizes), cumulative=False, reward_colors=[0, 1])
    env = gym_ramsey_env.RamseyGymEnv(n_vertices=FLAGS.n_vertices,
                                      clique_sizes=clique_sizes,
                                      init_method_name="empty",
                                      init_params=None,
                                      reward_strategy=reward_strategy,
                                      render_mode=FLAGS.render_mode,
                                      device=FLAGS.device)
    env = wrap_env(env, wrapper="gymnasium")

    memory = RandomMemory(memory_size=100000,
                          num_envs=1,
                          device=FLAGS.device,
                          replacement=False)

    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space,
                              FLAGS.device)
    for model in models.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

    cfg = CEM_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 5000
    cfg["percentile"] = 0.1  # select the top 10% episodes

    cfg["random_timesteps"] = 5000
    cfg["learning_starts"] = 5000

    cfg["experiment"]["directory"] = "runs/ramsey_cem"
    cfg["experiment"]["write_interval"] = 1
    cfg["experiment"]["checkpoint_interval"] = 5000

    agent = CEM(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=FLAGS.device)

    trainer = SequentialTrainer(cfg={"timesteps": 200000},
                                env=env,
                                agents=agent)

    trainer.train()


if __name__ == "__main__":
    app.run(main)
