"""Implementation of CEM algorithm using torchrl library."""

from absl import app
from absl import flags

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.envs import GymWrapper

from ramsey import gym_ramsey_env
from ramsey import rewards

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_vertices", 17, "Number of vertices")
flags.DEFINE_integer("max_clique_size", 4, "Max clique size")
flags.DEFINE_integer("num_iterations", 10, "Number of training iterations")
flags.DEFINE_integer("population_size", 128, "Population size for CEM")
flags.DEFINE_float("elite_fraction", 0.1, "Fraction of elite samples")
flags.DEFINE_string("device", "cpu", "Device: 'cpu' or 'cuda'")


class PolicyNetwork(nn.Module):
    """Simple policy network."""
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class CEMAgent:
    """Cross-Entropy Method agent for TorchRL environments."""
    
    def __init__(self, obs_dim, action_dim, device, population_size=128, elite_fraction=0.1):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.population_size = population_size
        self.n_elite = max(1, int(population_size * elite_fraction))
        
        self.policy = PolicyNetwork(obs_dim, action_dim).to(device)
        self.noise_std = 1.0
        self.min_noise_std = 0.01
        self.noise_decay = 0.99
    
    def generate_population(self):
        """Generate population with perturbed parameters."""
        population = []
        for _ in range(self.population_size):
            perturbed = PolicyNetwork(self.obs_dim, self.action_dim).to(self.device)
            with torch.no_grad():
                for mean_p, pert_p in zip(self.policy.parameters(), perturbed.parameters()):
                    noise = torch.randn_like(mean_p) * self.noise_std
                    pert_p.data = mean_p.data + noise
            population.append(perturbed)
        return population
    
    def evaluate_policy(self, policy, env, max_steps=200):
        """Evaluate a policy."""
        tensordict = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            obs = tensordict["observation"].to(self.device)
            
            with torch.no_grad():
                logits = policy(obs)
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze()
            
            tensordict["action"] = action.cpu()
            tensordict = env.step(tensordict)
            
            # TorchRL GymWrapper stores reward in 'next' sub-dict
            if "next" in tensordict.keys() and "reward" in tensordict["next"].keys():
                reward = tensordict["next"]["reward"].item()
            else:
                # Fallback if structure is different
                reward = 0.0
            
            done = tensordict.get("done", torch.tensor(False)).item() or \
                   tensordict.get("terminated", torch.tensor(False)).item()
            
            total_reward += reward
            steps += 1
        
        return total_reward, steps
    
    def update_policy(self, elite_policies):
        """Update mean policy from elites."""
        with torch.no_grad():
            for param_idx, mean_param in enumerate(self.policy.parameters()):
                elite_params = torch.stack([
                    list(policy.parameters())[param_idx].data 
                    for policy in elite_policies
                ])
                mean_param.data = elite_params.mean(dim=0)
        
        self.noise_std = max(self.min_noise_std, self.noise_std * self.noise_decay)


def main(_):
    device = torch.device(FLAGS.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup environment
    clique_sizes = [FLAGS.max_clique_size, FLAGS.max_clique_size]
    reward_strategy = rewards.SimpleRewardStrategy(
        max_clique_size=max(clique_sizes),
        cumulative=False,
        reward_colors=[0, 1]
    )

    env = gym_ramsey_env.RamseyGymEnv(
        n_vertices=FLAGS.n_vertices,
        clique_sizes=clique_sizes,
        reward_strategy=reward_strategy,
        init_method_name="empty",
        render_mode=None,
        device="cpu"
    )
    
    env = GymWrapper(env, device="cpu")
    
    observation_size = env.observation_spec["observation"].shape[-1]
    action_size = env.action_spec.space.n
    
    print(f"\nEnvironment Info:")
    print(f"  Observation size: {observation_size}")
    print(f"  Action size: {action_size}")
    print(f"  Vertices: {FLAGS.n_vertices}")
    print(f"  Clique sizes: {clique_sizes}")
    
    # Create CEM agent
    agent = CEMAgent(
        obs_dim=observation_size,
        action_dim=action_size,
        device=device,
        population_size=FLAGS.population_size,
        elite_fraction=FLAGS.elite_fraction
    )
    
    print(f"\nCEM Configuration:")
    print(f"  Population size: {FLAGS.population_size}")
    print(f"  Elite fraction: {FLAGS.elite_fraction}")
    print(f"  Elite count: {agent.n_elite}")
    print(f"  Iterations: {FLAGS.num_iterations}")
    
    print("\nStarting training...\n")
    best_reward = float('-inf')
    
    for iteration in range(FLAGS.num_iterations):
        population = agent.generate_population()
        
        rewards_list = []
        for policy in population:
            reward, steps = agent.evaluate_policy(policy, env)
            rewards_list.append(reward)
        
        # Select elites
        sorted_indices = np.argsort(rewards_list)[::-1]
        elite_indices = sorted_indices[:agent.n_elite]
        elite_policies = [population[i] for i in elite_indices]
        elite_rewards = [rewards_list[i] for i in elite_indices]
        
        # Update policy
        agent.update_policy(elite_policies)
        
        # Statistics
        mean_reward = np.mean(rewards_list)
        max_reward = np.max(rewards_list)
        elite_mean = np.mean(elite_rewards)
        
        if max_reward > best_reward:
            best_reward = max_reward
        
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"Iteration {iteration+1}/{FLAGS.num_iterations} | "
                  f"Mean: {mean_reward:.2f} | "
                  f"Max: {max_reward:.2f} | "
                  f"Elite Mean: {elite_mean:.2f} | "
                  f"Best: {best_reward:.2f} | "
                  f"Noise: {agent.noise_std:.4f}")
    
    print(f"\nTraining completed! Best reward: {best_reward:.2f}")
    
    # Final evaluation
    print("\nFinal evaluation (10 episodes)...")
    eval_rewards = []
    for i in range(10):
        reward, steps = agent.evaluate_policy(agent.policy, env, max_steps=200)
        eval_rewards.append(reward)
        print(f"  Episode {i+1}: Reward={reward:.2f}, Steps={steps}")
    
    print(f"\nEvaluation mean: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")


if __name__ == "__main__":
    app.run(main)