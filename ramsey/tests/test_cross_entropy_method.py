# pylint: disable=redefined-outer-name
"""Unit tests for cross_entropy_method module.

Covers PolicyNetwork architecture, CEM workflow functions (collect_trajectory,
collect_population, select_elite_by_fraction, train_on_elite), and tensor I/O
contracts.
"""

import torch
import torch.nn as nn

from ramsey.cross_entropy_method import (
    PolicyNetwork,
    Trajectory,
    collect_trajectory,
    collect_population,
    select_elite_by_fraction,
    train_on_elite,
)
from ramsey.gym_ramsey_env import Trajectory as GymTrajectory


class MockGymEnv:
    """Minimal Gymnasium-compatible environment for CEM tests.

    Runs for a fixed number of steps then terminates, recording each
    step in ``self.trajectory`` (a :class:`GymTrajectory`).
    """

    def __init__(self, obs_dim: int = 4, n_steps: int = 3) -> None:
        self.obs_dim = obs_dim
        self.n_steps = n_steps
        self._step_count = 0
        self.trajectory = GymTrajectory(observations=[], actions=[], rewards=[])

    def reset(self):
        self._step_count = 0
        self.trajectory = GymTrajectory(observations=[], actions=[], rewards=[])
        return torch.zeros(self.obs_dim), {}

    def step(self, action):
        self._step_count += 1
        obs = torch.zeros(self.obs_dim)
        reward = 1.0 if self._step_count >= self.n_steps else 0.0
        terminated = self._step_count >= self.n_steps
        truncated = False
        info = {}
        self.trajectory.add_step(obs, action, reward, info)
        return obs, reward, terminated, truncated, info


class TestPolicyNetworkForwardShape:
    """Verify (B, obs_dim) -> (B, action_dim) output shape contract."""

    def test_batched_forward_shape(self):
        obs_dim, action_dim, batch_size = 10, 4, 8
        net = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim)
        x = torch.randn(batch_size, obs_dim)
        out = net(x)
        assert out.shape == (batch_size, action_dim)

    def test_single_sample_forward_shape(self):
        obs_dim, action_dim = 6, 3
        net = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim)
        x = torch.randn(1, obs_dim)
        out = net(x)
        assert out.shape == (1, action_dim)

    def test_custom_hidden_size(self):
        obs_dim, action_dim = 8, 5
        net = PolicyNetwork(obs_dim=obs_dim,
                            action_dim=action_dim,
                            hidden_size=64)
        x = torch.randn(4, obs_dim)
        out = net(x)
        assert out.shape == (4, action_dim)


class TestPolicyNetworkActionProbsSumToOne:
    """Verify softmax output is a valid probability distribution."""

    def test_probs_sum_to_one_batch(self):
        obs_dim, action_dim = 8, 5
        net = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim)
        net.eval()
        x = torch.randn(4, obs_dim)
        probs = net.get_action_probs(x)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-6)

    def test_probs_non_negative(self):
        net = PolicyNetwork(obs_dim=6, action_dim=3)
        net.eval()
        x = torch.randn(10, 6)
        probs = net.get_action_probs(x)
        assert (probs >= 0).all()

    def test_probs_shape_matches_action_dim(self):
        obs_dim, action_dim = 10, 7
        net = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim)
        net.eval()
        x = torch.randn(3, obs_dim)
        probs = net.get_action_probs(x)
        assert probs.shape == (3, action_dim)


class TestPolicyNetworkSampleActionReturnsInt:
    """Verify that sample_action returns a Python int within a valid range."""

    def test_returns_python_int(self):
        obs_dim, action_dim = 6, 4
        net = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim)
        net.eval()
        x = torch.randn(1, obs_dim)
        action = net.sample_action(x)
        assert isinstance(action, int)

    def test_action_within_valid_range(self):
        obs_dim, action_dim = 6, 4
        net = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim)
        net.eval()
        x = torch.randn(1, obs_dim)
        for _ in range(30):
            action = net.sample_action(x)
            assert 0 <= action < action_dim


class TestCollectTrajectoryRollout:
    """Verify that collect_trajectory runs full episode and returns env."""

    def test_returns_same_env_instance(self):
        env = MockGymEnv(obs_dim=4, n_steps=3)
        policy = PolicyNetwork(obs_dim=4, action_dim=2)
        result = collect_trajectory(env, policy)
        assert result is env

    def test_trajectory_length_matches_steps(self):
        n_steps = 5
        env = MockGymEnv(obs_dim=4, n_steps=n_steps)
        policy = PolicyNetwork(obs_dim=4, action_dim=2)
        result = collect_trajectory(env, policy)
        assert len(result.trajectory.actions) == n_steps
        assert len(result.trajectory.observations) == n_steps
        assert len(result.trajectory.rewards) == n_steps

    def test_trajectory_actions_within_valid_range(self):
        obs_dim, action_dim = 4, 3
        env = MockGymEnv(obs_dim=obs_dim, n_steps=4)
        policy = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim)
        result = collect_trajectory(env, policy)
        for action in result.trajectory.actions:
            assert 0 <= action < action_dim

    def test_final_reward_is_recorded(self):
        env = MockGymEnv(obs_dim=4, n_steps=3)
        policy = PolicyNetwork(obs_dim=4, action_dim=2)
        result = collect_trajectory(env, policy)
        # MockGymEnv gives reward 1.0 only on the last step
        assert result.trajectory.rewards[-1] == 1.0


class TestCollectPopulationMultipleRollouts:
    """Verify population collection returns the correct number of trajectories."""

    def test_returns_correct_count(self):
        env = MockGymEnv(obs_dim=4, n_steps=3)
        policy = PolicyNetwork(obs_dim=4, action_dim=2)
        pop = collect_population(env, policy, population_size=5)
        assert len(pop) == 5

    def test_each_element_is_gym_trajectory(self):
        env = MockGymEnv(obs_dim=4, n_steps=3)
        policy = PolicyNetwork(obs_dim=4, action_dim=2)
        pop = collect_population(env, policy, population_size=4)
        for traj in pop:
            assert isinstance(traj, GymTrajectory)

    def test_trajectories_are_independent_objects(self):
        """Each collected trajectory must be a distinct object."""
        env = MockGymEnv(obs_dim=4, n_steps=3)
        policy = PolicyNetwork(obs_dim=4, action_dim=2)
        pop = collect_population(env, policy, population_size=3)
        # All trajectory objects should be distinct instances
        ids = [id(t) for t in pop]
        assert len(set(ids)) == len(ids)


class TestSelectEliteByFractionSorting:
    """Verify elite selection count and explicit final-reward sorting."""

    @staticmethod
    def _gym_traj(rewards_list):
        return GymTrajectory(observations=[], actions=[], rewards=rewards_list)

    def test_selects_correct_fraction(self):
        trajs = [self._gym_traj([float(i)]) for i in range(10)]
        elite = select_elite_by_fraction(trajs, elite_fraction=0.3)
        assert len(elite) == 3

    def test_minimum_one_elite_for_small_fraction(self):
        trajs = [self._gym_traj([1.0]), self._gym_traj([2.0])]
        elite = select_elite_by_fraction(trajs, elite_fraction=0.1)
        assert len(elite) == 1

    def test_full_fraction_returns_all(self):
        trajs = [self._gym_traj([float(i)]) for i in range(5)]
        elite = select_elite_by_fraction(trajs, elite_fraction=1.0)
        assert len(elite) == 5

    def test_list_rewards_sorted_by_final_reward(self):
        """Elite sorting uses the final episodic reward explicitly."""
        t_low = self._gym_traj([0.0, 1.0])
        t_mid = self._gym_traj([0.5, 0.5])
        t_high = self._gym_traj([1.0, 0.0])
        elite = select_elite_by_fraction([t_low, t_mid, t_high],
                                         elite_fraction=1 / 3)
        assert elite[0].rewards == [0.0, 1.0]

    def test_single_element_rewards_sorted_descending(self):
        trajs = [self._gym_traj([i * 0.5]) for i in range(5)
                ]  # 0.0, 0.5, 1.0, 1.5, 2.0
        elite = select_elite_by_fraction(trajs, elite_fraction=0.4)
        assert len(elite) == 2
        assert elite[0].rewards[0] >= elite[1].rewards[0]


class TestTrainOnEliteGradientSteps:
    """Verify that train_on_elite performs gradient updates and returns avg loss."""

    @staticmethod
    def _cem_traj(n_steps: int, obs_dim: int, action: int = 0):
        """Build a CEM Trajectory (cross_entropy_method.Trajectory)."""
        observations = [torch.randn(obs_dim) for _ in range(n_steps)]
        actions = [action] * n_steps
        return Trajectory(observations=observations, actions=actions, score=1.0)

    def test_returns_float(self):
        obs_dim, action_dim = 6, 3
        policy = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        traj = self._cem_traj(n_steps=5, obs_dim=obs_dim)
        loss = train_on_elite(policy, [traj],
                              optimizer,
                              criterion,
                              supervised_steps=2)
        assert isinstance(loss, float)

    def test_loss_is_non_negative(self):
        obs_dim, action_dim = 6, 3
        policy = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        traj = self._cem_traj(n_steps=5, obs_dim=obs_dim)
        loss = train_on_elite(policy, [traj],
                              optimizer,
                              criterion,
                              supervised_steps=3)
        assert loss >= 0.0

    def test_parameters_change_after_training(self):
        """Verifies that at least one weight tensor is updated by gradient descent."""
        obs_dim, action_dim = 6, 3
        policy = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim)
        params_before = [p.detach().clone() for p in policy.parameters()]
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()
        traj = self._cem_traj(n_steps=10, obs_dim=obs_dim)
        train_on_elite(policy, [traj], optimizer, criterion, supervised_steps=5)
        # detach().clone() produces requires_grad=False copies – compare values directly
        any_changed = any(
            not torch.equal(before, after)
            for before, after in zip(params_before, policy.parameters()))
        assert any_changed

    def test_average_loss_over_steps(self):
        """Returned value is average loss, not total, over supervised_steps.

        With dropout disabled (eval mode) and lr=0.0 (no weight updates) the
        per-step loss is constant, so the average must equal the single-step
        loss regardless of how many supervised_steps are requested.
        """
        obs_dim, action_dim = 4, 2
        policy = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim)
        policy.eval()  # disable dropout so forward pass is deterministic
        optimizer = torch.optim.SGD(policy.parameters(),
                                    lr=0.0)  # no weight update
        criterion = nn.CrossEntropyLoss()
        traj = self._cem_traj(n_steps=4, obs_dim=obs_dim)
        loss1 = train_on_elite(policy, [traj],
                               optimizer,
                               criterion,
                               supervised_steps=1)
        loss3 = train_on_elite(policy, [traj],
                               optimizer,
                               criterion,
                               supervised_steps=3)
        # Average over 1 step == average over 3 identical steps
        assert abs(loss1 - loss3) < 1e-5

    def test_multiple_trajectories_combined(self):
        obs_dim, action_dim = 6, 3
        policy = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        trajs = [self._cem_traj(n_steps=4, obs_dim=obs_dim) for _ in range(3)]
        loss = train_on_elite(policy,
                              trajs,
                              optimizer,
                              criterion,
                              supervised_steps=2)
        assert isinstance(loss, float) and loss >= 0.0


class TestCollectPopulationVectorized:
    """Verify vectorized population collection behavior."""

    def test_requires_factory_when_num_envs_gt_one(self):
        env = MockGymEnv(obs_dim=4, n_steps=3)
        policy = PolicyNetwork(obs_dim=4, action_dim=2)
        try:
            collect_population(env,
                               policy,
                               population_size=4,
                               num_envs=2,
                               env_factory=None)
            assert False, "Expected ValueError when env_factory is missing"
        except ValueError as exc:
            assert "env_factory" in str(exc)

    def test_vectorized_collection_returns_expected_count(self):
        policy = PolicyNetwork(obs_dim=4, action_dim=2)

        def make_env():
            return MockGymEnv(obs_dim=4, n_steps=3)

        pop = collect_population(env=None,
                                 policy=policy,
                                 population_size=7,
                                 num_envs=3,
                                 env_factory=make_env)
        assert len(pop) == 7

    def test_vectorized_collection_trajectories_have_steps(self):
        policy = PolicyNetwork(obs_dim=4, action_dim=2)

        def make_env():
            return MockGymEnv(obs_dim=4, n_steps=4)

        pop = collect_population(env=None,
                                 policy=policy,
                                 population_size=5,
                                 num_envs=2,
                                 env_factory=make_env)
        assert all(len(traj.actions) == 4 for traj in pop)


class TestTrainOnEliteEmptyTrajectories:
    """Verify graceful handling of empty elite list and empty trajectory steps."""

    def test_empty_trajectory_list_returns_zero(self):
        policy = PolicyNetwork(obs_dim=6, action_dim=3)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        loss = train_on_elite(policy, [],
                              optimizer,
                              criterion,
                              supervised_steps=3)
        assert loss == 0.0

    def test_trajectory_with_no_steps_returns_zero(self):
        policy = PolicyNetwork(obs_dim=6, action_dim=3)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        empty_traj = Trajectory(observations=[], actions=[], score=0.0)
        loss = train_on_elite(policy, [empty_traj],
                              optimizer,
                              criterion,
                              supervised_steps=3)
        assert loss == 0.0
