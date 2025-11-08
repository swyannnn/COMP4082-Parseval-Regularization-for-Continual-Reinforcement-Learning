import gymnasium as gym
from gymnasium.wrappers import (
    DtypeObservation, TimeLimit, RecordEpisodeStatistics, ClipAction
)
import numpy as np
import math


class ContinualPendulumSequence:
    def __init__(self,
                 change_freq=200000,
                 normalize_obs="straight",
                 normalize_avg_coef=0.0001,
                 normalize_rewards=True,
                 reset_obs_stats=False,
                 bias_correction=False,
                 seed=None):
        """
        ContinualPendulumSequence:
        A continual-learning version of the Pendulum-v1 environment
        where the task changes every `change_freq` steps by altering
        physical parameters (gravity and torque limit).

        Tasks (T1–T4):
            T1: low gravity, strong torque
            T2: low gravity, weak torque
            T3: high gravity, strong torque
            T4: high gravity, weak torque
        """

        self.change_freq = int(change_freq)
        self.normalize_obs = normalize_obs
        self.normalize_avg_coef = normalize_avg_coef
        self.normalize_rewards = normalize_rewards
        self.reset_obs_stats = reset_obs_stats
        self.bias_correction = bias_correction
        self.seed = seed

        # Track time and task
        self.timestep_counter = 0
        self.task_counter = 0

        # Define task parameters
        self.task_list = [
            {"g": 7.0,  "max_torque": 2.0},  # T1
            {"g": 7.0,  "max_torque": 1.0},  # T2
            {"g": 12.0, "max_torque": 2.0},  # T3
            {"g": 12.0, "max_torque": 1.0},  # T4
        ]
        self.current_task = 0

        # Create environment
        self.env = gym.make("Pendulum-v1")
        self.env = self._wrap_env(self.env)

        # Initialize obs stats
        obs_dim = self.env.observation_space.shape
        self.obs_mean = np.zeros(obs_dim, dtype=np.float64)
        self.obs_var = np.ones(obs_dim, dtype=np.float64)
        self.obs_count = 1e-4

        # Seed for reproducibility
        if self.seed is not None:
            self.env.reset(seed=self.seed)
            self.env.action_space.seed(self.seed)
            self.env.observation_space.seed(self.seed)

        # Apply first task parameters
        self._apply_task_params()

        print(f"[Task 1] g={self.task_list[0]['g']}, max_torque={self.task_list[0]['max_torque']}")

    # ---------------------------------------------------------------------
    # Wrappers
    # ---------------------------------------------------------------------
    def _wrap_env(self, env):
        env = DtypeObservation(env, dtype=np.float32)
        env = ClipAction(env)
        if self.normalize_rewards:
            env = gym.wrappers.TransformReward(env, lambda r: r / 500)
        env = TimeLimit(env, max_episode_steps=200)
        env = RecordEpisodeStatistics(env)
        return env

    # ---------------------------------------------------------------------
    # Task logic
    # ---------------------------------------------------------------------
    def _apply_task_params(self):
        """Apply current task’s gravity and torque settings."""
        params = self.task_list[self.current_task]
        env_unwrapped = self.env.unwrapped
        env_unwrapped.g = params["g"]
        env_unwrapped.max_torque = params["max_torque"]

    def _advance_task(self):
        """Switch to next task in sequence."""
        self.current_task = (self.current_task + 1) % len(self.task_list)
        self.task_counter += 1
        self._apply_task_params()
        print(f"[Switched to Task {self.current_task + 1}] "
              f"g={self.task_list[self.current_task]['g']}, "
              f"max_torque={self.task_list[self.current_task]['max_torque']}")

    # ---------------------------------------------------------------------
    # Gym interface
    # ---------------------------------------------------------------------
    def reset(self):
        """Reset environment and optionally reset normalization stats."""
        obs, info = self.env.reset()
        self.timestep_counter = 0
        if self.reset_obs_stats:
            self.obs_mean[:] = 0.0
            self.obs_var[:] = 1.0
            self.obs_count = 1e-4

        if self.normalize_obs:
            obs = self._normalize_obs(obs)
        return obs, info

    def step(self, action):
        """Step environment and handle periodic task switching."""
        self.timestep_counter += 1
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Normalize observation
        if self.normalize_obs:
            self._update_obs_statistics(obs)
            obs = self._normalize_obs(obs)

        # Check if we should switch tasks
        if (self.timestep_counter % self.change_freq) == 0:
            self._advance_task()
            truncated = True  # signal episode boundary to PPO

        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------------------------
    # Observation normalization
    # ---------------------------------------------------------------------
    def _update_obs_statistics(self, obs):
        if self.normalize_obs.lower() == "ema":
            self.obs_mean = (1 - self.normalize_avg_coef) * self.obs_mean + self.normalize_avg_coef * obs
            self.obs_var = (1 - self.normalize_avg_coef) * self.obs_var + self.normalize_avg_coef * (obs - self.obs_mean) ** 2
        elif self.normalize_obs.lower() == "straight":
            mean, var, count = self.obs_mean, self.obs_var, self.obs_count
            delta = obs - mean
            tot_count = count + 1
            new_mean = mean + delta / tot_count
            new_var = var * count / tot_count + delta ** 2 * count / tot_count ** 2
            self.obs_mean, self.obs_var, self.obs_count = new_mean, new_var, tot_count

    def _normalize_obs(self, obs):
        if self.timestep_counter == 0:
            return obs
        if self.bias_correction:
            bias = 1 - (1 - self.normalize_avg_coef) ** self.timestep_counter
            mean = self.obs_mean / bias
            var = self.obs_var / bias
        else:
            mean, var = self.obs_mean, self.obs_var
        return (obs - mean) / (np.sqrt(var) + 1e-8)

    # ---------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------
    def evaluate_agent(self, agent, mode="current", num_eval_episodes=10):
        """
        Evaluate the agent under different modes of continual learning analysis.

        Args:
            agent: the PPO agent (must have .act() method)
            mode: "current" (default), "all_tasks", or "zero_shot"
            num_eval_episodes: number of evaluation episodes per task

        Returns:
            results: dictionary containing:
                - 'episodic_returns': list or dict of returns
                - 'mean_return': averaged return
                - 'zero_shot': if applicable, returns for the phase start
                - 'adapted': if applicable, returns for phase end
        """

        results = {}

        # Define tasks to evaluate
        if mode == "current":
            task_indices = [self.current_task]
        elif mode == "all_tasks":
            task_indices = range(len(self.task_list))
        elif mode == "zero_shot":
            task_indices = [self.current_task]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        agent.eval()

        all_task_returns = {}

        for task_idx in task_indices:
            params = self.task_list[task_idx]
            test_env = gym.make("Pendulum-v1")
            test_env = self._wrap_env(test_env)
            test_env.unwrapped.g = params["g"]
            test_env.unwrapped.max_torque = params["max_torque"]

            episodic_returns = []
            for _ in range(num_eval_episodes):
                obs, _ = test_env.reset()
                if self.normalize_obs:
                    obs = self._normalize_obs(obs)
                done = False
                total_r = 0
                while not done:
                    action = agent.act(obs)
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    total_r += reward
                    done = terminated or truncated
                episodic_returns.append(total_r)

            mean_r = np.mean(episodic_returns)
            all_task_returns[f"Task{task_idx+1}"] = {
                "returns": episodic_returns,
                "mean": mean_r,
                "params": params
            }

        agent.train()

        # Aggregate for output
        results["episodic_returns"] = all_task_returns
        results["mean_return"] = np.mean([v["mean"] for v in all_task_returns.values()])

        # Convenience keys
        if mode == "zero_shot":
            results["R_ZS"] = results["mean_return"]
        if mode == "current":
            results["R_avg"] = results["mean_return"]
        if mode == "all_tasks":
            results["R_all"] = {k: v["mean"] for k, v in all_task_returns.items()}

        return results

if __name__ == '__main__':
    ...

    np.set_printoptions(suppress=True)
    env = ContinualPendulumSequence()
    obs, _ = env.reset()

    print(obs)
    for i in range(100):
        env.step(action=[0,0,0,0])
        print(obs)

    env.evaluate_agent(None)


class ContinuousDriftPendulum:
    def __init__(self,
                 drift_type="gravity",     # "gravity", "torque_limit", "target_angle"
                 drift_rate=1e-5,          # how fast to drift (lower = slower)
                 drift_amplitude=0.3,      # how strong the drift is (fractional)
                 normalize_obs="straight",
                 normalize_avg_coef=0.0001,
                 normalize_rewards=True,
                 reset_obs_stats=False,
                 bias_correction=False,
                 seed=None):
        """
        ContinuousDriftPendulum:
        A nonstationary Pendulum-v1 environment where dynamics or goal
        drift smoothly over time, to simulate continuous nonstationarity.

        Drift types:
            "gravity"      - varies gravitational constant g
            "torque_limit" - varies motor torque strength
            "target_angle" - moves the desired equilibrium angle
        """

        self.drift_type = drift_type
        self.drift_rate = drift_rate
        self.drift_amplitude = drift_amplitude
        self.normalize_obs = normalize_obs
        self.normalize_avg_coef = normalize_avg_coef
        self.normalize_rewards = normalize_rewards
        self.reset_obs_stats = reset_obs_stats
        self.bias_correction = bias_correction
        self.seed = seed

        self.timestep_counter = 0
        self.obs_mean = None
        self.obs_var = None
        self.obs_count = 1e-4

        # Base environment
        self.env = gym.make("Pendulum-v1")
        self.env = self._wrap_env(self.env)

        # Set random seed
        if self.seed is not None:
            self.env.reset(seed=self.seed)
            self.env.action_space.seed(self.seed)
            self.env.observation_space.seed(self.seed)

        # Base physics
        self.base_gravity = 9.81
        self.base_torque = 2.0
        self.target_angle = 0.0  # radians (upright)

        # Initialize normalization
        obs_dim = self.env.observation_space.shape
        self.obs_mean = np.zeros(obs_dim, dtype=np.float64)
        self.obs_var = np.ones(obs_dim, dtype=np.float64)

    # ------------------------------------------------------------
    # Wrappers
    # ------------------------------------------------------------
    def _wrap_env(self, env):
        env = DtypeObservation(env, dtype=np.float32)
        env = ClipAction(env)
        if self.normalize_rewards:
            env = gym.wrappers.TransformReward(env, lambda r: r / 500)
        env = TimeLimit(env, max_episode_steps=200)
        env = RecordEpisodeStatistics(env)
        return env

    # ------------------------------------------------------------
    # Drift logic
    # ------------------------------------------------------------
    def _apply_drift(self):
        t = self.timestep_counter
        ω = 2 * math.pi / (4 * self.change_freq)
        env_unwrapped = self.env.unwrapped

        if self.drift_type == "gravity":
            env_unwrapped.g = self.base_gravity + self.amp_gravity * math.sin(ω * t)

        elif self.drift_type == "torque_limit":
            env_unwrapped.max_torque = self.base_torque + self.amp_torque * math.sin(ω * t)

        elif self.drift_type == "target_angle":
            # oscillate target between -π/2 and +π/2 across full 4-task cycle
            self.target_angle = (math.pi / 2) * math.sin(ω * t)

    # ------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------
    def reset(self):
        obs, info = self.env.reset()
        self.timestep_counter = 0
        if self.reset_obs_stats:
            self.obs_mean[:] = 0.0
            self.obs_var[:] = 1.0
            self.obs_count = 1e-4

        if self.normalize_obs:
            obs = self._normalize_obs(obs)
        return obs, info

    def step(self, action):
        self.timestep_counter += 1
        self._apply_drift()

        obs, reward, terminated, truncated, info = self.env.step(action)

        # If target_angle drift is active, recompute reward
        if self.drift_type == "target_angle":
            theta = np.arctan2(obs[1], obs[0])
            theta_dot = obs[2]
            torque_penalty = 0.001 * np.square(action).sum()
            reward = -((theta - self.target_angle) ** 2 + 0.1 * theta_dot ** 2 + torque_penalty)
            if self.normalize_rewards:
                reward /= 500.0

        # Normalize observation
        if self.normalize_obs:
            self._update_obs_statistics(obs)
            obs = self._normalize_obs(obs)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------
    # Observation normalization
    # ------------------------------------------------------------
    def _update_obs_statistics(self, obs):
        if self.normalize_obs.lower() == "ema":
            self.obs_mean = (1 - self.normalize_avg_coef) * self.obs_mean + self.normalize_avg_coef * obs
            self.obs_var = (1 - self.normalize_avg_coef) * self.obs_var + self.normalize_avg_coef * (obs - self.obs_mean) ** 2
        elif self.normalize_obs.lower() == "straight":
            mean, var, count = self.obs_mean, self.obs_var, self.obs_count
            delta = obs - mean
            tot_count = count + 1
            new_mean = mean + delta / tot_count
            new_var = var * count / tot_count + delta ** 2 * count / tot_count ** 2
            self.obs_mean, self.obs_var, self.obs_count = new_mean, new_var, tot_count

    def _normalize_obs(self, obs):
        if self.timestep_counter == 0:
            return obs
        if self.bias_correction:
            bias = 1 - (1 - self.normalize_avg_coef) ** self.timestep_counter
            mean = self.obs_mean / bias
            var = self.obs_var / bias
        else:
            mean, var = self.obs_mean, self.obs_var
        return (obs - mean) / (np.sqrt(var) + 1e-8)

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------
    def evaluate_agent(self, agent, num_eval_episodes=5):
        """Evaluate agent in the current drifted configuration."""
        test_env = gym.make("Pendulum-v1")
        test_env = self._wrap_env(test_env)
        obs, _ = test_env.reset()

        episodic_returns = []
        agent.eval()

        while len(episodic_returns) < num_eval_episodes:
            action = agent.act(obs)
            next_obs, _, terminated, truncated, info = test_env.step(action)
            if "episode" in info:
                episodic_returns.append(info["episode"]["r"])
            obs = next_obs
            if terminated or truncated:
                obs, _ = test_env.reset()

        agent.train()
        return {"episodic_returns": episodic_returns}
