import torch
import gym
import numpy as np
import matplotlib.pyplot as plt

class ContinuousDriftWrapper(gym.Wrapper):
    def __init__(self, env, param_name="gravity", drift_rate=1e-5, drift_range=(5.0, 15.0), oscillate=True):
        super().__init__(env)
        self.param_name = param_name
        self.drift_rate = drift_rate
        self.drift_range = drift_range
        self.oscillate = oscillate
        self.t = 0
        self.param_val = getattr(env, param_name, 9.8)
        print(f"[ContinuousDriftWrapper] param={param_name}, drift_rate={drift_rate}")

    def step(self, action):
        # Ensure the action is the correct shape
        if np.isscalar(action):
            action = np.array([action], dtype=np.float32)
        elif isinstance(action, np.ndarray) and action.shape == ():
            action = np.expand_dims(action, axis=0).astype(np.float32)

        obs, reward, done, truncated, info = self.env.step(action)

        # ✅ Cast everything to float32 to avoid MPS dtype errors
        obs = obs.astype(np.float32)
        reward = np.float32(reward)

        self._apply_drift()
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        self.t = 0
        return self.env.reset(**kwargs)

    def render(self):
        return self.env.render()

    def _apply_drift(self):
        self.t += 1
        if self.oscillate:
            # # sinusoidal drift between bounds
            # val = (self.drift_range[1] - self.drift_range[0]) / 2 * np.sin(self.drift_rate * self.t) \
            #       + np.mean(self.drift_range)
            val = 9.81
        else:
            # val = self.param_val + self.drift_rate
            val = 9.81
        setattr(self.env, self.param_name, float(val))

    def evaluate_agent(self, agent, num_eval_runs=5, render=True):
        """Simple evaluation loop for Gym environments (handles scalars, tensors, etc.)."""
        episodic_returns = []
        for _ in range(num_eval_runs):
            obs, info = self.reset()
            done = False
            total_reward = 0
            steps = 0
            while not done and steps < 1000:
                # Get action from agent (no gradients)
                action = agent.act(obs)

                # ✅ Handle all possible action formats
                if torch.is_tensor(action):
                    action = action.detach().cpu().numpy()
                if np.isscalar(action):
                    action = np.array([action], dtype=np.float32)
                elif isinstance(action, np.ndarray) and action.shape == ():
                    action = np.expand_dims(action, axis=0).astype(np.float32)

                obs, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1

                if render:
                    frame = self.env.render()
                    plt.imshow(frame)
                    plt.axis("off")
                    plt.pause(0.001)
                    plt.clf()

            episodic_returns.append(total_reward)

        # ✅ Return structure compatible with Parseval repo
        return {
            "episodic_returns": episodic_returns,
            "successes": np.zeros(num_eval_runs),  # for consistency
        }

class DiscreteDriftWrapper(gym.Wrapper):
    def __init__(self, env, param_name="gravity", task_values=(5.0, 10.0, 15.0), change_freq=50000):
        super().__init__(env)
        self.param_name = param_name
        self.task_values = task_values
        self.change_freq = change_freq
        self.current_task_idx = 0
        self.t = 0
        setattr(self.env, self.param_name, self.task_values[self.current_task_idx])
        print(f"[DiscreteDriftWrapper] {param_name}={self.task_values[0]} (freq={change_freq})")

    def step(self, action):
        # ✅ Handle all possible action formats
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        if np.isscalar(action):
            action = np.array([action], dtype=np.float32)
        elif isinstance(action, np.ndarray) and action.shape == ():
            action = np.expand_dims(action, axis=0).astype(np.float32)
        obs, reward, done, truncated, info = self.env.step(action)
        self.t += 1

        # switch task every change_freq steps
        if self.t % self.change_freq == 0:
            self.current_task_idx = (self.current_task_idx + 1) % len(self.task_values)
            new_val = self.task_values[self.current_task_idx]
            setattr(self.env, self.param_name, new_val)
            print(f"[DiscreteDriftWrapper] Switched to gravity={new_val}")

        return obs.astype(np.float32), np.float32(reward), done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self):
        return self.env.render()

    def evaluate_agent(self, agent, num_eval_runs=5, render=True):
        """Evaluate the trained agent across all discrete gravity settings."""
        episodic_returns = []
        gravity_values = []

        for gravity in self.task_values:
            setattr(self.env, self.param_name, gravity)
            print(f"[Eval] gravity={gravity}")
            for _ in range(num_eval_runs):
                obs, info = self.reset()
                done = False
                total_reward = 0
                steps = 0
                while not done and steps < 1000:
                    # Get action from agent
                    action = agent.act(obs)
                    if torch.is_tensor(action):
                        action = action.detach().cpu().numpy()
                    if np.isscalar(action):
                        action = np.array([action], dtype=np.float32)
                    elif isinstance(action, np.ndarray) and action.shape == ():
                        action = np.expand_dims(action, axis=0).astype(np.float32)

                    obs, reward, done, truncated, info = self.env.step(action)
                    total_reward += reward
                    steps += 1

                    if render:
                        frame = self.env.render()
                        plt.imshow(frame)
                        plt.axis("off")
                        plt.pause(0.001)
                        plt.clf()

                episodic_returns.append(total_reward)
                gravity_values.append(gravity)

        # Return structure compatible with main.py expectations
        return {
            "episodic_returns": episodic_returns,
            "successes": np.zeros(len(episodic_returns)),
            "gravity_values": gravity_values
        }