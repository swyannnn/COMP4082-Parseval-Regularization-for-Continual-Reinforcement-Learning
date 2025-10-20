import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.stats

# === helper functions ===
def iqm(data, axis=0):
    """Interquartile mean (robust average)"""
    return scipy.stats.trim_mean(data, 0.25, axis)

def iqm_error_bars(data, axis=0):
    """Interquartile mean confidence intervals"""
    lower, upper = scipy.stats.mstats.trimmed_mean_ci(data, limits=(0.25, 0.25), axis=axis)
    return lower, upper

# === main plotting ===
def load_results(algorithm, env, num_repeats=1, base_folder="results/"):
    """Load training results for a given algorithm and environment."""
    data_list = []
    for i in range(num_repeats):
        file_path = f"{base_folder}data_{env}_{algorithm}_{i}.pkl"
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            data_list.append(data)
    # aggregate over repeats
    all_data = {k: [d[k] for d in data_list if k in d] for k in data_list[0].keys()}
    return all_data

def plot_learning_curves(env="gym_pendulum_discrete", algorithms=["base", "parseval"], save_freq=25000, num_repeats=1, base_folder="results/"):
    metric = "mean_eval_return"  # for pendulum we care about total reward
    plt.figure(figsize=(8, 5))
    SHADED_ALPHA = 0.3

    for alg in algorithms:
        data = load_results(alg, env, num_repeats, base_folder)
        curve_data = np.array(data[metric])  # shape: (num_repeats, num_eval_steps)
        
        # Compute robust mean + CI
        mean_curve = iqm(curve_data, axis=0)
        low_curve, high_curve = iqm_error_bars(curve_data, axis=0)

        xs = save_freq * np.arange(len(mean_curve))
        plots = plt.plot(xs, mean_curve, label=alg, linewidth=2)
        plt.fill_between(xs, low_curve, high_curve, color=plots[0].get_color(), alpha=SHADED_ALPHA)

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Eval Return (higher is better)")
    plt.title(f"Pendulum Performance ({env})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{base_folder}/{env}_learning_curve.png", dpi=300)
    plt.show()

# === run ===
if __name__ == "__main__":
    plot_learning_curves(
        env="gym_pendulum_discrete",
        # algorithms=["base", "parseval"],
        algorithms=["base"],
        save_freq=25000,    # must match your training save frequency
        num_repeats=1,
        base_folder="results_base/"
    )
