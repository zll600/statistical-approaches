"""
Real-World Applications of Uniform Distribution

The uniform distribution appears in many practical scenarios:
1. Random number generation (foundation of all simulation)
2. Monte Carlo simulations
3. Waiting times when arrival time is unknown
4. Randomization in experiments
5. Initial conditions in optimization
6. Password/ID generation
7. Fair selection processes

Key insight: Uniform distribution represents "complete uncertainty"
or "maximum entropy" - when we know only the range but nothing about
which values are more or less likely.
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def monte_carlo_estimate_pi():
    """
    Use uniform random numbers to estimate pi using Monte Carlo method.

    Method: Generate random points in a square, count how many fall
    inside a quarter circle. The ratio approximates pi/4.
    """
    print("\n" + "=" * 60)
    print("MONTE CARLO SIMULATION: Estimating Pi")
    print("=" * 60)

    sample_sizes = [100, 1000, 10000, 100000]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    estimates = []

    for idx, n in enumerate(sample_sizes):
        ax = axes[idx // 2, idx % 2]

        # Generate uniform random points in [0, 1] x [0, 1]
        x = np.random.uniform(0, 1, n)
        y = np.random.uniform(0, 1, n)

        # Check if inside quarter circle
        distances = np.sqrt(x**2 + y**2)
        inside = distances <= 1

        # Estimate pi
        pi_estimate = 4 * np.sum(inside) / n
        estimates.append(pi_estimate)

        # Plot (subsample for large n)
        plot_n = min(n, 5000)
        plot_indices = np.random.choice(n, plot_n, replace=False)

        ax.scatter(
            x[plot_indices][inside[plot_indices]],
            y[plot_indices][inside[plot_indices]],
            c="red",
            s=1,
            alpha=0.5,
            label="Inside circle",
        )
        ax.scatter(
            x[plot_indices][~inside[plot_indices]],
            y[plot_indices][~inside[plot_indices]],
            c="blue",
            s=1,
            alpha=0.5,
            label="Outside circle",
        )

        # Draw quarter circle
        theta = np.linspace(0, np.pi / 2, 100)
        ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=2)

        error = abs(pi_estimate - np.pi)
        error_pct = (error / np.pi) * 100

        stats_text = f"n = {n:,}\n"
        stats_text += f"Pi estimate: {pi_estimate:.6f}\n"
        stats_text += f"True pi: {np.pi:.6f}\n"
        stats_text += f"Error: {error:.6f} ({error_pct:.2f}%)"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=9,
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"n = {n:,} samples")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Monte Carlo Estimation of Pi Using Uniform Random Numbers",
        fontsize=14,
        weight="bold",
    )
    plt.tight_layout()
    plt.savefig("monte_carlo_pi.png", dpi=150)
    print("Saved: monte_carlo_pi.png")
    plt.show()

    print("\nResults:")
    for n, est in zip(sample_sizes, estimates):
        error = abs(est - np.pi)
        print(f"  n={n:6d}: pi ~ {est:.6f}, error = {error:.6f}")

    print(f"\nTrue pi: {np.pi:.10f}")
    print("\nKey insight: More samples -> better estimate (Law of Large Numbers)")


def monte_carlo_integration():
    """
    Use uniform random numbers to estimate integrals.

    Example: Estimate integral of f(x) = x^2 from 0 to 1
    True answer: 1/3 = 0.333...
    """
    print("\n" + "=" * 60)
    print("MONTE CARLO INTEGRATION")
    print("=" * 60)

    def f(x):
        """Function to integrate: f(x) = x^2"""
        return x**2

    a, b = 0, 1  # Integration bounds
    true_value = 1 / 3  # Analytical solution

    sample_sizes = [100, 500, 2000, 10000]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, n in enumerate(sample_sizes):
        ax = axes[idx // 2, idx % 2]

        # Generate uniform random points
        x_random = np.random.uniform(a, b, n)
        y_random = f(x_random)

        # Monte Carlo estimate
        estimate = (b - a) * np.mean(y_random)

        # For visualization
        x_plot = np.linspace(a, b, 100)
        y_plot = f(x_plot)

        # Plot function
        ax.plot(x_plot, y_plot, "b-", linewidth=2, label="f(x) = x^2")
        ax.fill_between(x_plot, y_plot, alpha=0.2, color="blue")

        # Plot sample points (subset for clarity)
        plot_n = min(n, 500)
        plot_indices = np.random.choice(n, plot_n, replace=False)
        ax.scatter(
            x_random[plot_indices],
            y_random[plot_indices],
            c="red",
            s=10,
            alpha=0.3,
            label="Sample points",
        )

        error = abs(estimate - true_value)
        error_pct = (error / true_value) * 100

        stats_text = f"n = {n:,}\n"
        stats_text += f"Estimate: {estimate:.6f}\n"
        stats_text += f"True value: {true_value:.6f}\n"
        stats_text += f"Error: {error:.6f} ({error_pct:.2f}%)"
        ax.text(
            0.98,
            0.02,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=9,
        )

        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(f"n = {n:,} samples")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Monte Carlo Integration Using Uniform Random Sampling",
        fontsize=14,
        weight="bold",
    )
    plt.tight_layout()
    plt.savefig("monte_carlo_integration.png", dpi=150)
    print("Saved: monte_carlo_integration.png")
    plt.show()

    print(f"\nIntegrating f(x) = x^2 from {a} to {b}")
    print(f"True answer: {true_value:.10f}\n")

    for n in sample_sizes:
        x_random = np.random.uniform(a, b, n)
        estimate = (b - a) * np.mean(f(x_random))
        error = abs(estimate - true_value)
        print(f"  n={n:5d}: estimate = {estimate:.6f}, error = {error:.6f}")


def waiting_time_problem():
    """
    Classic waiting time problem: Bus arrives uniformly in next 20 minutes.

    Demonstrates practical use of uniform distribution for uncertainty modeling.
    """
    print("\n" + "=" * 60)
    print("WAITING TIME PROBLEM")
    print("=" * 60)

    print("\nScenario: You arrive at a bus stop. The bus schedule says")
    print("buses come every 20 minutes, but you don't know when the last one left.")
    print("Your waiting time is uniform: U(0, 20) minutes")

    a, b = 0, 20
    dist = stats.uniform(loc=a, scale=b - a)

    # Simulate many arrivals
    n_simulations = 10000
    waiting_times = np.random.uniform(a, b, n_simulations)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Distribution of waiting times
    ax = axes[0, 0]
    ax.hist(
        waiting_times,
        bins=50,
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        label="Simulated",
    )
    x = np.linspace(a, b, 100)
    ax.plot(x, dist.pdf(x), "r-", linewidth=2, label="Theoretical")
    ax.set_xlabel("Waiting Time (minutes)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Waiting Times")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: CDF
    ax = axes[0, 1]
    sorted_times = np.sort(waiting_times)
    ax.plot(
        sorted_times,
        np.linspace(0, 1, len(sorted_times)),
        "b-",
        linewidth=2,
        label="Empirical CDF",
    )
    x = np.linspace(a, b, 100)
    ax.plot(x, dist.cdf(x), "r--", linewidth=2, label="Theoretical CDF")
    ax.set_xlabel("Waiting Time (minutes)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Cumulative Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Probability of waiting less than various times
    ax = axes[1, 0]
    wait_thresholds = [5, 10, 15]
    colors = ["green", "orange", "red"]
    for threshold, color in zip(wait_thresholds, colors):
        prob = dist.cdf(threshold)
        x_fill = np.linspace(a, threshold, 100)
        ax.fill_between(
            x_fill,
            dist.pdf(x_fill),
            alpha=0.3,
            color=color,
            label=f"P(wait < {threshold}) = {prob:.2f}",
        )
    x = np.linspace(a, b, 100)
    ax.plot(x, dist.pdf(x), "k-", linewidth=2)
    ax.set_xlabel("Waiting Time (minutes)")
    ax.set_ylabel("Density")
    ax.set_title("Probability of Waiting Less Than...")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Statistics summary
    ax = axes[1, 1]
    ax.axis("off")

    mean_wait = (a + b) / 2
    std_wait = (b - a) / np.sqrt(12)
    median_wait = (a + b) / 2

    # Calculate various probabilities
    prob_less_5 = dist.cdf(5)
    prob_more_15 = 1 - dist.cdf(15)
    prob_between_5_15 = dist.cdf(15) - dist.cdf(5)

    stats_text = "WAITING TIME STATISTICS\n"
    stats_text += "=" * 40 + "\n\n"
    stats_text += f"Average wait: {mean_wait:.1f} minutes\n"
    stats_text += f"Median wait: {median_wait:.1f} minutes\n"
    stats_text += f"Std deviation: {std_wait:.2f} minutes\n\n"
    stats_text += "PROBABILITIES\n"
    stats_text += "-" * 40 + "\n"
    stats_text += f"P(wait < 5 min):  {prob_less_5:.1%}\n"
    stats_text += f"P(wait > 15 min): {prob_more_15:.1%}\n"
    stats_text += f"P(5 < wait < 15): {prob_between_5_15:.1%}\n\n"
    stats_text += "PRACTICAL INSIGHTS\n"
    stats_text += "-" * 40 + "\n"
    stats_text += "- On average, wait half the interval\n"
    stats_text += "- Every minute has equal probability\n"
    stats_text += "- 50% chance of waiting < 10 min\n"

    ax.text(
        0.1,
        0.9,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=10,
    )

    plt.suptitle(
        "Waiting Time Analysis: Uniform Distribution U(0, 20)",
        fontsize=14,
        weight="bold",
    )
    plt.tight_layout()
    plt.savefig("waiting_time.png", dpi=150)
    print("Saved: waiting_time.png")
    plt.show()

    print(f"\nKey statistics:")
    print(f"  Average waiting time: {mean_wait:.1f} minutes")
    print(f"  Standard deviation: {std_wait:.2f} minutes")
    print(f"  P(wait < 5 min) = {prob_less_5:.2%}")
    print(f"  P(wait > 15 min) = {prob_more_15:.2%}")


def random_number_generation_demo():
    """
    Demonstrate how uniform random numbers are the foundation
    of all pseudo-random number generation.
    """
    print("\n" + "=" * 60)
    print("RANDOM NUMBER GENERATION FOUNDATION")
    print("=" * 60)

    print("\nUniform distribution is the foundation of ALL random number generation!")
    print("Computer RNGs generate U(0,1), then transform to other distributions.\n")

    # Generate sequences
    n = 1000
    uniform_samples = np.random.uniform(0, 1, n)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Plot 1: Uniform U(0,1) - the source
    ax = axes[0, 0]
    ax.hist(
        uniform_samples,
        bins=50,
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    ax.axhline(1, color="red", linestyle="--", linewidth=2, label="Theory: 1")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Source: U(0, 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Scaled to U(5, 15)
    ax = axes[0, 1]
    scaled = 5 + (15 - 5) * uniform_samples  # Transform to U(5, 15)
    ax.hist(
        scaled, bins=50, density=True, alpha=0.7, color="lightgreen", edgecolor="black"
    )
    ax.axhline(1 / 10, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Scaled: U(5, 15)")
    ax.text(
        0.5,
        0.95,
        "x_new = a + (b-a)*U(0,1)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
    )
    ax.grid(True, alpha=0.3)

    # Plot 3: Discrete uniform (like dice)
    ax = axes[0, 2]
    dice = np.floor(1 + 6 * uniform_samples[:1000])  # 1 to 6
    values, counts = np.unique(dice, return_counts=True)
    ax.bar(values, counts / len(dice), alpha=0.7, color="salmon", edgecolor="black")
    ax.axhline(1 / 6, color="red", linestyle="--", linewidth=2, label="Theory: 1/6")
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability")
    ax.set_title("Discrete Uniform (Dice)")
    ax.set_xticks(values)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 4: Sequence plot showing randomness
    ax = axes[1, 0]
    ax.plot(uniform_samples[:100], "b-", linewidth=1, alpha=0.7)
    ax.scatter(range(100), uniform_samples[:100], s=20, c="blue", alpha=0.5)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.set_title("Sequence of Random Numbers")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Plot 5: Scatter plot (2D randomness)
    ax = axes[1, 1]
    ax.scatter(uniform_samples[::2], uniform_samples[1::2], s=10, alpha=0.5, c="purple")
    ax.set_xlabel("U_i")
    ax.set_ylabel("U_{i+1}")
    ax.set_title("Pairs of Random Numbers\n(Tests for independence)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Plot 6: Statistics
    ax = axes[1, 2]
    ax.axis("off")

    mean_val = np.mean(uniform_samples)
    std_val = np.std(uniform_samples)
    min_val = np.min(uniform_samples)
    max_val = np.max(uniform_samples)

    stats_text = "U(0,1) PROPERTIES\n"
    stats_text += "=" * 30 + "\n\n"
    stats_text += f"Sample mean: {mean_val:.4f}\n"
    stats_text += f"Theory mean: 0.5000\n\n"
    stats_text += f"Sample std: {std_val:.4f}\n"
    stats_text += f"Theory std: {1 / np.sqrt(12):.4f}\n\n"
    stats_text += f"Min value: {min_val:.4f}\n"
    stats_text += f"Max value: {max_val:.4f}\n\n"
    stats_text += "KEY INSIGHT\n"
    stats_text += "-" * 30 + "\n"
    stats_text += "All computer RNGs\n"
    stats_text += "start with U(0,1),\n"
    stats_text += "then transform to\n"
    stats_text += "other distributions"

    ax.text(
        0.1,
        0.9,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=9,
    )

    plt.suptitle(
        "Uniform Distribution: Foundation of Random Number Generation",
        fontsize=14,
        weight="bold",
    )
    plt.tight_layout()
    plt.savefig("rng_foundation.png", dpi=150)
    print("Saved: rng_foundation.png")
    plt.show()


def simulation_example():
    """
    Simulate a real-world process using uniform random numbers.

    Example: Simulating customer arrivals at a store
    """
    print("\n" + "=" * 60)
    print("SIMULATION: Customer Service Times")
    print("=" * 60)

    print("\nScenario: Service times are uniformly distributed")
    print("between 2 and 8 minutes: U(2, 8)")

    # Parameters
    n_customers = 1000
    min_service = 2
    max_service = 8

    # Generate service times
    service_times = np.random.uniform(min_service, max_service, n_customers)

    # Calculate cumulative time
    cumulative_times = np.cumsum(service_times)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Distribution of service times
    ax = axes[0, 0]
    ax.hist(
        service_times,
        bins=50,
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        label="Simulated",
    )
    x = np.linspace(min_service, max_service, 100)
    ax.axhline(
        1 / (max_service - min_service),
        color="red",
        linestyle="--",
        linewidth=2,
        label="Theoretical",
    )
    ax.set_xlabel("Service Time (minutes)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Service Times")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Service times over sequence
    ax = axes[0, 1]
    ax.plot(service_times[:100], "b-", linewidth=1, alpha=0.7)
    ax.axhline(
        (min_service + max_service) / 2,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Mean",
    )
    ax.set_xlabel("Customer Number")
    ax.set_ylabel("Service Time (minutes)")
    ax.set_title("First 100 Service Times")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Cumulative time
    ax = axes[1, 0]
    ax.plot(cumulative_times, "g-", linewidth=1)
    ax.set_xlabel("Customer Number")
    ax.set_ylabel("Cumulative Time (minutes)")
    ax.set_title("Total Time to Serve Customers")
    ax.grid(True, alpha=0.3)

    # Plot 4: Statistics
    ax = axes[1, 1]
    ax.axis("off")

    mean_service = np.mean(service_times)
    std_service = np.std(service_times)
    total_time = cumulative_times[-1]
    avg_per_customer = total_time / n_customers

    theoretical_mean = (min_service + max_service) / 2
    theoretical_std = (max_service - min_service) / np.sqrt(12)

    stats_text = "SIMULATION RESULTS\n"
    stats_text += "=" * 35 + "\n\n"
    stats_text += f"Customers served: {n_customers}\n"
    stats_text += f"Total time: {total_time:.1f} min\n"
    stats_text += f"         = {total_time / 60:.1f} hours\n\n"
    stats_text += "SERVICE TIME\n"
    stats_text += "-" * 35 + "\n"
    stats_text += f"Observed mean: {mean_service:.2f} min\n"
    stats_text += f"Theory mean:   {theoretical_mean:.2f} min\n\n"
    stats_text += f"Observed std:  {std_service:.2f} min\n"
    stats_text += f"Theory std:    {theoretical_std:.2f} min\n\n"
    stats_text += "INSIGHTS\n"
    stats_text += "-" * 35 + "\n"
    stats_text += f"Fastest service: {np.min(service_times):.2f} min\n"
    stats_text += f"Slowest service: {np.max(service_times):.2f} min\n"
    stats_text += f"% under 4 min:   {np.mean(service_times < 4) * 100:.1f}%\n"
    stats_text += f"% over 6 min:    {np.mean(service_times > 6) * 100:.1f}%"

    ax.text(
        0.1,
        0.9,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=9,
    )

    plt.suptitle(
        f"Customer Service Simulation: U({min_service}, {max_service}) minutes",
        fontsize=14,
        weight="bold",
    )
    plt.tight_layout()
    plt.savefig("service_simulation.png", dpi=150)
    print("Saved: service_simulation.png")
    plt.show()

    print(f"\nSimulation completed for {n_customers} customers")
    print(f"Total service time: {total_time:.1f} minutes ({total_time / 60:.1f} hours)")
    print(f"Average per customer: {avg_per_customer:.2f} minutes")


if __name__ == "__main__":
    print("UNIFORM DISTRIBUTION: REAL-WORLD APPLICATIONS")
    print("=" * 60)

    print("\n1. Monte Carlo estimation of Pi...")
    monte_carlo_estimate_pi()

    print("\n2. Monte Carlo integration...")
    monte_carlo_integration()

    print("\n3. Waiting time problem...")
    waiting_time_problem()

    print("\n4. Random number generation foundation...")
    random_number_generation_demo()

    print("\n5. Customer service simulation...")
    simulation_example()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAY:")
    print("Uniform distribution is fundamental to:")
    print("  - All random number generation")
    print("  - Monte Carlo simulations")
    print("  - Modeling complete uncertainty")
    print("  - Fair/unbiased selection")
    print("=" * 60)
