"""
Central Limit Theorem (CLT)

One of the most important theorems in statistics!

The Central Limit Theorem states:
When you take many random samples from ANY distribution and calculate their means,
the distribution of those sample means will approximate a normal distribution,
regardless of the original distribution's shape.

Key Points:
1. Works for ANY original distribution (uniform, exponential, even your die rolls!)
2. The larger the sample size, the better the approximation
3. The mean of sample means ≈ population mean (μ)
4. The standard deviation of sample means = σ / √n (called Standard Error)
   where σ is population std dev and n is sample size

Why is this important?
- It's why we can use normal distribution methods for many real-world problems
- It's the foundation of hypothesis testing and confidence intervals
- It explains why averages are more "stable" than individual measurements
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def visualize_clt_with_dice():
    """
    Demonstrate CLT using dice rolls (uniform distribution).
    A single die has uniform distribution (each outcome equally likely),
    but the average of multiple dice approaches normal distribution!
    """
    print("\n" + "="*60)
    print("CENTRAL LIMIT THEOREM: DICE ROLLING EXPERIMENT")
    print("="*60)

    # Simulate rolling n dice, take their average, repeat many times
    sample_sizes = [1, 2, 5, 30]  # Number of dice per roll
    num_experiments = 10000  # How many times to repeat

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, n in enumerate(sample_sizes):
        # For each experiment, roll n dice and take their average
        sample_means = []
        for _ in range(num_experiments):
            dice_rolls = np.random.randint(1, 7, size=n)  # Roll n dice
            sample_means.append(np.mean(dice_rolls))  # Take average

        sample_means = np.array(sample_means)

        # Calculate statistics
        mean_of_means = np.mean(sample_means)
        std_of_means = np.std(sample_means)

        # Theoretical values
        population_mean = 3.5  # Expected value of single die: (1+2+3+4+5+6)/6
        population_std = np.sqrt(35/12)  # Variance of single die
        theoretical_std_error = population_std / np.sqrt(n)

        # Plot histogram
        ax = axes[idx]
        ax.hist(sample_means, bins=50, density=True, alpha=0.7,
                color='skyblue', edgecolor='black', label='Sample means')

        # Overlay theoretical normal distribution
        x = np.linspace(sample_means.min(), sample_means.max(), 100)
        theoretical_pdf = stats.norm(population_mean, theoretical_std_error).pdf(x)
        ax.plot(x, theoretical_pdf, 'r-', linewidth=2, label='Theoretical Normal')

        # Add vertical line at population mean
        ax.axvline(population_mean, color='green', linestyle='--',
                   linewidth=2, label=f'μ={population_mean}')

        ax.set_xlabel('Average of dice rolls')
        ax.set_ylabel('Density')
        ax.set_title(f'n={n} dice per roll')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics box
        stats_text = f'Sample mean: {mean_of_means:.3f}\n'
        stats_text += f'Sample SE: {std_of_means:.3f}\n'
        stats_text += f'Theoretical SE: {theoretical_std_error:.3f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        print(f"\nn = {n} dice:")
        print(f"  Sample mean of means: {mean_of_means:.3f} (theory: {population_mean})")
        print(f"  Sample standard error: {std_of_means:.3f} (theory: {theoretical_std_error:.3f})")

    plt.suptitle('Central Limit Theorem: Distribution of Dice Roll Averages',
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig("clt_dice.png", dpi=150)
    print("\nSaved: clt_dice.png")
    plt.show()

    print("\nNotice how as n increases:")
    print("  - The distribution becomes more bell-shaped (normal)")
    print("  - The spread (standard error) gets narrower")
    print("  - The distribution is always centered around μ=3.5")


def clt_with_different_distributions():
    """
    Show CLT works for various original distributions:
    - Uniform (flat)
    - Exponential (heavily skewed)
    - Bimodal (two peaks)
    """
    sample_size = 30
    num_experiments = 10000

    # Define three very different distributions
    distributions = [
        ("Uniform [0, 10]", lambda: np.random.uniform(0, 10, sample_size)),
        ("Exponential (λ=1)", lambda: np.random.exponential(1, sample_size)),
        ("Bimodal", lambda: np.concatenate([np.random.normal(0, 1, sample_size//2),
                                             np.random.normal(5, 1, sample_size//2)]))
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    for idx, (name, sampler) in enumerate(distributions):
        # Left plot: Show original distribution
        ax_orig = axes[idx, 0]
        original_data = np.concatenate([sampler() for _ in range(100)])
        ax_orig.hist(original_data, bins=50, density=True, alpha=0.7,
                     color='lightcoral', edgecolor='black')
        ax_orig.set_title(f'Original: {name}')
        ax_orig.set_ylabel('Density')
        ax_orig.grid(True, alpha=0.3)

        # Right plot: Distribution of sample means
        ax_means = axes[idx, 1]
        sample_means = [np.mean(sampler()) for _ in range(num_experiments)]
        sample_means = np.array(sample_means)

        ax_means.hist(sample_means, bins=50, density=True, alpha=0.7,
                      color='skyblue', edgecolor='black', label='Sample means')

        # Fit and overlay normal distribution
        mean_fit = np.mean(sample_means)
        std_fit = np.std(sample_means)
        x = np.linspace(sample_means.min(), sample_means.max(), 100)
        ax_means.plot(x, stats.norm(mean_fit, std_fit).pdf(x),
                      'r-', linewidth=2, label='Fitted Normal')

        ax_means.set_title(f'Distribution of Sample Means (n={sample_size})')
        ax_means.legend()
        ax_means.grid(True, alpha=0.3)

        # Statistics
        stats_text = f'μ={mean_fit:.2f}\nσ={std_fit:.2f}'
        ax_means.text(0.98, 0.98, stats_text, transform=ax_means.transAxes,
                      verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('CLT Works for ANY Distribution!', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig("clt_different_distributions.png", dpi=150)
    print("Saved: clt_different_distributions.png")
    plt.show()


def effect_of_sample_size():
    """
    Show how sample size affects the convergence to normal distribution.
    Larger samples → better approximation.
    """
    # Use exponential distribution (very skewed) as source
    source_dist = lambda size: np.random.exponential(2, size)

    sample_sizes = [2, 5, 10, 30, 100]
    num_experiments = 5000

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    # First plot: show original distribution
    ax = axes[0]
    original_data = source_dist(10000)
    ax.hist(original_data, bins=50, density=True, alpha=0.7,
            color='lightcoral', edgecolor='black')
    ax.set_title('Original: Exponential(λ=0.5)')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)

    # Remaining plots: sample means for different n
    for idx, n in enumerate(sample_sizes):
        ax = axes[idx + 1]

        # Generate sample means
        sample_means = [np.mean(source_dist(n)) for _ in range(num_experiments)]
        sample_means = np.array(sample_means)

        # Plot histogram
        ax.hist(sample_means, bins=40, density=True, alpha=0.7,
                color='skyblue', edgecolor='black', label='Sample means')

        # Overlay fitted normal
        mean_fit = np.mean(sample_means)
        std_fit = np.std(sample_means)
        x = np.linspace(sample_means.min(), sample_means.max(), 100)
        ax.plot(x, stats.norm(mean_fit, std_fit).pdf(x),
                'r-', linewidth=2, label='Normal fit')

        ax.set_title(f'Sample size n={n}')
        ax.set_xlabel('Sample mean')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Effect of Sample Size on CLT Convergence', fontsize=14)
    plt.tight_layout()
    plt.savefig("clt_sample_size_effect.png", dpi=150)
    print("Saved: clt_sample_size_effect.png")
    plt.show()


def standard_error_demonstration():
    """
    Demonstrate the Standard Error formula: SE = σ / √n

    Standard Error measures how much sample means vary around the population mean.
    """
    print("\n" + "="*60)
    print("STANDARD ERROR: σ / √n")
    print("="*60)

    population_std = 15  # e.g., IQ scores
    sample_sizes = np.arange(1, 101)

    theoretical_se = population_std / np.sqrt(sample_sizes)

    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, theoretical_se, linewidth=2)
    plt.xlabel('Sample Size (n)')
    plt.ylabel('Standard Error (SE)')
    plt.title('Standard Error Decreases with Sample Size: SE = σ / √n')
    plt.grid(True, alpha=0.3)

    # Add annotations
    for n in [1, 10, 25, 50, 100]:
        se = population_std / np.sqrt(n)
        plt.plot(n, se, 'ro', markersize=8)
        plt.annotate(f'n={n}\nSE={se:.2f}',
                     xy=(n, se), xytext=(n+5, se+0.5),
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                     fontsize=8)

    plt.tight_layout()
    plt.savefig("standard_error.png", dpi=150)
    print("Saved: standard_error.png")
    plt.show()

    print("\nKey insight:")
    print(f"  To halve the standard error, you need 4× the sample size!")
    print(f"  Example: SE(n=25) = {population_std/np.sqrt(25):.2f}")
    print(f"           SE(n=100) = {population_std/np.sqrt(100):.2f}")


def real_world_example():
    """
    Real-world application: Estimating average height from samples
    """
    print("\n" + "="*60)
    print("REAL-WORLD EXAMPLE: Estimating Average Height")
    print("="*60)

    # True population: adult heights ~ N(170 cm, 10 cm)
    true_mean = 170
    true_std = 10

    print(f"\nTrue population: μ={true_mean} cm, σ={true_std} cm")
    print("(In practice, we don't know these values!)")

    sample_size = 50
    num_samples = 1000

    # Take many samples and calculate their means
    sample_means = []
    for _ in range(num_samples):
        sample = np.random.normal(true_mean, true_std, sample_size)
        sample_means.append(np.mean(sample))

    sample_means = np.array(sample_means)

    # According to CLT, sample_means should be ~ N(μ, σ/√n)
    predicted_mean = true_mean
    predicted_se = true_std / np.sqrt(sample_size)

    plt.figure(figsize=(12, 6))

    # Plot distribution of sample means
    plt.hist(sample_means, bins=40, density=True, alpha=0.7,
             color='skyblue', edgecolor='black', label=f'Sample means (n={sample_size})')

    # Overlay CLT prediction
    x = np.linspace(sample_means.min(), sample_means.max(), 100)
    plt.plot(x, stats.norm(predicted_mean, predicted_se).pdf(x),
             'r-', linewidth=3, label='CLT Prediction')

    plt.axvline(true_mean, color='green', linestyle='--',
                linewidth=2, label=f'True mean = {true_mean}')

    plt.xlabel('Sample Mean Height (cm)')
    plt.ylabel('Density')
    plt.title(f'Distribution of Sample Means\n(Each sample has {sample_size} people)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add statistics box
    actual_mean = np.mean(sample_means)
    actual_se = np.std(sample_means)
    stats_text = f'Predicted: μ={predicted_mean}, SE={predicted_se:.3f}\n'
    stats_text += f'Observed: μ={actual_mean:.2f}, SE={actual_se:.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig("clt_real_world.png", dpi=150)
    print("Saved: clt_real_world.png")
    plt.show()

    print(f"\nCLT predicts: sample means ~ N({predicted_mean}, {predicted_se:.3f})")
    print(f"We observed: μ={actual_mean:.2f}, SE={actual_se:.3f}")
    print("\n95% of sample means should fall within μ ± 2×SE:")
    lower = predicted_mean - 2*predicted_se
    upper = predicted_mean + 2*predicted_se
    print(f"  [{lower:.2f}, {upper:.2f}] cm")

    actual_within = np.sum((sample_means >= lower) & (sample_means <= upper)) / len(sample_means)
    print(f"  Actual percentage within range: {actual_within*100:.1f}%")


if __name__ == "__main__":
    print("CENTRAL LIMIT THEOREM DEMONSTRATIONS")
    print("=" * 60)

    print("\n1. CLT with dice rolling...")
    visualize_clt_with_dice()

    print("\n2. CLT with different source distributions...")
    clt_with_different_distributions()

    print("\n3. Effect of sample size on CLT...")
    effect_of_sample_size()

    print("\n4. Standard Error demonstration...")
    standard_error_demonstration()

    print("\n5. Real-world example...")
    real_world_example()

    print("\n" + "="*60)
    print("KEY TAKEAWAYS:")
    print("="*60)
    print("1. CLT works for ANY original distribution")
    print("2. Larger samples → better normal approximation")
    print("3. Sample means cluster around population mean")
    print("4. Standard Error = σ / √n gets smaller with larger n")
    print("5. This is why we can use normal-based methods in statistics!")
    print("="*60)
