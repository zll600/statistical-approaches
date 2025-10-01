"""
Uniform Distribution Basics

The uniform distribution is one of the simplest probability distributions.
All values in a given range are equally likely to occur - that's what makes it "uniform"!

Types:
1. Continuous Uniform: U(a, b) - any value in range [a, b] equally likely
2. Discrete Uniform: finite number of equally likely outcomes (e.g., dice roll)

Key Properties (Continuous U(a, b)):
- PDF (Probability Density Function): f(x) = 1/(b-a) for a <= x <= b, 0 otherwise
- CDF (Cumulative Distribution Function): F(x) = (x-a)/(b-a)
- Mean: (a + b) / 2 (midpoint of the range)
- Variance: (b - a)^2 / 12
- Standard Deviation: (b - a) / sqrt(12)

Key Difference from Normal:
- Uniform: flat, constant probability across range (rectangular shape)
- Normal: bell-shaped, highest probability at mean

Why Uniform Matters:
- Foundation of random number generation
- Used to generate other distributions
- Models complete uncertainty (all values equally likely)
- Common in simulations and Monte Carlo methods
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Tuple

def visualize_continuous_uniform():
    """
    Visualize continuous uniform distributions with different parameters.

    The PDF is a flat rectangle - hence uniform (same height everywhere).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Different uniform distributions
    distributions = [
        (0, 1, "U(0, 1) - Standard Uniform"),
        (0, 10, "U(0, 10) - Wider range"),
        (-5, 5, "U(-5, 5) - Symmetric around 0"),
        (10, 15, "U(10, 15) - Shifted range"),
    ]

    for idx, (a, b, label) in enumerate(distributions):
        ax: Axes = axes[idx // 2, idx % 2]

        # Create uniform distribution
        dist = stats.uniform(loc=a, scale=b - a)

        # Generate x values (extend beyond range to show zero probability)
        x = np.linspace(a - 2, b + 2, 1000)
        pdf = dist.pdf(x)

        # Plot PDF
        ax.plot(x, pdf, "b-", linewidth=2, label="PDF")
        ax.fill_between(x, pdf, alpha=0.3, color="blue")

        # Mark boundaries
        ax.axvline(a, color="red", linestyle="--", linewidth=2, label=f"a={a}")
        ax.axvline(b, color="red", linestyle="--", linewidth=2, label=f"b={b}")

        # Mark mean
        mean = (a + b) / 2
        ax.axvline(
            mean, color="green", linestyle=":", linewidth=2, label=f"mean={mean}"
        )

        # Add statistics
        variance = (b - a) ** 2 / 12
        std = np.sqrt(variance)
        height = 1 / (b - a)

        stats_text = f"Mean: {mean}\n"
        stats_text += f"Std Dev: {std:.3f}\n"
        stats_text += f"Height: {height:.3f}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        ax.set_xlabel("x")
        ax.set_ylabel("Probability Density")
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.suptitle("Continuous Uniform Distribution: PDF", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig("uniform_pdf_continuous.png", dpi=150)
    print("Saved: uniform_pdf_continuous.png")
    plt.show()

    print("\nKey observation: PDF is flat (constant) within [a, b], zero outside")
    print("The height 1/(b-a) ensures the total area under curve equals 1")


def visualize_cdf():
    """
    Visualize CDF for uniform distribution.

    The CDF is a straight line from 0 to 1 within the range.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Example: U(2, 8)
    a, b = 2, 8
    dist = stats.uniform(loc=a, scale=b - a)

    x = np.linspace(a - 2, b + 2, 1000)
    pdf = dist.pdf(x)
    cdf = dist.cdf(x)

    # Plot PDF
    ax: Axes = axes[0]
    ax.plot(x, pdf, "b-", linewidth=2)
    ax.fill_between(x, pdf, alpha=0.3, color="blue")
    ax.axvline(a, color="red", linestyle="--", linewidth=2)
    ax.axvline(b, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"PDF: U({a}, {b})")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Plot CDF
    ax = axes[1]
    ax.plot(x, cdf, "g-", linewidth=2)
    ax.axvline(a, color="red", linestyle="--", linewidth=2, label=f"a={a}")
    ax.axvline(b, color="red", linestyle="--", linewidth=2, label=f"b={b}")
    ax.axhline(0.5, color="orange", linestyle=":", alpha=0.7, label="50%")
    ax.set_xlabel("x")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(f"CDF: U({a}, {b})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.suptitle("Uniform Distribution: PDF and CDF", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig("uniform_cdf.png", dpi=150)
    print("Saved: uniform_cdf.png")
    plt.show()

    print("\nKey observation: CDF is a straight line from 0 to 1")
    print("This linear relationship means probability increases uniformly")


def discrete_uniform_examples():
    """
    Examples of discrete uniform distributions.

    Classic examples: dice rolls, coin flips (if we assign numbers).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    examples = [
        ("Single Die (1-6)", 1, 6),
        ("Two Dice Sum (2-12)", 2, 12),
        ("Single Digit (0-9)", 0, 9),
        ("Playing Card Rank (1-13)", 1, 13),
    ]

    for idx, (name, low, high) in enumerate(examples):
        ax: Axes = axes[idx // 2, idx % 2]

        # Create discrete uniform
        # Note: high parameter in stats.randint is exclusive
        dist = stats.randint(low=low, high=high + 1)

        # Get all possible values
        x = np.arange(low, high + 1)
        pmf = dist.pmf(x)

        # Bar plot for PMF
        ax.bar(x, pmf, alpha=0.7, color="skyblue", edgecolor="black", label="PMF")

        # Mark mean
        mean = (low + high) / 2
        ax.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"Mean={mean}")

        # Add value labels on bars
        for xi, pi in zip(x, pmf):
            ax.text(xi, pi + 0.01, f"{pi:.3f}", ha="center", va="bottom", fontsize=8)

        # Calculate statistics
        variance = ((high - low + 1) ** 2 - 1) / 12
        std = np.sqrt(variance)

        stats_text = f"Range: [{low}, {high}]\n"
        stats_text += f"n values: {high - low + 1}\n"
        stats_text += f"P(each): {1 / (high - low + 1):.4f}\n"
        stats_text += f"Mean: {mean:.1f}\n"
        stats_text += f"Std Dev: {std:.3f}"
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=9,
        )

        ax.set_xlabel("Value")
        ax.set_ylabel("Probability")
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(x)

    plt.suptitle("Discrete Uniform Distribution Examples", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig("uniform_discrete.png", dpi=150)
    print("Saved: uniform_discrete.png")
    plt.show()


def probability_calculations():
    """
    Practical probability calculations with uniform distribution.
    """
    print("\n" + "=" * 60)
    print("PROBABILITY CALCULATIONS WITH UNIFORM DISTRIBUTION")
    print("=" * 60)

    # Example 1: Bus arrival time U(0, 20) minutes
    print("\nExample 1: Bus Arrival Time")
    print("-" * 60)
    print("You arrive at a bus stop. The bus arrives uniformly")
    print("between 0 and 20 minutes from now: U(0, 20)")

    a, b = 0, 20
    dist = stats.uniform(loc=a, scale=b - a)

    # Q1: Probability of waiting less than 5 minutes
    prob1 = dist.cdf(5) - dist.cdf(0)
    print(f"\nQ1: P(wait < 5 minutes) = {prob1:.4f} = {prob1 * 100:.1f}%")
    print(f"    Calculation: (5-0)/(20-0) = {prob1:.4f}")

    # Q2: Probability of waiting between 10 and 15 minutes
    prob2 = dist.cdf(15) - dist.cdf(10)
    print(f"\nQ2: P(10 < wait < 15) = {prob2:.4f} = {prob2 * 100:.1f}%")
    print(f"    Calculation: (15-10)/(20-0) = {prob2:.4f}")

    # Q3: Expected waiting time
    mean_wait = (a + b) / 2
    print(f"\nQ3: Expected waiting time = {mean_wait:.1f} minutes")
    print(f"    Formula: (a+b)/2 = ({a}+{b})/2 = {mean_wait}")

    # Example 2: Random number between -1 and 1
    print("\n\nExample 2: Random Error Term U(-1, 1)")
    print("-" * 60)
    a, b = -1, 1
    dist = stats.uniform(loc=a, scale=b - a)

    # Q1: Probability of positive error
    prob_pos = 1 - dist.cdf(0)
    print(f"\nQ1: P(error > 0) = {prob_pos:.4f} = {prob_pos * 100:.0f}%")
    print("    (Makes sense: symmetric around 0)")

    # Q2: Probability of error within 0.5 of center
    prob_small = dist.cdf(0.5) - dist.cdf(-0.5)
    print(f"\nQ2: P(-0.5 < error < 0.5) = {prob_small:.4f} = {prob_small * 100:.0f}%")
    print(f"    Calculation: (0.5-(-0.5))/(1-(-1)) = 1.0/2.0 = {prob_small}")


def compare_widths():
    """
    Show how the width parameter affects the distribution.
    """
    print("\n" + "=" * 60)
    print("EFFECT OF RANGE WIDTH (b-a)")
    print("=" * 60)

    # All centered at 0, different widths
    widths = [1, 2, 5, 10]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, width in enumerate(widths):
        ax: Axes = axes[idx // 2, idx % 2]

        a, b = -width / 2, width / 2
        dist = stats.uniform(loc=a, scale=b - a)

        x = np.linspace(-width - 2, width + 2, 1000)
        pdf = dist.pdf(x)

        ax.plot(x, pdf, "b-", linewidth=2)
        ax.fill_between(x, pdf, alpha=0.3, color="blue")
        ax.axvline(a, color="red", linestyle="--", linewidth=2)
        ax.axvline(b, color="red", linestyle="--", linewidth=2)
        ax.axvline(0, color="green", linestyle=":", linewidth=2, label="mean=0")

        height = 1 / width
        std = width / np.sqrt(12)

        stats_text = f"Range: [{a:.1f}, {b:.1f}]\n"
        stats_text += f"Width: {width}\n"
        stats_text += f"Height: {height:.3f}\n"
        stats_text += f"Std Dev: {std:.3f}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        ax.set_xlabel("x")
        ax.set_ylabel("Probability Density")
        ax.set_title(f"U({a:.1f}, {b:.1f}), width={width}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0, top=max(pdf) * 1.2)

    plt.suptitle(
        "Effect of Range Width: Wider Range = Lower Height", fontsize=14, weight="bold"
    )
    plt.tight_layout()
    plt.savefig("uniform_width_effect.png", dpi=150)
    print("Saved: uniform_width_effect.png")
    plt.show()

    print("\nKey insight: Width and height are inversely related")
    print("  Height = 1 / Width")
    print("  This keeps total area = 1 (required for probability)")


def generate_and_visualize_samples():
    """
    Generate random samples from uniform distribution and compare with theory.
    """
    a, b = 0, 10
    sample_sizes = [50, 500, 5000]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, n in enumerate(sample_sizes):
        # Generate samples
        samples = np.random.uniform(a, b, n)

        ax: Axes = axes[idx]

        # Plot histogram
        ax.hist(
            samples,
            bins=30,
            density=True,
            alpha=0.6,
            color="skyblue",
            edgecolor="black",
            label=f"Sample data (n={n})",
        )

        # Overlay theoretical PDF
        x = np.linspace(a, b, 100)
        theoretical_height = 1 / (b - a)
        ax.axhline(
            theoretical_height,
            color="red",
            linewidth=2,
            linestyle="--",
            label="Theoretical PDF",
        )

        # Add boundaries
        ax.axvline(a, color="green", linestyle="--", alpha=0.5)
        ax.axvline(b, color="green", linestyle="--", alpha=0.5)

        # Calculate statistics
        sample_mean = np.mean(samples)
        sample_std = np.std(samples, ddof=1)
        theoretical_mean = (a + b) / 2
        theoretical_std = (b - a) / np.sqrt(12)

        stats_text = f"Sample:\n  mean={sample_mean:.2f}\n  std={sample_std:.2f}\n"
        stats_text += (
            f"Theory:\n  mean={theoretical_mean:.2f}\n  std={theoretical_std:.2f}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=9,
        )

        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title(f"n = {n}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Random Samples from U({a}, {b})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("uniform_samples.png", dpi=150)
    print("Saved: uniform_samples.png")
    plt.show()

    print("\nNotice: As sample size increases, histogram approaches flat line!")


if __name__ == "__main__":
    print("UNIFORM DISTRIBUTION BASICS")
    print("=" * 60)

    print("\n1. Visualizing continuous uniform distributions...")
    visualize_continuous_uniform()

    print("\n2. Visualizing CDF...")
    visualize_cdf()

    print("\n3. Discrete uniform examples...")
    discrete_uniform_examples()

    print("\n4. Probability calculations...")
    probability_calculations()

    print("\n5. Effect of range width...")
    compare_widths()

    print("\n6. Generating and visualizing samples...")
    generate_and_visualize_samples()

    print("\n" + "=" * 60)
    print("KEY FORMULAS (Continuous Uniform U(a, b)):")
    print("=" * 60)
    print("PDF: f(x) = 1/(b-a) for a <= x <= b")
    print("CDF: F(x) = (x-a)/(b-a)")
    print("Mean: (a+b)/2")
    print("Variance: (b-a)^2/12")
    print("Std Dev: (b-a)/sqrt(12)")
    print("=" * 60)
