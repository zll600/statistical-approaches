"""
Normal Distribution Basics

The normal distribution (also called Gaussian distribution) is one of the most important
probability distributions in statistics. It's characterized by its bell-shaped curve.

Key Properties:
- Defined by two parameters: μ (mean) and σ (standard deviation)
- μ controls the center (location) of the distribution
- σ controls the spread (width) of the distribution
- Symmetric around the mean
- Total area under the curve = 1 (represents 100% probability)

The 68-95-99.7 Rule (Empirical Rule):
- 68% of data falls within 1 standard deviation of the mean (μ ± σ)
- 95% of data falls within 2 standard deviations of the mean (μ ± 2σ)
- 99.7% of data falls within 3 standard deviations of the mean (μ ± 3σ)
"""

from typing import Any
import numpy as np
from numpy.typing import NDArray
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def visualize_pdf_with_different_parameters() -> None:
    """
    PDF (Probability Density Function) shows the relative likelihood
    of different values occurring.

    For a normal distribution, the PDF is highest at the mean and
    decreases symmetrically on both sides.
    """
    x: NDArray[np.floating[Any]] = np.linspace(-10, 20, 1000)

    # Create multiple normal distributions with different parameters
    distributions = [
        (0, 1, "μ=0, σ=1 (Standard Normal)"),
        (0, 2, "μ=0, σ=2 (Wider spread)"),
        (5, 1, "μ=5, σ=1 (Shifted right)"),
        (5, 0.5, "μ=5, σ=0.5 (Narrow spread)"),
    ]

    plt.figure(figsize=(12, 6))

    for mu, sigma, label in distributions:
        # Create a normal distribution object
        # dist is a rc_continuous_fronzed type
        dist = stats.norm(mu, sigma)
        # Calculate PDF values
        pdf: NDArray[np.floating[Any]] = dist.pdf(x)
        plt.plot(x, pdf, label=label, linewidth=2)

    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.title("Normal Distribution PDF: Effect of μ (mean) and σ (std dev)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig("normal_pdf_comparison.png", dpi=150)
    # print("Saved: normal_pdf_comparison.png")
    plt.show()


def visualize_cdf() -> None:
    """
    CDF (Cumulative Distribution Function) shows the probability that
    a random variable is less than or equal to a specific value.

    The CDF always increases from 0 to 1.
    At the mean, CDF = 0.5 (50% probability of being below the mean)
    """
    x: NDArray[np.floating[Any]] = np.linspace(-5, 15, 1000)

    distributions = [(0, 1, "μ=0, σ=1"), (0, 2, "μ=0, σ=2"), (5, 1, "μ=5, σ=1")]

    plt.figure(figsize=(12, 6))

    for mu, sigma, label in distributions:
        dist = stats.norm(mu, sigma)
        cdf: NDArray[np.floating[Any]] = dist.cdf(x)
        plt.plot(x, cdf, label=label, linewidth=2)

    # Add horizontal line at 0.5
    plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% probability")

    plt.xlabel("x")
    plt.ylabel("Cumulative Probability")
    plt.title("Normal Distribution CDF")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig("normal_cdf_comparison.png", dpi=150)
    # print("Saved: normal_cdf_comparison.png")
    plt.show()


def demonstrate_empirical_rule() -> None:
    """
    The 68-95-99.7 rule is a quick way to understand where most
    of the data lies in a normal distribution.
    """
    mu: int = 100
    sigma: int = 15  # Example: IQ scores
    x: NDArray[np.floating[Any]] = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)

    dist = stats.norm(mu, sigma)
    pdf: NDArray[np.floating[Any]] = dist.pdf(x)

    fig: Figure
    axes: NDArray[Any]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 68% rule (1 standard deviation)
    ax: Axes = axes[0]
    ax.plot(x, pdf, "b-", linewidth=2)
    ax.fill_between(
        x,
        pdf,
        where=(x >= mu - sigma) & (x <= mu + sigma),
        alpha=0.3,
        color="blue",
        label="68% of data",
    )
    ax.axvline(mu, color="red", linestyle="--", linewidth=2, label=f"Mean (μ={mu})")
    ax.axvline(mu - sigma, color="green", linestyle="--", alpha=0.7)
    ax.axvline(mu + sigma, color="green", linestyle="--", alpha=0.7)
    ax.set_title(f"68% Rule: μ ± σ = [{mu - sigma}, {mu + sigma}]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 95% rule (2 standard deviations)
    ax = axes[1]
    ax.plot(x, pdf, "b-", linewidth=2)
    ax.fill_between(
        x,
        pdf,
        where=(x >= mu - 2 * sigma) & (x <= mu + 2 * sigma),
        alpha=0.3,
        color="orange",
        label="95% of data",
    )
    ax.axvline(mu, color="red", linestyle="--", linewidth=2, label=f"Mean (μ={mu})")
    ax.axvline(mu - 2 * sigma, color="green", linestyle="--", alpha=0.7)
    ax.axvline(mu + 2 * sigma, color="green", linestyle="--", alpha=0.7)
    ax.set_title(f"95% Rule: μ ± 2σ = [{mu - 2 * sigma}, {mu + 2 * sigma}]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 99.7% rule (3 standard deviations)
    ax = axes[2]
    ax.plot(x, pdf, "b-", linewidth=2)
    ax.fill_between(
        x,
        pdf,
        where=(x >= mu - 3 * sigma) & (x <= mu + 3 * sigma),
        alpha=0.3,
        color="green",
        label="99.7% of data",
    )
    ax.axvline(mu, color="red", linestyle="--", linewidth=2, label=f"Mean (μ={mu})")
    ax.axvline(mu - 3 * sigma, color="green", linestyle="--", alpha=0.7)
    ax.axvline(mu + 3 * sigma, color="green", linestyle="--", alpha=0.7)
    ax.set_title(f"99.7% Rule: μ ± 3σ = [{mu - 3 * sigma}, {mu + 3 * sigma}]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.savefig("empirical_rule.png", dpi=150)
    # print("Saved: empirical_rule.png")
    plt.show()


def probability_calculations() -> None:
    """
    Practical examples of calculating probabilities using normal distribution.
    """
    print("\n" + "=" * 60)
    print("PROBABILITY CALCULATIONS WITH NORMAL DISTRIBUTION")
    print("=" * 60)

    # Example: IQ scores are normally distributed with μ=100, σ=15
    mu: int = 100
    sigma: int = 15
    dist = stats.norm(mu, sigma)

    print(f"\nExample: IQ scores ~ N(μ={mu}, σ={sigma})")
    print("-" * 60)

    # Question 1: What's the probability of IQ < 115?
    prob_below_115: float = dist.cdf(115)
    print(f"\n1. P(IQ < 115) = {prob_below_115:.4f} ({prob_below_115 * 100:.2f}%)")

    # Question 2: What's the probability of IQ > 130?
    prob_above_130: float = 1 - dist.cdf(130)
    print(f"2. P(IQ > 130) = {prob_above_130:.4f} ({prob_above_130 * 100:.2f}%)")

    # Question 3: What's the probability of 85 < IQ < 115?
    prob_between: float = dist.cdf(115) - dist.cdf(85)
    print(f"3. P(85 < IQ < 115) = {prob_between:.4f} ({prob_between * 100:.2f}%)")

    # Question 4: What IQ score is at the 90th percentile?
    iq_90th: float = dist.ppf(0.90)  # ppf = percent point function (inverse of CDF)
    print(f"4. 90th percentile IQ = {iq_90th:.2f}")
    print(f"   (90% of people have IQ below {iq_90th:.2f})")

    # Question 5: What IQ range contains the middle 50% of people?
    iq_25th: float = dist.ppf(0.25)
    iq_75th: float = dist.ppf(0.75)
    print(f"5. Middle 50% (25th to 75th percentile): [{iq_25th:.2f}, {iq_75th:.2f}]")


def generate_and_visualize_samples() -> None:
    """
    Generate random samples from a normal distribution and compare
    with the theoretical distribution.
    """
    mu: int = 10
    sigma: int = 2
    sample_sizes: list[int] = [50, 500, 5000]

    fig: Figure
    axes: NDArray[Any]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x: NDArray[np.floating[Any]] = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
    theoretical_pdf: NDArray[np.floating[Any]] = stats.norm(mu, sigma).pdf(x)

    for idx, n in enumerate(sample_sizes):
        # Generate random samples
        samples: NDArray[np.floating[Any]] = np.random.normal(mu, sigma, n)

        ax: Axes = axes[idx]
        # Plot histogram of samples
        ax.hist(
            samples,
            bins=30,
            density=True,
            alpha=0.6,
            color="skyblue",
            edgecolor="black",
            label=f"Sample data (n={n})",
        )
        # Plot theoretical PDF
        ax.plot(x, theoretical_pdf, "r-", linewidth=2, label="Theoretical PDF")
        ax.axvline(mu, color="green", linestyle="--", linewidth=2, label=f"μ={mu}")

        # Add statistics
        sample_mean: float = np.mean(samples)
        sample_std: float = np.std(samples, ddof=1)
        ax.text(
            0.02,
            0.98,
            f"Sample μ={sample_mean:.2f}\nSample σ={sample_std:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title(f"n = {n}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Random Samples from N(μ={mu}, σ={sigma})", fontsize=14, y=1.02)
    plt.tight_layout()
    # plt.savefig("normal_samples.png", dpi=150)
    # print("Saved: normal_samples.png")
    plt.show()

    print(
        "\nNotice: As sample size increases, the histogram looks more like the theoretical curve!"
    )


if __name__ == "__main__":
    print("NORMAL DISTRIBUTION BASICS")
    print("=" * 60)

    print("\n1. Visualizing PDF with different parameters...")
    visualize_pdf_with_different_parameters()

    print("\n2. Visualizing CDF...")
    visualize_cdf()

    print("\n3. Demonstrating the 68-95-99.7 Rule...")
    demonstrate_empirical_rule()

    print("\n4. Probability calculations...")
    probability_calculations()

    print("\n5. Generating and visualizing random samples...")
    generate_and_visualize_samples()

    print("\n" + "=" * 60)
    print("All visualizations complete! Check the saved PNG files.")
    print("=" * 60)
