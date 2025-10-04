"""
Understanding Likelihood in Probability Theory

Key Concepts:
--------------
1. PROBABILITY vs LIKELIHOOD - The crucial difference:
   - Probability: Given parameters, what's the chance of observing data?
     P(data | parameters) - parameters are KNOWN, data varies

   - Likelihood: Given observed data, how likely are different parameters?
     L(parameters | data) - data is FIXED, parameters vary

2. Maximum Likelihood Estimation (MLE):
   Find the parameter values that make the observed data most likely.

Real-World Application:
You observe some data and want to find the "best" model parameters.
"""

from typing import Any
import numpy as np
from numpy.typing import NDArray
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def coins_flipping() -> None:
    """
    Example 1: Coin Flipping - Understanding the Difference

    Scenario: You flip a coin 10 times and get 7 heads.
    """
    print("=" * 70)
    print("EXAMPLE 1: PROBABILITY vs LIKELIHOOD")
    print("=" * 70)

    n_flips: int = 10
    n_heads: int = 7

    print(f"\nObserved: {n_heads} heads in {n_flips} flips")
    print("\n" + "-" * 70)

    # PROBABILITY: If coin is fair (p=0.5), what's P(7 heads)?
    p_fair: float = 0.5
    prob_7_heads_given_fair: float = stats.binom.pmf(n_heads, n_flips, p_fair)

    print("\n1. PROBABILITY QUESTION:")
    print(f"   IF the coin is fair (p={p_fair}), what's P(getting {n_heads} heads)?")
    print(
        f"   Answer: {prob_7_heads_given_fair:.4f} ({prob_7_heads_given_fair * 100:.2f}%)"
    )
    print(f"   → Parameters FIXED, data varies")

    # LIKELIHOOD: Given we saw 7 heads, what values of p are more likely?
    print("\n2. LIKELIHOOD QUESTION:")
    print(f"   GIVEN we observed {n_heads} heads, which probability p is most likely?")

    # Calculate likelihood for different values of p
    p_values: NDArray[np.floating[Any]] = np.linspace(0.01, 0.99, 100)
    likelihoods: NDArray[np.floating[Any]] = np.array(
        [stats.binom.pmf(n_heads, n_flips, p) for p in p_values]
    )

    # Find maximum likelihood estimate
    max_idx: int = int(np.argmax(likelihoods))
    mle_p: float = p_values[max_idx]

    print(f"   Answer: p = {mle_p:.2f} (Maximum Likelihood Estimate)")
    print(f"   → Data FIXED, parameters vary")

    # Visualize
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    x: NDArray[np.floating[Any]] = np.arange(0, 11)
    probs: NDArray[np.floating[Any]] = stats.binom.pmf(x, n_flips, p_fair)
    plt.bar(x, probs, alpha=0.7, color="skyblue", edgecolor="black")
    plt.axvline(
        n_heads,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Observed: {n_heads} heads",
    )
    plt.xlabel("Number of Heads")
    plt.ylabel("Probability")
    plt.title(f"PROBABILITY: P(heads | p={p_fair})\nParameters fixed, data varies")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(p_values, likelihoods, "b-", linewidth=2)
    plt.axvline(
        mle_p, color="red", linestyle="--", linewidth=2, label=f"MLE: p={mle_p:.2f}"
    )
    plt.axvline(
        p_fair,
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label=f"Fair coin: p={p_fair}",
    )
    plt.xlabel("Probability of Heads (p)")
    plt.ylabel("Likelihood")
    plt.title(
        f"LIKELIHOOD: L(p | {n_heads} heads observed)\nData fixed, parameters vary"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def normal_distribution_mle() -> None:
    """
    Example 2: Estimating Mean and Std Dev of Normal Distribution

    Real scenario: You measure heights of 50 people.
    What are the best μ and σ for the population?
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: MAXIMUM LIKELIHOOD ESTIMATION - Normal Distribution")
    print("=" * 70)

    # Simulate: True population has μ=170, σ=10 (unknown to us)
    np.random.seed(42)
    true_mu: float = 170.0
    true_sigma: float = 10.0
    sample_size: int = 50

    # Observed data: heights of 50 people
    observed_heights: NDArray[np.floating[Any]] = np.random.normal(
        true_mu, true_sigma, sample_size
    )

    print(f"\nObserved: {sample_size} height measurements")
    print(f"Sample mean: {np.mean(observed_heights):.2f}")
    print(f"Sample std: {np.std(observed_heights, ddof=1):.2f}")

    # Maximum Likelihood Estimates
    mle_mu: float = np.mean(observed_heights)
    mle_sigma: float = np.std(observed_heights, ddof=0)  # MLE uses ddof=0

    print(f"\nMaximum Likelihood Estimates:")
    print(f"  μ_MLE = {mle_mu:.2f} (sample mean)")
    print(f"  σ_MLE = {mle_sigma:.2f}")
    print(f"\nTrue values (unknown in practice):")
    print(f"  μ_true = {true_mu:.2f}")
    print(f"  σ_true = {true_sigma:.2f}")

    # Likelihood function for different μ values (σ fixed at MLE)
    mu_values: NDArray[np.floating[Any]] = np.linspace(mle_mu - 5, mle_mu + 5, 100)
    log_likelihoods_mu: list[float] = []

    for mu in mu_values:
        # Log-likelihood: sum of log probabilities
        log_lik: float = float(
            np.sum(stats.norm.logpdf(observed_heights, mu, mle_sigma))
        )
        log_likelihoods_mu.append(log_lik)

    # Visualize
    fig: Figure
    axes: NDArray[Any]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Data histogram with fitted distribution
    ax: Axes = axes[0]
    ax.hist(
        observed_heights,
        bins=15,
        density=True,
        alpha=0.6,
        color="skyblue",
        edgecolor="black",
        label="Observed data",
    )

    x: NDArray[np.floating[Any]] = np.linspace(
        observed_heights.min() - 10, observed_heights.max() + 10, 200
    )

    # MLE distribution
    pdf_mle: NDArray[np.floating[Any]] = stats.norm.pdf(x, mle_mu, mle_sigma)
    ax.plot(
        x, pdf_mle, "r-", linewidth=2, label=f"MLE: μ={mle_mu:.1f}, σ={mle_sigma:.1f}"
    )

    # True distribution
    pdf_true: NDArray[np.floating[Any]] = stats.norm.pdf(x, true_mu, true_sigma)
    ax.plot(
        x,
        pdf_true,
        "g--",
        linewidth=2,
        alpha=0.7,
        label=f"True: μ={true_mu:.1f}, σ={true_sigma:.1f}",
    )

    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Density")
    ax.set_title("Data with MLE vs True Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Likelihood function
    ax = axes[1]
    ax.plot(mu_values, log_likelihoods_mu, "b-", linewidth=2)
    ax.axvline(
        mle_mu, color="red", linestyle="--", linewidth=2, label=f"MLE: μ={mle_mu:.2f}"
    )
    ax.axvline(
        true_mu,
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"True: μ={true_mu:.2f}",
    )
    ax.set_xlabel("Mean (μ)")
    ax.set_ylabel("Log-Likelihood")
    ax.set_title("Likelihood Function for μ (σ fixed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def real_world_dice() -> None:
    """
    Example 3: Is This Dice Fair?

    Real scenario: You suspect a dice might be loaded.
    Roll it 60 times and observe the results.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: REAL-WORLD - Is This Dice Fair?")
    print("=" * 70)

    # Simulated data: dice is slightly loaded (6 appears more often)
    np.random.seed(42)
    # Unfair probabilities: [0.15, 0.15, 0.15, 0.15, 0.15, 0.25]
    observed_rolls: NDArray[np.int_] = np.random.choice(
        [1, 2, 3, 4, 5, 6], size=60, p=[0.15, 0.15, 0.15, 0.15, 0.15, 0.25]
    )

    # Count occurrences
    unique: NDArray[np.int_]
    counts: NDArray[np.int_]
    unique, counts = np.unique(observed_rolls, return_counts=True)

    print(f"\nObserved {len(observed_rolls)} dice rolls:")
    for face, count in zip(unique, counts):
        print(
            f"  Face {face}: {count} times ({count / len(observed_rolls) * 100:.1f}%)"
        )

    # Compare with expected (fair dice)
    expected_count: float = len(observed_rolls) / 6
    print(f"\nExpected (fair dice): {expected_count:.1f} times each face")

    # Calculate likelihood for fair dice
    fair_probs: NDArray[np.floating[Any]] = np.array([1 / 6] * 6)
    log_lik_fair: float = 0.0

    for face in observed_rolls:
        log_lik_fair += np.log(fair_probs[face - 1])

    # Calculate likelihood for MLE (observed frequencies)
    mle_probs: NDArray[np.floating[Any]] = counts / len(observed_rolls)
    log_lik_mle: float = 0.0

    for face in observed_rolls:
        log_lik_mle += np.log(mle_probs[face - 1])

    print(f"\nLog-Likelihood Comparison:")
    print(f"  Fair dice hypothesis: {log_lik_fair:.2f}")
    print(f"  MLE (observed frequencies): {log_lik_mle:.2f}")
    print(f"  Difference: {log_lik_mle - log_lik_fair:.2f}")

    if log_lik_mle > log_lik_fair + 2:  # Rule of thumb
        print("\n️  Evidence suggests dice might be LOADED!")
    else:
        print("\n✓ Data consistent with fair dice (within random variation)")

    # Visualize
    plt.figure(figsize=(12, 5))

    # Observed vs Expected
    plt.subplot(1, 2, 1)
    x_pos: NDArray[np.floating[Any]] = np.arange(1, 7)
    plt.bar(
        x_pos - 0.2,
        counts,
        width=0.4,
        label="Observed",
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    plt.bar(
        x_pos + 0.2,
        [expected_count] * 6,
        width=0.4,
        label="Expected (fair)",
        alpha=0.7,
        color="lightcoral",
        edgecolor="black",
    )
    plt.xlabel("Dice Face")
    plt.ylabel("Count")
    plt.title("Observed vs Expected Frequencies")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.xticks(x_pos)

    # Probabilities
    plt.subplot(1, 2, 2)
    plt.bar(
        x_pos - 0.2,
        mle_probs,
        width=0.4,
        label="MLE (observed)",
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    plt.bar(
        x_pos + 0.2,
        fair_probs,
        width=0.4,
        label="Fair dice",
        alpha=0.7,
        color="lightcoral",
        edgecolor="black",
    )
    plt.axhline(y=1 / 6, color="red", linestyle="--", alpha=0.5)
    plt.xlabel("Dice Face")
    plt.ylabel("Probability")
    plt.title("Estimated Probabilities")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.xticks(x_pos)

    plt.tight_layout()
    plt.show()


def likelihood_ratio_test() -> None:
    """
    Example 4: Hypothesis Testing with Likelihood Ratios

    Compare two models: Which one fits the data better?
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: LIKELIHOOD RATIO TEST")
    print("=" * 70)

    # Generate data from two different normal distributions (mixture)
    np.random.seed(42)
    group1: NDArray[np.floating[Any]] = np.random.normal(100, 15, 30)
    group2: NDArray[np.floating[Any]] = np.random.normal(110, 15, 30)
    all_data: NDArray[np.floating[Any]] = np.concatenate([group1, group2])

    print("\nScenario: Test scores from two teaching methods")
    print(f"Total samples: {len(all_data)}")

    # Model 1 (Null): Same mean for both groups
    mu_combined: float = np.mean(all_data)
    sigma_combined: float = np.std(all_data, ddof=1)
    log_lik_null: float = float(
        np.sum(stats.norm.logpdf(all_data, mu_combined, sigma_combined))
    )

    print(f"\nModel 1 (Null Hypothesis): Same distribution")
    print(f"  μ = {mu_combined:.2f}, σ = {sigma_combined:.2f}")
    print(f"  Log-likelihood: {log_lik_null:.2f}")

    # Model 2 (Alternative): Different means for each group
    mu1: float = np.mean(group1)
    mu2: float = np.mean(group2)
    sigma1: float = np.std(group1, ddof=1)
    sigma2: float = np.std(group2, ddof=1)

    log_lik_alt: float = float(np.sum(stats.norm.logpdf(group1, mu1, sigma1))) + float(
        np.sum(stats.norm.logpdf(group2, mu2, sigma2))
    )

    print(f"\nModel 2 (Alternative): Different distributions")
    print(f"  Group 1: μ = {mu1:.2f}, σ = {sigma1:.2f}")
    print(f"  Group 2: μ = {mu2:.2f}, σ = {sigma2:.2f}")
    print(f"  Log-likelihood: {log_lik_alt:.2f}")

    # Likelihood ratio test
    lr_stat: float = 2 * (log_lik_alt - log_lik_null)
    print(f"\nLikelihood Ratio Test Statistic: {lr_stat:.2f}")

    # Chi-square approximation (df = difference in parameters)
    df: int = 2  # 2 additional parameters in alternative model
    p_value: float = 1 - stats.chi2.cdf(lr_stat, df)

    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("\n✓ REJECT null hypothesis: Teaching methods have DIFFERENT effects!")
    else:
        print("\n✗ Cannot reject null hypothesis: No significant difference")

    # Visualize
    plt.figure(figsize=(14, 5))

    # Data distributions
    plt.subplot(1, 2, 1)
    plt.hist(
        group1,
        bins=15,
        alpha=0.6,
        label="Group 1 (Method A)",
        color="skyblue",
        edgecolor="black",
        density=True,
    )
    plt.hist(
        group2,
        bins=15,
        alpha=0.6,
        label="Group 2 (Method B)",
        color="lightcoral",
        edgecolor="black",
        density=True,
    )

    x: NDArray[np.floating[Any]] = np.linspace(60, 140, 200)
    plt.plot(x, stats.norm.pdf(x, mu1, sigma1), "b-", linewidth=2, alpha=0.7)
    plt.plot(x, stats.norm.pdf(x, mu2, sigma2), "r-", linewidth=2, alpha=0.7)
    plt.plot(
        x,
        stats.norm.pdf(x, mu_combined, sigma_combined),
        "g--",
        linewidth=2,
        label="Combined (Null)",
        alpha=0.7,
    )

    plt.xlabel("Test Score")
    plt.ylabel("Density")
    plt.title("Data Distributions: Two Groups")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Log-likelihood comparison
    plt.subplot(1, 2, 2)
    models: list[str] = ["Null\n(Same Mean)", "Alternative\n(Different Means)"]
    log_liks: list[float] = [log_lik_null, log_lik_alt]
    colors: list[str] = ["lightcoral", "lightgreen"]

    bars = plt.bar(models, log_liks, color=colors, alpha=0.7, edgecolor="black")
    plt.ylabel("Log-Likelihood")
    plt.title(f"Model Comparison (LR stat = {lr_stat:.2f}, p = {p_value:.4f})")
    plt.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, log_liks):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 2,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show()


def practical_regression() -> None:
    """
    Example 5: Linear Regression via Maximum Likelihood

    Real scenario: Predict house prices from size.
    MLE finds the best-fit line!
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: LINEAR REGRESSION AS MAXIMUM LIKELIHOOD")
    print("=" * 70)

    # Generate data: house size vs price (with noise)
    np.random.seed(42)
    house_sizes: NDArray[np.floating[Any]] = np.random.uniform(
        50, 200, 50
    )  # square meters
    true_slope: float = 2.5  # price increases $2.5k per sq meter
    true_intercept: float = 50.0  # base price $50k
    noise: NDArray[np.floating[Any]] = np.random.normal(0, 20, 50)

    prices: NDArray[np.floating[Any]] = (
        true_slope * house_sizes + true_intercept + noise
    )

    print("\nData: 50 houses (size vs price)")
    print(f"True model: Price = {true_slope} × Size + {true_intercept}")

    # MLE for linear regression (same as least squares!)
    X: NDArray[np.floating[Any]] = np.column_stack(
        [np.ones(len(house_sizes)), house_sizes]
    )
    # MLE estimates (β = (X'X)^(-1) X'y)
    beta: NDArray[np.floating[Any]] = np.linalg.inv(X.T @ X) @ X.T @ prices
    mle_intercept: float = float(beta[0])
    mle_slope: float = float(beta[1])

    # Residual standard error (MLE for σ)
    predictions: NDArray[np.floating[Any]] = mle_intercept + mle_slope * house_sizes
    residuals: NDArray[np.floating[Any]] = prices - predictions
    mle_sigma: float = float(np.sqrt(np.sum(residuals**2) / len(prices)))

    print(f"\nMLE estimates:")
    print(f"  Intercept (β₀) = {mle_intercept:.2f}")
    print(f"  Slope (β₁) = {mle_slope:.2f}")
    print(f"  Residual σ = {mle_sigma:.2f}")

    # Calculate log-likelihood
    log_lik: float = float(np.sum(stats.norm.logpdf(prices, predictions, mle_sigma)))
    print(f"\nLog-likelihood: {log_lik:.2f}")

    # Visualize
    plt.figure(figsize=(14, 5))

    # Scatter plot with regression line
    plt.subplot(1, 2, 1)
    plt.scatter(
        house_sizes,
        prices,
        alpha=0.6,
        color="skyblue",
        edgecolor="black",
        s=50,
        label="Observed data",
    )

    # True line
    x_line: NDArray[np.floating[Any]] = np.linspace(50, 200, 100)
    y_true: NDArray[np.floating[Any]] = true_slope * x_line + true_intercept
    plt.plot(
        x_line,
        y_true,
        "g--",
        linewidth=2,
        alpha=0.7,
        label=f"True: y = {true_slope}x + {true_intercept}",
    )

    # MLE line
    y_mle: NDArray[np.floating[Any]] = mle_slope * x_line + mle_intercept
    plt.plot(
        x_line,
        y_mle,
        "r-",
        linewidth=2,
        label=f"MLE: y = {mle_slope:.2f}x + {mle_intercept:.2f}",
    )

    plt.xlabel("House Size (sq meters)")
    plt.ylabel("Price ($1000s)")
    plt.title("Linear Regression via MLE")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Residuals distribution
    plt.subplot(1, 2, 2)
    plt.hist(
        residuals,
        bins=15,
        density=True,
        alpha=0.6,
        color="skyblue",
        edgecolor="black",
        label="Residuals",
    )

    x_res: NDArray[np.floating[Any]] = np.linspace(
        residuals.min(), residuals.max(), 100
    )
    pdf_res: NDArray[np.floating[Any]] = stats.norm.pdf(x_res, 0, mle_sigma)
    plt.plot(x_res, pdf_res, "r-", linewidth=2, label=f"Normal(0, {mle_sigma:.2f})")

    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.title("Residuals Distribution (MLE assumes normality)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n Key Insight: MLE for linear regression = Least squares regression!")
    print("   Both minimize the same objective function.")


if __name__ == "__main__":
    print("UNDERSTANDING LIKELIHOOD IN PROBABILITY THEORY")
    print("Real Python Examples from Basic to Advanced")

    print("\nWhat you'll learn:")
    print("  1. Probability vs Likelihood - The fundamental difference")
    print("  2. Maximum Likelihood Estimation (MLE) for normal distributions")
    print("  3. Real-world application: Testing if a dice is fair")
    print("  4. Likelihood ratio tests for hypothesis testing")
    print("  5. Linear regression as maximum likelihood estimation")

    input("\nPress Enter to start Example 1...")
    coins_flipping()

    input("\nPress Enter to continue to Example 2...")
    normal_distribution_mle()

    input("\nPress Enter to continue to Example 3...")
    real_world_dice()

    input("\nPress Enter to continue to Example 4...")
    likelihood_ratio_test()

    input("\nPress Enter to continue to Example 5...")
    practical_regression()

    print("\n" + "=" * 70)
    print("🎉 TUTORIAL COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  ✓ Likelihood helps us find best parameters from observed data")
    print("  ✓ MLE is widely used: regression, classification, neural networks")
    print("  ✓ Likelihood ratios help compare competing hypotheses")
    print("  ✓ Understanding likelihood is key to modern machine learning")
    print("=" * 70)
