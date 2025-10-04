"""
Testing for Normality and Analyzing Real Data

In practice, you often need to determine whether your data follows
a normal distribution. This is important because many statistical
tests and methods assume normality.

Methods to Test Normality:
1. Visual inspection (histograms, Q-Q plots)
2. Statistical tests (Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov)
3. Descriptive statistics (skewness, kurtosis)

Q-Q Plot (Quantile-Quantile Plot):
- Compares quantiles of your data vs. quantiles of a theoretical normal distribution
- If data is normal, points should fall approximately on a straight line
- Deviations from the line indicate non-normality

Statistical Tests:
- Null hypothesis (H0): Data comes from a normal distribution
- If p-value < 0.05: reject H0 (data is not normal)
- If p-value ≥ 0.05: fail to reject H0 (data may be normal)
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def create_qq_plot(data, title="Q-Q Plot"):
    """
    Create a Q-Q (Quantile-Quantile) plot to assess normality.

    If the data is normally distributed, the points should roughly
    follow the red diagonal line.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create Q-Q plot
    stats.probplot(data, dist="norm", plot=ax)

    ax.set_title(title, fontsize=14, weight="bold")
    ax.grid(True, alpha=0.3)

    return fig


def test_normality_multiple_methods(data, name="Data"):
    """
    Run multiple normality tests on the data.
    """
    print(f"\n{'=' * 60}")
    print(f"NORMALITY TESTS FOR: {name}")
    print(f"{'=' * 60}")

    # Descriptive statistics
    print(f"\nDescriptive Statistics:")
    print(f"  Sample size: {len(data)}")
    print(f"  Mean: {np.mean(data):.4f}")
    print(f"  Std Dev: {np.std(data, ddof=1):.4f}")
    print(f"  Median: {np.median(data):.4f}")
    print(f"  Skewness: {stats.skew(data):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(data):.4f}")

    print(f"\nInterpretation:")
    print(f"  Skewness near 0: symmetric (normal is symmetric)")
    print(f"  Kurtosis near 0: normal tail behavior")

    # Statistical tests
    print(f"\n{'-' * 60}")
    print("Statistical Tests for Normality:")
    print(f"{'-' * 60}")

    # 1. Shapiro-Wilk Test (good for small to medium samples)
    shapiro_stat, shapiro_p = stats.shapiro(data)
    print(f"\n1. Shapiro-Wilk Test:")
    print(f"   Statistic: {shapiro_stat:.6f}")
    print(f"   P-value: {shapiro_p:.6f}")
    if shapiro_p > 0.05:
        print(f"   Result: FAIL TO REJECT H0 (data may be normal)")
    else:
        print(f"   Result: REJECT H0 (data is likely NOT normal)")

    # 2. Anderson-Darling Test
    anderson_result = stats.anderson(data, dist="norm")
    print(f"\n2. Anderson-Darling Test:")
    print(f"   Statistic: {anderson_result.statistic:.6f}")
    print(f"   Critical values: {anderson_result.critical_values}")
    print(f"   Significance levels: {anderson_result.significance_level}%")
    if anderson_result.statistic < anderson_result.critical_values[2]:  # 5% level
        print(f"   Result: Data appears normal at 5% significance level")
    else:
        print(f"   Result: Data does NOT appear normal at 5% significance level")

    # 3. Kolmogorov-Smirnov Test
    # First normalize the data
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    ks_stat, ks_p = stats.kstest((data - mean) / std, "norm")
    print(f"\n3. Kolmogorov-Smirnov Test:")
    print(f"   Statistic: {ks_stat:.6f}")
    print(f"   P-value: {ks_p:.6f}")
    if ks_p > 0.05:
        print(f"   Result: FAIL TO REJECT H0 (data may be normal)")
    else:
        print(f"   Result: REJECT H0 (data is likely NOT normal)")

    print(f"\n{'=' * 60}\n")


def compare_distributions():
    """
    Compare different types of data:
    - Normal data
    - Slightly skewed data
    - Heavily skewed data
    - Uniform data
    """
    np.random.seed(42)

    datasets = [
        ("Normal", np.random.normal(100, 15, 1000)),
        ("Slightly Skewed", np.random.gamma(5, 2, 1000)),
        ("Heavily Skewed", np.random.exponential(2, 1000)),
        ("Uniform", np.random.uniform(0, 100, 1000)),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(15, 14))

    for idx, (name, data) in enumerate(datasets):
        # Column 1: Histogram
        ax = axes[idx, 0]
        ax.hist(
            data, bins=30, density=True, alpha=0.7, color="skyblue", edgecolor="black"
        )

        # Overlay fitted normal curve
        mu, sigma = np.mean(data), np.std(data, ddof=1)
        x = np.linspace(data.min(), data.max(), 100)
        ax.plot(
            x, stats.norm(mu, sigma).pdf(x), "r-", linewidth=2, label="Fitted Normal"
        )

        ax.set_title(f"{name} - Histogram")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Column 2: Q-Q Plot
        ax = axes[idx, 1]
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(f"{name} - Q-Q Plot")
        ax.grid(True, alpha=0.3)

        # Column 3: Box Plot
        ax = axes[idx, 2]
        ax.boxplot(data, vert=True)
        ax.set_title(f"{name} - Box Plot")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

        # Add statistics text
        shapiro_stat, shapiro_p = stats.shapiro(data)
        skewness = stats.skew(data)
        text = f"Skew: {skewness:.2f}\nShapiro p: {shapiro_p:.4f}"
        axes[idx, 0].text(
            0.02,
            0.98,
            text,
            transform=axes[idx, 0].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.suptitle(
        "Comparing Distributions: Visual and Statistical Assessment",
        fontsize=14,
        weight="bold",
    )
    plt.tight_layout()
    plt.savefig("compare_distributions.png", dpi=150)
    print("Saved: compare_distributions.png")
    plt.show()

    # Print detailed test results
    for name, data in datasets:
        test_normality_multiple_methods(data, name)


def sample_size_effect_on_tests():
    """
    Show how sample size affects normality tests.
    Larger samples are more sensitive to deviations from normality.
    """
    np.random.seed(42)

    # Generate slightly non-normal data (t-distribution with df=10)
    # As df increases, t-distribution approaches normal
    sample_sizes = [30, 100, 500, 2000]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    print("\n" + "=" * 60)
    print("EFFECT OF SAMPLE SIZE ON NORMALITY TESTS")
    print("=" * 60)
    print("\nUsing t-distribution with df=10 (slightly non-normal)")

    for idx, n in enumerate(sample_sizes):
        data = stats.t.rvs(df=10, size=n)

        ax = axes[idx]
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(f"n = {n}")
        ax.grid(True, alpha=0.3)

        # Run Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(data)

        # Add test results
        text = f"Shapiro-Wilk:\np = {shapiro_p:.6f}\n"
        if shapiro_p > 0.05:
            text += "Appears normal"
            color = "lightgreen"
        else:
            text += "Not normal"
            color = "lightcoral"

        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor=color, alpha=0.8),
        )

        print(f"\nn = {n}:")
        print(f"  Shapiro-Wilk p-value: {shapiro_p:.6f}")
        print(
            f"  Conclusion: {'Appears normal' if shapiro_p > 0.05 else 'Significantly non-normal'}"
        )

    plt.suptitle(
        "Sample Size Effect: Same Distribution, Different Sample Sizes",
        fontsize=14,
        weight="bold",
    )
    plt.tight_layout()
    plt.savefig("sample_size_effect.png", dpi=150)
    print("\nSaved: sample_size_effect.png")
    plt.show()

    print("\nKey insight:")
    print("  Larger samples are more sensitive to small deviations from normality.")
    print("  This is why visual inspection (Q-Q plots) is also important!")


def demonstrate_transformations():
    """
    Show how to transform non-normal data to approximate normality.
    Common transformations:
    - Log transformation (for right-skewed data)
    - Square root transformation (for moderately skewed data)
    - Box-Cox transformation (data-driven)
    """
    np.random.seed(42)

    # Generate right-skewed data (exponential)
    data = np.random.exponential(2, 1000)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    transformations = [
        ("Original", data, lambda x: x),
        ("Log", data[data > 0], np.log),
        ("Square Root", data, np.sqrt),
        ("Box-Cox", data[data > 0], lambda x: stats.boxcox(x)[0]),
    ]

    for idx, (name, d, transform) in enumerate(transformations):
        transformed = transform(d)

        # Top row: Histogram
        ax = axes[0, idx]
        ax.hist(
            transformed,
            bins=30,
            density=True,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )

        # Fit and overlay normal
        mu, sigma = np.mean(transformed), np.std(transformed, ddof=1)
        x = np.linspace(transformed.min(), transformed.max(), 100)
        ax.plot(x, stats.norm(mu, sigma).pdf(x), "r-", linewidth=2)

        ax.set_title(f"{name}")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)

        # Bottom row: Q-Q plot
        ax = axes[1, idx]
        stats.probplot(transformed, dist="norm", plot=ax)
        ax.grid(True, alpha=0.3)

        # Test normality
        shapiro_stat, shapiro_p = stats.shapiro(transformed)
        skewness = stats.skew(transformed)

        text = f"Skew: {skewness:.2f}\np: {shapiro_p:.4f}"
        axes[0, idx].text(
            0.02,
            0.98,
            text,
            transform=axes[0, idx].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.suptitle(
        "Transforming Non-Normal Data to Approximate Normality",
        fontsize=14,
        weight="bold",
    )
    plt.tight_layout()
    plt.savefig("transformations.png", dpi=150)
    print("Saved: transformations.png")
    plt.show()

    print("\n" + "=" * 60)
    print("TRANSFORMATION RESULTS")
    print("=" * 60)

    for name, d, transform in transformations:
        transformed = transform(d)
        shapiro_stat, shapiro_p = stats.shapiro(transformed)
        skewness = stats.skew(transformed)

        print(f"\n{name}:")
        print(f"  Skewness: {skewness:.4f}")
        print(f"  Shapiro-Wilk p-value: {shapiro_p:.6f}")
        print(
            f"  Conclusion: {'Appears normal' if shapiro_p > 0.05 else 'Still non-normal'}"
        )


def real_world_example():
    """
    Analyze a simulated real-world dataset: heights of adult men and women.
    """
    np.random.seed(42)

    # Simulate heights: men ~ N(175, 7), women ~ N(162, 6.5)
    men_heights = np.random.normal(175, 7, 500)
    women_heights = np.random.normal(162, 6.5, 500)
    combined_heights = np.concatenate([men_heights, women_heights])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    datasets = [
        ("Men Only", men_heights),
        ("Women Only", women_heights),
        ("Combined (Bimodal)", combined_heights),
    ]

    for idx, (name, data) in enumerate(datasets):
        # Histogram
        ax = axes[0, idx]
        ax.hist(
            data, bins=30, density=True, alpha=0.7, color="skyblue", edgecolor="black"
        )

        # Fit normal
        mu, sigma = np.mean(data), np.std(data, ddof=1)
        x = np.linspace(data.min(), data.max(), 100)
        ax.plot(
            x, stats.norm(mu, sigma).pdf(x), "r-", linewidth=2, label="Fitted Normal"
        )

        ax.set_title(f"{name}")
        ax.set_xlabel("Height (cm)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Q-Q Plot
        ax = axes[1, idx]
        stats.probplot(data, dist="norm", plot=ax)
        ax.grid(True, alpha=0.3)

        # Statistics
        shapiro_stat, shapiro_p = stats.shapiro(data)
        text = f"n={len(data)}\nμ={mu:.1f}\nσ={sigma:.1f}\np={shapiro_p:.4f}"
        axes[0, idx].text(
            0.02,
            0.98,
            text,
            transform=axes[0, idx].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.suptitle(
        "Real World Example: Importance of Understanding Your Data",
        fontsize=14,
        weight="bold",
    )
    plt.tight_layout()
    plt.savefig("real_world_heights.png", dpi=150)
    print("Saved: real_world_heights.png")
    plt.show()

    print("\n" + "=" * 60)
    print("REAL-WORLD EXAMPLE: HEIGHTS")
    print("=" * 60)

    for name, data in datasets:
        test_normality_multiple_methods(data, name)

    print("\nKey insight:")
    print("  Men's and women's heights are each normal, but combining them")
    print("  creates a bimodal distribution that is NOT normal!")
    print("  This shows why understanding your data is crucial.")


if __name__ == "__main__":
    print("TESTING FOR NORMALITY AND DATA ANALYSIS")
    print("=" * 60)

    print("\n1. Comparing different distributions...")
    compare_distributions()

    print("\n2. Sample size effect on normality tests...")
    sample_size_effect_on_tests()

    print("\n3. Transforming non-normal data...")
    demonstrate_transformations()

    print("\n4. Real-world example: heights...")
    real_world_example()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("1. Use multiple methods to assess normality (visual + statistical)")
    print("2. Q-Q plots are your friend - they show HOW data deviates from normal")
    print("3. Larger samples are more sensitive to small deviations")
    print("4. Consider transformations for skewed data")
    print("5. Understand your data - don't just apply tests blindly!")
    print("=" * 60)
