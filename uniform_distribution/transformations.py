"""
Using Uniform Distribution to Generate Other Distributions

KEY CONCEPT: All pseudo-random number generators start with U(0,1)
and then transform it to create other distributions!

Methods covered:
1. Inverse Transform Method - Most general approach
2. Box-Muller Transform - Specifically for normal distribution
3. Acceptance-Rejection Sampling - For complex distributions
4. Direct transformations - For simple cases

Why this matters:
- Understanding the foundation of random number generation
- Implementing custom distributions
- Appreciating how RNG libraries work under the hood
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def inverse_transform_method():
    """
    Inverse Transform Method: U ~ Uniform(0,1) -> X = F^(-1)(U)

    Where F^(-1) is the inverse CDF of the target distribution.

    This works because if U ~ Uniform(0,1), then F^(-1)(U) ~ F
    """
    print("\n" + "=" * 60)
    print("INVERSE TRANSFORM METHOD")
    print("=" * 60)
    print("\nMethod: Generate U ~ Uniform(0,1), then X = F_inverse(U)")
    print("where F_inverse is the inverse CDF of target distribution")

    n = 10000
    uniform_samples = np.random.uniform(0, 1, n)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Example 1: Exponential distribution
    # F(x) = 1 - exp(-lambda*x), F^(-1)(u) = -ln(1-u)/lambda
    ax = axes[0, 0]
    lambda_param = 1.0
    exp_samples = -np.log(1 - uniform_samples) / lambda_param

    ax.hist(
        exp_samples,
        bins=50,
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        label="Generated",
    )
    x = np.linspace(0, np.max(exp_samples), 100)
    ax.plot(
        x,
        stats.expon(scale=1 / lambda_param).pdf(x),
        "r-",
        linewidth=2,
        label="Theoretical",
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(f"Exponential(lambda={lambda_param})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Example 2: Triangular distribution
    # F^(-1)(u) = a + sqrt(u*(b-a)*(c-a)) if u < (c-a)/(b-a)
    ax = axes[0, 1]
    a, b, c = 0, 10, 3  # min, max, mode
    tri_samples = np.zeros(n)
    for i, u in enumerate(uniform_samples):
        if u < (c - a) / (b - a):
            tri_samples[i] = a + np.sqrt(u * (b - a) * (c - a))
        else:
            tri_samples[i] = b - np.sqrt((1 - u) * (b - a) * (b - c))

    ax.hist(
        tri_samples,
        bins=50,
        density=True,
        alpha=0.7,
        color="lightgreen",
        edgecolor="black",
        label="Generated",
    )
    x = np.linspace(a, b, 100)
    ax.plot(
        x,
        stats.triang(c=(c - a) / (b - a), loc=a, scale=b - a).pdf(x),
        "r-",
        linewidth=2,
        label="Theoretical",
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(f"Triangular({a}, {b}, mode={c})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Example 3: Pareto distribution
    ax = axes[0, 2]
    alpha = 2.0
    xm = 1.0
    pareto_samples = xm / (uniform_samples ** (1 / alpha))
    pareto_samples = pareto_samples[pareto_samples < 10]  # Clip for visualization

    ax.hist(
        pareto_samples,
        bins=50,
        density=True,
        alpha=0.7,
        color="salmon",
        edgecolor="black",
        label="Generated",
    )
    x = np.linspace(xm, 10, 100)
    ax.plot(
        x, stats.pareto(alpha, scale=xm).pdf(x), "r-", linewidth=2, label="Theoretical"
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(f"Pareto(alpha={alpha}, xm={xm})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom row: Show the transformation process for exponential
    ax = axes[1, 0]
    # Show uniform samples
    ax.hist(
        uniform_samples,
        bins=50,
        density=True,
        alpha=0.7,
        color="gray",
        edgecolor="black",
    )
    ax.axhline(1, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("U ~ Uniform(0,1)")
    ax.set_ylabel("Density")
    ax.set_title("Step 1: Generate Uniform")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    # Show transformation
    u_plot = np.linspace(0.01, 0.99, 100)
    x_plot = -np.log(1 - u_plot) / lambda_param
    ax.plot(u_plot, x_plot, "b-", linewidth=2)
    ax.set_xlabel("U (Uniform)")
    ax.set_ylabel("X = -ln(1-U)/lambda")
    ax.set_title("Step 2: Apply Inverse CDF")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    # Show result
    ax.hist(
        exp_samples,
        bins=50,
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    x = np.linspace(0, np.max(exp_samples), 100)
    ax.plot(x, stats.expon(scale=1 / lambda_param).pdf(x), "r-", linewidth=2)
    ax.set_xlabel("X ~ Exponential")
    ax.set_ylabel("Density")
    ax.set_title("Step 3: Result")
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Inverse Transform Method: U(0,1) -> Other Distributions",
        fontsize=14,
        weight="bold",
    )
    plt.tight_layout()
    plt.savefig("inverse_transform.png", dpi=150)
    print("Saved: inverse_transform.png")
    plt.show()


def box_muller_transform():
    """
    Box-Muller Transform: Convert two U(0,1) to two independent N(0,1)

    Given U1, U2 ~ Uniform(0,1):
    Z1 = sqrt(-2*ln(U1)) * cos(2*pi*U2)
    Z2 = sqrt(-2*ln(U1)) * sin(2*pi*U2)
    Then Z1, Z2 ~ N(0,1) independently
    """
    print("\n" + "=" * 60)
    print("BOX-MULLER TRANSFORM")
    print("=" * 60)
    print("\nTransforms pairs of uniform to pairs of standard normal")
    print("Z1 = sqrt(-2*ln(U1)) * cos(2*pi*U2)")
    print("Z2 = sqrt(-2*ln(U1)) * sin(2*pi*U2)")

    n = 5000
    u1 = np.random.uniform(0, 1, n)
    u2 = np.random.uniform(0, 1, n)

    # Box-Muller transformation
    r = np.sqrt(-2 * np.log(u1))
    theta = 2 * np.pi * u2
    z1 = r * np.cos(theta)
    z2 = r * np.sin(theta)

    fig = plt.figure(figsize=(15, 10))

    # Original uniform samples
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(u1[:500], u2[:500], s=10, alpha=0.5, c="blue")
    ax1.set_xlabel("U1 ~ Uniform(0,1)")
    ax1.set_ylabel("U2 ~ Uniform(0,1)")
    ax1.set_title("Input: Two Uniform Variables")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Polar transformation visualization
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(
        r[:500] * np.cos(theta[:500]),
        r[:500] * np.sin(theta[:500]),
        s=10,
        alpha=0.5,
        c="green",
    )
    ax2.set_xlabel("r * cos(theta)")
    ax2.set_ylabel("r * sin(theta)")
    ax2.set_title("Intermediate: Polar Coordinates")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    # Result: Bivariate normal
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(z1[:500], z2[:500], s=10, alpha=0.5, c="red")
    ax3.set_xlabel("Z1 ~ N(0,1)")
    ax3.set_ylabel("Z2 ~ N(0,1)")
    ax3.set_title("Output: Two Normal Variables")
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect("equal")

    # Distribution of Z1
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(
        z1,
        bins=50,
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        label="Z1 generated",
    )
    x = np.linspace(-4, 4, 100)
    ax4.plot(x, stats.norm(0, 1).pdf(x), "r-", linewidth=2, label="N(0,1) theoretical")
    ax4.set_xlabel("Z1")
    ax4.set_ylabel("Density")
    ax4.set_title("Distribution of Z1")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Distribution of Z2
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(
        z2,
        bins=50,
        density=True,
        alpha=0.7,
        color="lightgreen",
        edgecolor="black",
        label="Z2 generated",
    )
    ax5.plot(x, stats.norm(0, 1).pdf(x), "r-", linewidth=2, label="N(0,1) theoretical")
    ax5.set_xlabel("Z2")
    ax5.set_ylabel("Density")
    ax5.set_title("Distribution of Z2")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Q-Q plot to verify normality
    ax6 = plt.subplot(2, 3, 6)
    stats.probplot(z1, dist="norm", plot=ax6)
    ax6.set_title("Q-Q Plot: Checking Normality of Z1")
    ax6.grid(True, alpha=0.3)

    plt.suptitle("Box-Muller Transform: Uniform -> Normal", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig("box_muller.png", dpi=150)
    print("Saved: box_muller.png")
    plt.show()

    print(f"\nGenerated {n} pairs of normal random variables")
    print(f"Z1: mean={np.mean(z1):.4f}, std={np.std(z1):.4f}")
    print(f"Z2: mean={np.mean(z2):.4f}, std={np.std(z2):.4f}")
    print(f"Correlation: {np.corrcoef(z1, z2)[0, 1]:.4f} (should be ~0)")


def acceptance_rejection_sampling():
    """
    Acceptance-Rejection Sampling: For complex distributions

    Method:
    1. Generate X from proposal distribution g(x)
    2. Generate U ~ Uniform(0, 1)
    3. Accept X if U <= f(X) / (M * g(X)), where M is chosen so M*g(x) >= f(x)
    """
    print("\n" + "=" * 60)
    print("ACCEPTANCE-REJECTION SAMPLING")
    print("=" * 60)
    print("\nUseful when inverse CDF is difficult to compute")
    print("Example: Beta(2, 5) distribution using Uniform proposal")

    # Target: Beta(2, 5)
    target_dist = stats.beta(2, 5)

    # Proposal: Uniform(0, 1) - easy to sample from
    # Find M such that M * g(x) >= f(x) for all x
    # For Beta(2,5) with Uniform proposal, M = max(Beta PDF)
    x_grid = np.linspace(0, 1, 1000)
    M = np.max(target_dist.pdf(x_grid))

    # Sampling process
    n_desired = 5000
    samples = []
    n_proposals = 0
    n_accepted = 0

    proposal_x = []
    proposal_y = []
    accepted_x = []
    accepted_y = []
    rejected_x = []
    rejected_y = []

    while len(samples) < n_desired:
        # Generate from proposal
        x = np.random.uniform(0, 1)
        u = np.random.uniform(0, 1)
        n_proposals += 1

        # Acceptance criterion
        if u <= target_dist.pdf(x) / M:
            samples.append(x)
            n_accepted += 1
            if len(samples) <= 500:
                accepted_x.append(x)
                accepted_y.append(u * M)
        else:
            if len(rejected_x) < 500:
                rejected_x.append(x)
                rejected_y.append(u * M)

    samples = np.array(samples)
    acceptance_rate = n_accepted / n_proposals

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Visualization of acceptance/rejection
    ax = axes[0, 0]
    x = np.linspace(0, 1, 1000)
    ax.plot(x, target_dist.pdf(x), "b-", linewidth=2, label="Target: Beta(2,5)")
    ax.fill_between(
        x, 0, M, alpha=0.2, color="gray", label=f"Proposal bound (M={M:.2f})"
    )
    ax.scatter(accepted_x, accepted_y, s=1, c="green", alpha=0.5, label="Accepted")
    ax.scatter(rejected_x, rejected_y, s=1, c="red", alpha=0.5, label="Rejected")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title("Acceptance-Rejection Process")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Resulting distribution
    ax = axes[0, 1]
    ax.hist(
        samples,
        bins=50,
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        label="Generated samples",
    )
    x = np.linspace(0, 1, 100)
    ax.plot(x, target_dist.pdf(x), "r-", linewidth=2, label="Target Beta(2,5)")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title("Resulting Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Cumulative comparison
    ax = axes[1, 0]
    sorted_samples = np.sort(samples)
    empirical_cdf = np.linspace(0, 1, len(sorted_samples))
    ax.plot(sorted_samples, empirical_cdf, "b-", linewidth=2, label="Empirical CDF")
    x = np.linspace(0, 1, 100)
    ax.plot(x, target_dist.cdf(x), "r--", linewidth=2, label="Theoretical CDF")
    ax.set_xlabel("x")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Statistics
    ax = axes[1, 1]
    ax.axis("off")

    stats_text = "ACCEPTANCE-REJECTION STATS\n"
    stats_text += "=" * 35 + "\n\n"
    stats_text += f"Target: Beta(2, 5)\n"
    stats_text += f"Proposal: Uniform(0, 1)\n\n"
    stats_text += f"Samples desired: {n_desired}\n"
    stats_text += f"Proposals made: {n_proposals}\n"
    stats_text += f"Acceptance rate: {acceptance_rate:.1%}\n\n"
    stats_text += f"M (bound): {M:.3f}\n\n"
    stats_text += "SAMPLE STATISTICS\n"
    stats_text += "-" * 35 + "\n"
    stats_text += f"Sample mean: {np.mean(samples):.4f}\n"
    stats_text += f"True mean: {target_dist.mean():.4f}\n\n"
    stats_text += f"Sample std: {np.std(samples):.4f}\n"
    stats_text += f"True std: {target_dist.std():.4f}\n\n"
    stats_text += "EFFICIENCY\n"
    stats_text += "-" * 35 + "\n"
    stats_text += f"Lower M -> higher efficiency\n"
    stats_text += f"Choose proposal close to target"

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
        "Acceptance-Rejection Sampling: Generating Beta(2,5)",
        fontsize=14,
        weight="bold",
    )
    plt.tight_layout()
    plt.savefig("acceptance_rejection.png", dpi=150)
    print("Saved: acceptance_rejection.png")
    plt.show()

    print(f"\nGenerated {n_desired} samples using {n_proposals} proposals")
    print(f"Acceptance rate: {acceptance_rate:.1%}")


def compare_generation_methods():
    """
    Compare different methods for generating the same distribution.
    """
    print("\n" + "=" * 60)
    print("COMPARING GENERATION METHODS")
    print("=" * 60)

    n = 10000

    # Target: Standard Normal N(0,1)
    # Method 1: Box-Muller
    u1 = np.random.uniform(0, 1, n // 2)
    u2 = np.random.uniform(0, 1, n // 2)
    r = np.sqrt(-2 * np.log(u1))
    theta = 2 * np.pi * u2
    normal_box_muller = np.concatenate([r * np.cos(theta), r * np.sin(theta)])

    # Method 2: NumPy built-in (uses ziggurat algorithm)
    normal_numpy = np.random.normal(0, 1, n)

    # Method 3: Central Limit Theorem approximation
    # Sum of 12 Uniform(0,1) - 6 ~ N(0,1) approximately
    normal_clt = np.sum(np.random.uniform(0, 1, (n, 12)), axis=1) - 6

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.linspace(-4, 4, 100)
    theoretical_pdf = stats.norm(0, 1).pdf(x)

    datasets = [
        (normal_box_muller, "Box-Muller Transform", axes[0, 0]),
        (normal_numpy, "NumPy (Ziggurat)", axes[0, 1]),
        (normal_clt, "CLT Approximation (12 uniforms)", axes[1, 0]),
    ]

    for data, title, ax in datasets:
        ax.hist(
            data,
            bins=50,
            density=True,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
            label="Generated",
        )
        ax.plot(x, theoretical_pdf, "r-", linewidth=2, label="N(0,1)")

        mean_val = np.mean(data)
        std_val = np.std(data)

        stats_text = f"mean: {mean_val:.4f}\nstd: {std_val:.4f}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Comparison table
    ax = axes[1, 1]
    ax.axis("off")

    stats_text = "METHOD COMPARISON\n"
    stats_text += "=" * 40 + "\n\n"
    stats_text += "BOX-MULLER\n"
    stats_text += "  + Exact (theoretically)\n"
    stats_text += "  + Produces pairs\n"
    stats_text += "  - Slow (trig functions)\n\n"
    stats_text += "ZIGGURAT (NumPy default)\n"
    stats_text += "  + Very fast\n"
    stats_text += "  + Exact\n"
    stats_text += "  + Industry standard\n\n"
    stats_text += "CLT APPROXIMATION\n"
    stats_text += "  + Educational value\n"
    stats_text += "  + Simple to understand\n"
    stats_text += "  - Only approximate\n"
    stats_text += "  - Slower than ziggurat\n\n"
    stats_text += "RECOMMENDATION\n"
    stats_text += "-" * 40 + "\n"
    stats_text += "Use library functions (np.random)\n"
    stats_text += "unless you need special control\n"
    stats_text += "or are implementing your own RNG"

    ax.text(
        0.1,
        0.9,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        fontsize=9,
    )

    plt.suptitle(
        "Comparing Methods to Generate Normal Distribution", fontsize=14, weight="bold"
    )
    plt.tight_layout()
    plt.savefig("generation_methods_comparison.png", dpi=150)
    print("Saved: generation_methods_comparison.png")
    plt.show()


if __name__ == "__main__":
    print("TRANSFORMATIONS: USING UNIFORM TO GENERATE OTHER DISTRIBUTIONS")
    print("=" * 60)

    print("\n1. Inverse Transform Method...")
    inverse_transform_method()

    print("\n2. Box-Muller Transform...")
    box_muller_transform()

    print("\n3. Acceptance-Rejection Sampling...")
    acceptance_rejection_sampling()

    print("\n4. Comparing generation methods...")
    compare_generation_methods()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("1. ALL random number generation starts with Uniform(0,1)")
    print("2. Inverse transform: Most general, needs CDF^(-1)")
    print("3. Box-Muller: Elegant way to get normal from uniform")
    print("4. Accept-Reject: Works when inverse CDF is hard")
    print("5. In practice, use library functions (highly optimized)")
    print("=" * 60)
