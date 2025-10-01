"""
Standard Normal Distribution and Z-Scores

The Standard Normal Distribution is a special normal distribution with:
- Mean (μ) = 0
- Standard Deviation (σ) = 1
- Notation: Z ~ N(0, 1)

Why is it important?
- It's a common reference point for all normal distributions
- Z-scores allow us to compare values from different normal distributions
- Statistical tables and many software tools use the standard normal

Z-Score (Standardization):
The z-score tells you how many standard deviations a value is from the mean.

Formula: z = (x - μ) / σ

Where:
- x = the value you're examining
- μ = population mean
- σ = population standard deviation

Interpretation:
- z = 0: value is exactly at the mean
- z = 1: value is 1 standard deviation above the mean
- z = -1: value is 1 standard deviation below the mean
- z = 2.5: value is 2.5 standard deviations above the mean
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def visualize_standard_normal():
    """
    Visualize the standard normal distribution and key properties.
    """
    z = np.linspace(-4, 4, 1000)
    standard_normal = stats.norm(0, 1)
    pdf = standard_normal.pdf(z)

    plt.figure(figsize=(12, 6))

    # Plot PDF
    plt.plot(z, pdf, 'b-', linewidth=2, label='Standard Normal PDF')
    plt.fill_between(z, pdf, alpha=0.3)

    # Mark important z-scores
    important_z = [-3, -2, -1, 0, 1, 2, 3]
    for zi in important_z:
        plt.axvline(zi, color='red', linestyle='--', alpha=0.5, linewidth=1)
        plt.text(zi, -0.01, f'z={zi}', ha='center', fontsize=9)

    # Mark mean
    plt.axvline(0, color='green', linestyle='-', linewidth=2, label='Mean (μ=0)')

    # Annotate key areas
    # 68% area (±1σ)
    z_68 = z[(z >= -1) & (z <= 1)]
    pdf_68 = pdf[(z >= -1) & (z <= 1)]
    plt.fill_between(z_68, pdf_68, alpha=0.5, color='yellow', label='68% (±1σ)')

    plt.xlabel('Z-score (standard deviations from mean)')
    plt.ylabel('Probability Density')
    plt.title('Standard Normal Distribution: Z ~ N(0, 1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("standard_normal.png", dpi=150)
    print("Saved: standard_normal.png")
    plt.show()


def demonstrate_standardization():
    """
    Show how any normal distribution can be converted to standard normal.
    """
    # Original distribution: heights ~ N(170, 10)
    mu, sigma = 170, 10
    original_dist = stats.norm(mu, sigma)

    # Create data
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    original_pdf = original_dist.pdf(x)

    # Standardize: z = (x - μ) / σ
    z = (x - mu) / sigma
    standard_pdf = stats.norm(0, 1).pdf(z)

    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Original distribution
    ax = axes[0]
    ax.plot(x, original_pdf, 'b-', linewidth=2)
    ax.fill_between(x, original_pdf, alpha=0.3)
    ax.axvline(mu, color='green', linestyle='--', linewidth=2, label=f'μ={mu}')
    ax.axvline(mu - sigma, color='red', linestyle='--', alpha=0.5, label=f'μ-σ={mu-sigma}')
    ax.axvline(mu + sigma, color='red', linestyle='--', alpha=0.5, label=f'μ+σ={mu+sigma}')
    ax.set_xlabel('Height (cm)')
    ax.set_ylabel('Density')
    ax.set_title(f'Original: N(μ={mu}, σ={sigma})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Standardized distribution
    ax = axes[1]
    ax.plot(z, standard_pdf, 'r-', linewidth=2)
    ax.fill_between(z, standard_pdf, alpha=0.3, color='red')
    ax.axvline(0, color='green', linestyle='--', linewidth=2, label='μ=0')
    ax.axvline(-1, color='orange', linestyle='--', alpha=0.5, label='z=-1')
    ax.axvline(1, color='orange', linestyle='--', alpha=0.5, label='z=1')
    ax.set_xlabel('Z-score')
    ax.set_ylabel('Density')
    ax.set_title('Standardized: N(μ=0, σ=1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Standardization Transforms Any Normal to Standard Normal', fontsize=14)
    plt.tight_layout()
    plt.savefig("standardization_demo.png", dpi=150)
    print("Saved: standardization_demo.png")
    plt.show()


def zscore_examples():
    """
    Practical examples of calculating and interpreting z-scores.
    """
    print("\n" + "="*60)
    print("Z-SCORE EXAMPLES")
    print("="*60)

    # Example 1: Test scores
    print("\nExample 1: Test Scores")
    print("-" * 60)
    mu_test, sigma_test = 75, 10
    print(f"Test scores ~ N(μ={mu_test}, σ={sigma_test})")

    scores = [65, 75, 85, 95, 60]
    print(f"\nStudent scores and their z-scores:")
    for score in scores:
        z = (score - mu_test) / sigma_test
        print(f"  Score {score}: z = {z:+.2f}", end="")

        if z == 0:
            print(" (exactly average)")
        elif z > 0:
            print(f" ({abs(z):.1f} std devs above average)")
        else:
            print(f" ({abs(z):.1f} std devs below average)")

    # Example 2: Comparing apples to oranges
    print("\n\nExample 2: Comparing Different Distributions")
    print("-" * 60)
    print("Alice's math score: 85 in a class with μ=75, σ=10")
    print("Bob's history score: 90 in a class with μ=80, σ=5")
    print("\nWho performed better relative to their class?")

    z_alice = (85 - 75) / 10
    z_bob = (90 - 80) / 5

    print(f"\nAlice's z-score: {z_alice:+.2f}")
    print(f"Bob's z-score: {z_bob:+.2f}")

    if z_alice > z_bob:
        print("\nAlice performed better relative to her class!")
    else:
        print("\nBob performed better relative to his class!")

    print("(Z-scores let us compare performance across different scales)")

    # Example 3: Percentiles from z-scores
    print("\n\nExample 3: Z-scores to Percentiles")
    print("-" * 60)
    z_scores = [-2, -1, 0, 1, 2]
    print("Z-score → Percentile:")
    for z in z_scores:
        percentile = stats.norm(0, 1).cdf(z) * 100
        print(f"  z = {z:+.0f}: {percentile:.1f}th percentile")


def probability_with_zscores():
    """
    Using z-scores to calculate probabilities.
    """
    print("\n" + "="*60)
    print("PROBABILITY CALCULATIONS USING Z-SCORES")
    print("="*60)

    # IQ scores: μ=100, σ=15
    mu, sigma = 100, 15
    print(f"\nIQ scores ~ N(μ={mu}, σ={sigma})")
    print("-" * 60)

    # Question 1: P(IQ > 130)?
    x1 = 130
    z1 = (x1 - mu) / sigma
    prob1 = 1 - stats.norm(0, 1).cdf(z1)
    print(f"\n1. What's the probability of IQ > {x1}?")
    print(f"   Convert to z-score: z = ({x1} - {mu}) / {sigma} = {z1:.2f}")
    print(f"   P(Z > {z1:.2f}) = {prob1:.4f} = {prob1*100:.2f}%")

    # Question 2: P(85 < IQ < 115)?
    x2_low, x2_high = 85, 115
    z2_low = (x2_low - mu) / sigma
    z2_high = (x2_high - mu) / sigma
    prob2 = stats.norm(0, 1).cdf(z2_high) - stats.norm(0, 1).cdf(z2_low)
    print(f"\n2. What's the probability of {x2_low} < IQ < {x2_high}?")
    print(f"   Convert to z-scores:")
    print(f"     z_low = ({x2_low} - {mu}) / {sigma} = {z2_low:.2f}")
    print(f"     z_high = ({x2_high} - {mu}) / {sigma} = {z2_high:.2f}")
    print(f"   P({z2_low:.2f} < Z < {z2_high:.2f}) = {prob2:.4f} = {prob2*100:.2f}%")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Original scale
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    pdf = stats.norm(mu, sigma).pdf(x)
    ax = axes[0]
    ax.plot(x, pdf, 'b-', linewidth=2)
    ax.fill_between(x, pdf, where=(x > x1), alpha=0.5, color='red',
                     label=f'P(IQ > {x1}) = {prob1:.3f}')
    ax.axvline(x1, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('IQ Score')
    ax.set_ylabel('Density')
    ax.set_title('Original Scale')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Z-score scale
    z = np.linspace(-4, 4, 1000)
    pdf_z = stats.norm(0, 1).pdf(z)
    ax = axes[1]
    ax.plot(z, pdf_z, 'r-', linewidth=2)
    ax.fill_between(z, pdf_z, where=(z > z1), alpha=0.5, color='red',
                     label=f'P(Z > {z1:.2f}) = {prob1:.3f}')
    ax.axvline(z1, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Z-score')
    ax.set_ylabel('Density')
    ax.set_title('Standardized (Z-score) Scale')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Same Probability, Different Scales', fontsize=14)
    plt.tight_layout()
    plt.savefig("probability_with_z.png", dpi=150)
    print("\nSaved: probability_with_z.png")
    plt.show()


def inverse_zscore_problems():
    """
    Given a percentile or probability, find the corresponding value.
    """
    print("\n" + "="*60)
    print("INVERSE PROBLEMS: From Percentile to Value")
    print("="*60)

    mu, sigma = 100, 15
    print(f"\nIQ scores ~ N(μ={mu}, σ={sigma})")
    print("-" * 60)

    # Question 1: What IQ is at the 90th percentile?
    percentile = 90
    z_90 = stats.norm(0, 1).ppf(0.90)  # ppf = inverse CDF
    iq_90 = mu + z_90 * sigma
    print(f"\n1. What IQ score is at the {percentile}th percentile?")
    print(f"   Step 1: Find z-score for {percentile}th percentile: z = {z_90:.2f}")
    print(f"   Step 2: Convert to IQ: x = μ + z×σ = {mu} + {z_90:.2f}×{sigma} = {iq_90:.2f}")

    # Question 2: What scores bound the middle 95%?
    print(f"\n2. What IQ range contains the middle 95% of people?")
    z_lower = stats.norm(0, 1).ppf(0.025)  # 2.5th percentile
    z_upper = stats.norm(0, 1).ppf(0.975)  # 97.5th percentile
    iq_lower = mu + z_lower * sigma
    iq_upper = mu + z_upper * sigma
    print(f"   Step 1: Find z-scores for 2.5th and 97.5th percentiles:")
    print(f"     z_lower = {z_lower:.2f}, z_upper = {z_upper:.2f}")
    print(f"   Step 2: Convert to IQ:")
    print(f"     IQ_lower = {iq_lower:.2f}, IQ_upper = {iq_upper:.2f}")
    print(f"   Answer: Middle 95% is between {iq_lower:.2f} and {iq_upper:.2f}")

    # Visualize
    plt.figure(figsize=(12, 6))
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    pdf = stats.norm(mu, sigma).pdf(x)

    plt.plot(x, pdf, 'b-', linewidth=2)
    plt.fill_between(x, pdf, where=(x >= iq_lower) & (x <= iq_upper),
                     alpha=0.5, color='green', label='Middle 95%')

    plt.axvline(iq_90, color='red', linestyle='--', linewidth=2,
                label=f'90th percentile = {iq_90:.1f}')
    plt.axvline(iq_lower, color='orange', linestyle='--', linewidth=1.5)
    plt.axvline(iq_upper, color='orange', linestyle='--', linewidth=1.5)

    plt.xlabel('IQ Score')
    plt.ylabel('Density')
    plt.title('Finding Values from Percentiles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("inverse_z_problems.png", dpi=150)
    print("\nSaved: inverse_z_problems.png")
    plt.show()


def zscore_table_visualization():
    """
    Create a visual z-score table (like traditional statistical tables).
    """
    print("\n" + "="*60)
    print("Z-SCORE TABLE (Cumulative Probabilities)")
    print("="*60)

    # Create z-scores from -3.0 to 3.0
    z_values = np.arange(-3.0, 3.1, 0.5)
    probabilities = stats.norm(0, 1).cdf(z_values)

    print("\nZ-score | Cumulative Prob | Percentile | Interpretation")
    print("-" * 60)
    for z, prob in zip(z_values, probabilities):
        print(f" {z:+5.1f}  |     {prob:.4f}      |   {prob*100:5.1f}%   | "
              f"P(Z ≤ {z:+.1f})")

    # Visual representation
    fig, ax = plt.subplots(figsize=(12, 8))

    # Hide axes
    ax.axis('tight')
    ax.axis('off')

    # Create table data
    table_data = []
    table_data.append(['Z-score', 'P(Z ≤ z)', 'Percentile', 'Visual'])

    for z, prob in zip(z_values, probabilities):
        # Create mini sparkline
        visual = '█' * int(prob * 20) + '░' * (20 - int(prob * 20))
        table_data.append([f'{z:+.1f}', f'{prob:.4f}', f'{prob*100:.1f}%', visual])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.15, 0.15, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title('Standard Normal Distribution: Cumulative Probability Table',
              fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("z_table.png", dpi=150, bbox_inches='tight')
    print("\nSaved: z_table.png")
    plt.show()


if __name__ == "__main__":
    print("STANDARD NORMAL DISTRIBUTION AND Z-SCORES")
    print("=" * 60)

    print("\n1. Visualizing the standard normal distribution...")
    visualize_standard_normal()

    print("\n2. Demonstrating standardization...")
    demonstrate_standardization()

    print("\n3. Z-score examples...")
    zscore_examples()

    print("\n4. Probability calculations with z-scores...")
    probability_with_zscores()

    print("\n5. Inverse problems (percentile to value)...")
    inverse_zscore_problems()

    print("\n6. Z-score table...")
    zscore_table_visualization()

    print("\n" + "="*60)
    print("KEY FORMULAS:")
    print("="*60)
    print("1. Standardization: z = (x - μ) / σ")
    print("2. Reverse: x = μ + z × σ")
    print("3. Percentile to z: z = norm.ppf(percentile/100)")
    print("4. Z to percentile: percentile = norm.cdf(z) × 100")
    print("="*60)
