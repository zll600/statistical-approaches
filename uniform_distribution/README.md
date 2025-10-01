# Uniform Distribution Learning Package

Learn about uniform distributions through interactive Python modules with visualizations and real-world examples.

## Overview

This package contains three core modules that teach uniform distribution concepts:

1. **basics.py** - Uniform Distribution Fundamentals
2. **applications.py** - Real-World Applications
3. **transformations.py** - Using Uniform to Generate Other Distributions

## Quick Start

### Interactive Menu (Recommended)

```bash
# From project root
python src/uniform_distribution/main.py
```

### Run Individual Modules

```bash
python src/uniform_distribution/basics.py
python src/uniform_distribution/applications.py
python src/uniform_distribution/transformations.py
```

## Module Descriptions

### 1. basics.py - Fundamentals

Learn core concepts:
- Continuous uniform U(a,b) and discrete uniform
- PDF and CDF visualization
- Properties: mean, variance, standard deviation
- Effect of range parameters
- Generating random samples

**Outputs:** 5 PNG visualizations

**Run it:**
```bash
python src/uniform_distribution/basics.py
```

---

### 2. applications.py - Real-World Applications

Practical applications:
- Monte Carlo estimation of pi
- Monte Carlo integration
- Waiting time problems
- Foundation of random number generation
- Customer service simulations

**Outputs:** 5 PNG visualizations

**Run it:**
```bash
python src/uniform_distribution/applications.py
```

---

### 3. transformations.py - Distribution Generation

Advanced techniques:
- Inverse transform method
- Box-Muller transform (uniform to normal)
- Acceptance-rejection sampling
- Why uniform is fundamental to all RNG
- Comparing generation methods

**Outputs:** 4 PNG visualizations

**Run it:**
```bash
python src/uniform_distribution/transformations.py
```

---

## Learning Path

### For Beginners
1. Start with `basics.py` to understand fundamental concepts
2. Explore `applications.py` to see practical uses
3. Study `transformations.py` to understand RNG foundations

### For Intermediate Learners
1. Review `basics.py` as refresher
2. Deep dive into `transformations.py`
3. Experiment with `applications.py` examples

## Key Concepts Covered

### Distribution Properties
- Flat probability density (constant PDF)
- Linear CDF
- Mean = (a+b)/2
- Variance = (b-a)^2/12
- All values equally likely

### Practical Applications
- Random number generation
- Monte Carlo methods
- Simulation modeling
- Uncertainty quantification
- Fair selection processes

### Theoretical Foundations
- Inverse CDF method
- Box-Muller transform
- Acceptance-rejection sampling
- Relationship to other distributions

## Key Formulas

**Continuous Uniform U(a, b):**
```
PDF: f(x) = 1/(b-a) for a <= x <= b
CDF: F(x) = (x-a)/(b-a)
Mean: (a+b)/2
Variance: (b-a)^2/12
Std Dev: (b-a)/sqrt(12)
```

**Discrete Uniform {a, a+1, ..., b}:**
```
PMF: P(X=x) = 1/(b-a+1)
Mean: (a+b)/2
Variance: [(b-a+1)^2 - 1]/12
```

## Requirements

- Python 3.13+
- numpy
- scipy
- matplotlib

All dependencies are in `pyproject.toml`.

## Tips for Learning

1. Run code yourself - don't just read!
2. Modify parameters to see effects
3. Save generated plots for reference
4. Experiment with examples
5. Compare uniform with normal distribution

## Learning Outcomes

After completing these modules, you will:

- Understand uniform distribution properties
- Calculate probabilities for uniform distributions
- Apply uniform distributions to real problems
- Understand how ALL RNG starts with uniform
- Implement transformation methods
- Use Monte Carlo methods effectively

## Additional Notes

### Why Uniform Matters

Uniform distribution is fundamental because:
- It's the simplest probability distribution
- Represents "complete uncertainty" or "maximum entropy"
- Foundation of ALL random number generation
- Used extensively in simulations
- Models fair/unbiased selection

### Connection to Other Distributions

Uniform is used to generate:
- Normal (via Box-Muller)
- Exponential (via inverse transform)
- Any distribution (via inverse CDF or accept-reject)

---

**Happy Learning!**

Start exploring:
```bash
python src/uniform_distribution/main.py
```
