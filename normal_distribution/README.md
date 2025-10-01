# Normal Distribution Learning Package

A comprehensive, hands-on guide to learning about normal distributions and probability through code and visualizations.

## 📚 Overview

This package contains five interactive modules that build your understanding from basic concepts to advanced analysis:

1. **basics.py** - Normal Distribution Fundamentals
2. **central_limit_theorem.py** - The Central Limit Theorem
3. **standard_normal.py** - Z-scores and Standardization
4. **analysis.py** - Testing for Normality
5. **visualizer.py** - Interactive GUI Explorer

## 🚀 Quick Start

### Interactive Menu (Recommended)

The easiest way to explore all modules:

```bash
# From the project root directory
python src/normal_distribution/main.py
```

This presents an interactive menu to select which module to run.

### Run Individual Modules

```bash
# From the project root directory
python src/normal_distribution/basics.py
python src/normal_distribution/central_limit_theorem.py
python src/normal_distribution/standard_normal.py
python src/normal_distribution/analysis.py
python src/normal_distribution/visualizer.py
```

Each module can be run independently to focus on specific concepts.

## 📖 Module Descriptions

### 1. basics.py - Normal Distribution Fundamentals

Learn the core concepts of normal distributions:
- What is a normal distribution? (bell curve)
- Parameters: μ (mean) and σ (standard deviation)
- PDF (Probability Density Function) and CDF (Cumulative Distribution Function)
- The 68-95-99.7 Rule (Empirical Rule)
- Calculating probabilities
- Generating random samples

**Key Visualizations:**
- PDF with different parameters
- CDF comparison
- Empirical rule demonstration
- Random sample generation

**Run it:**
```bash
python src/normal_distribution/basics.py
```

---

### 2. central_limit_theorem.py - The Central Limit Theorem

Understand one of the most important theorems in statistics:
- What is the Central Limit Theorem (CLT)?
- How sample means approximate normal distribution
- CLT works for ANY original distribution
- Standard Error: SE = σ / √n
- Effect of sample size on convergence
- Real-world applications

**Key Visualizations:**
- Dice rolling experiments
- CLT with different source distributions
- Sample size effects
- Standard error demonstration

**Run it:**
```bash
python src/normal_distribution/central_limit_theorem.py
```

---

### 3. standard_normal.py - Z-scores and Standardization

Master the standard normal distribution and z-scores:
- Standard normal distribution: N(0, 1)
- Converting any normal to standard normal
- Z-score formula: z = (x - μ) / σ
- Interpreting z-scores
- Using z-scores to compare different distributions
- Calculating probabilities with z-scores
- Finding values from percentiles

**Key Visualizations:**
- Standard normal distribution
- Standardization transformation
- Probability calculations with z-scores
- Z-score table

**Run it:**
```bash
python src/normal_distribution/standard_normal.py
```

---

### 4. analysis.py - Testing for Normality

Learn how to determine if data is normally distributed:
- Q-Q plots (Quantile-Quantile plots)
- Statistical tests: Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov
- Understanding test results and p-values
- Sample size effects on normality tests
- Data transformations (log, square root, Box-Cox)
- Real-world examples

**Key Visualizations:**
- Q-Q plots for various distributions
- Comparison of normal vs. non-normal data
- Effect of sample size on tests
- Transformation demonstrations

**Run it:**
```bash
python src/normal_distribution/analysis.py
```

---

### 5. visualizer.py - Interactive GUI Explorer

Explore normal distributions interactively:
- Interactive sliders to adjust μ and σ
- Real-time PDF and CDF visualization
- Shaded areas showing 68-95-99.7 rule
- Statistics and probabilities display
- Preset distributions (Standard Normal, IQ scores, etc.)
- Generate and visualize random samples

**Features:**
- Adjust mean (μ) from -10 to 10
- Adjust standard deviation (σ) from 0.1 to 10
- See changes in real-time
- Compare theoretical vs. sample distributions

**Run it:**
```bash
python src/normal_distribution/visualizer.py
```

## 🎯 Learning Path

### For Beginners
1. Start with `basics.py` to understand fundamental concepts
2. Explore `visualizer.py` to see how parameters affect the distribution
3. Move to `standard_normal.py` to learn about z-scores
4. Study `central_limit_theorem.py` to see why normal distribution is so important

### For Intermediate Learners
1. Review `basics.py` and `standard_normal.py` as refresher
2. Deep dive into `central_limit_theorem.py`
3. Learn practical skills with `analysis.py`
4. Use `visualizer.py` to reinforce concepts

## 📊 Key Concepts Covered

### Probability Distributions
- PDF (Probability Density Function)
- CDF (Cumulative Distribution Function)
- Parameters (mean, standard deviation)
- Percentiles and quantiles

### Normal Distribution Properties
- Bell-shaped curve
- Symmetric around the mean
- 68-95-99.7 Rule
- Area under curve = 1

### Statistical Theory
- Central Limit Theorem
- Standard Error
- Sampling distributions
- Law of Large Numbers

### Practical Skills
- Calculating probabilities
- Finding percentiles
- Testing for normality
- Data transformations
- Comparing distributions

## 🛠️ Requirements

All dependencies are listed in `pyproject.toml`:
- numpy: Numerical computing
- scipy: Statistical functions
- matplotlib: Plotting and visualization
- PySide6: GUI application (for visualizer.py)
- arviz: Advanced visualization (optional, used in some examples)

## 💡 Tips for Learning

1. **Run the code yourself** - Don't just read, execute each module and examine the outputs
2. **Modify parameters** - Change values in the code to see how results differ
3. **Save the plots** - All modules save plots as PNG files for later reference
4. **Use the visualizer** - The interactive tool helps build intuition
5. **Read the comments** - Each file has extensive educational comments
6. **Try the exercises** - Modify the examples to test your understanding

## 🎓 Learning Outcomes

After completing these modules, you will be able to:
- Explain what a normal distribution is and its properties
- Calculate probabilities using normal distributions
- Understand and apply z-scores
- Explain the Central Limit Theorem and its importance
- Test whether data follows a normal distribution
- Transform non-normal data to approximate normality
- Use normal distributions to solve real-world problems

## 📝 Additional Resources

### Key Formulas

**Normal Distribution PDF:**
```
f(x) = (1 / (σ√(2π))) * e^(-((x-μ)²) / (2σ²))
```

**Z-Score (Standardization):**
```
z = (x - μ) / σ
```

**Standard Error:**
```
SE = σ / √n
```

**Reverse Z-Score:**
```
x = μ + z × σ
```

### Important Values
- z = ±1.96 → 95% confidence interval
- z = ±2.58 → 99% confidence interval
- z = 0 → mean/median/mode (for normal distribution)

## 🤝 Contributing

Feel free to extend these modules with:
- Additional examples
- New visualizations
- More real-world datasets
- Interactive exercises
- Jupyter notebook versions

## 📧 Questions?

These modules are designed for self-study. Work through them at your own pace, and don't hesitate to modify the code to test your understanding!

---

**Happy Learning! 📊🎓**
