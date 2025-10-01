# Statistical Approaches

An interactive, code-based learning platform for statistics and probability concepts. Learn by doing with comprehensive Python modules featuring visualizations, examples, and hands-on exercises.

## Overview

This repository contains educational modules designed to help you understand statistical concepts through interactive code and visualizations. Each module is self-contained and can be run independently.

### Current Topics

#### Normal Distributions (Complete)
Comprehensive learning package covering:
- Normal distribution fundamentals (PDF, CDF, empirical rule)
- Central Limit Theorem demonstrations
- Z-scores and standardization
- Testing for normality and data analysis
- Interactive GUI visualizer

Location: `src/normal_distribution/`

#### Uniform Distributions (Complete)
Comprehensive learning package covering:
- Uniform distribution fundamentals (continuous and discrete)
- Real-world applications and Monte Carlo methods
- Using uniform to generate other distributions
- Foundation of random number generation

Location: `src/uniform_distribution/`

## Quick Start

### Prerequisites

- Python 3.13 or higher
- UV package manager (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/statistical-approaches.git
cd statistical-approaches

# Install dependencies
uv sync
```

### Running the Modules

Each distribution package has an interactive menu:

```bash
# Normal distribution modules
python src/normal_distribution/main.py

# Uniform distribution modules
python src/uniform_distribution/main.py
```

Or run individual modules directly:

```bash
# Normal distribution
python src/normal_distribution/basics.py
python src/normal_distribution/central_limit_theorem.py

# Uniform distribution
python src/uniform_distribution/basics.py
python src/uniform_distribution/applications.py
```

## Project Structure

```
statistical-approaches/
├── README.md
├── pyproject.toml
├── src/
│   ├── probabilities.py
│   ├── normal_distribution/
│   │   ├── main.py
│   │   ├── basics.py
│   │   ├── central_limit_theorem.py
│   │   ├── standard_normal.py
│   │   ├── analysis.py
│   │   ├── visualizer.py
│   │   └── README.md
│   └── uniform_distribution/
│       ├── main.py
│       ├── basics.py
│       ├── applications.py
│       ├── transformations.py
│       └── README.md
└── uv.lock
```

## Module Overview

### Normal Distribution Package

**basics.py** - Fundamentals: PDF, CDF, 68-95-99.7 rule
**central_limit_theorem.py** - CLT demonstrations
**standard_normal.py** - Z-scores and standardization
**analysis.py** - Normality tests and Q-Q plots
**visualizer.py** - Interactive GUI

See `src/normal_distribution/README.md` for details.

### Uniform Distribution Package

**basics.py** - Continuous and discrete uniform fundamentals
**applications.py** - Monte Carlo methods and simulations
**transformations.py** - Generating other distributions from uniform

See `src/uniform_distribution/README.md` for details.

## Key Features

- **Educational Focus**: Extensive comments explaining concepts
- **Interactive Learning**: GUI visualizers and modifiable code
- **Comprehensive Coverage**: Theory and practical applications
- **High-Quality Visualizations**: Professional plots saved as PNG files
- **Real-World Examples**: Practical applications of concepts
- **Progressive Difficulty**: Build understanding step-by-step

## Dependencies

All managed through `pyproject.toml`:

- numpy (>=2.3.2): Numerical computing
- scipy (>=1.16.1): Statistical functions
- matplotlib (>=3.10.5): Plotting and visualization
- PySide6 (>=6.9.1): GUI framework
- arviz (>=0.22.0): Advanced statistical visualization

## Learning Outcomes

### Normal Distribution
- Understand normal distribution properties
- Calculate probabilities and z-scores
- Apply Central Limit Theorem
- Test for normality
- Interpret statistical results

### Uniform Distribution
- Understand uniform distribution properties
- Apply Monte Carlo methods
- Generate other distributions from uniform
- Understand foundations of RNG
- Model complete uncertainty

## Future Topics

Planned additions:
- Exponential distribution
- Binomial and Poisson distributions
- Hypothesis testing
- Confidence intervals
- Regression analysis
- Bayesian statistics

## Technical Notes

- Python 3.13+ required
- UV package manager recommended
- Uses matplotlib for all visualizations
- High-resolution outputs (150 DPI)
- No unicode characters in code (ASCII only)

## Troubleshooting

### Plot Windows Don't Show
- Run from terminal, not IDE
- Check matplotlib backend configuration

### Import Errors
- Run from project root directory
- Ensure dependencies installed: `uv sync`

## Contributing

Contributions welcome:
- Additional statistical topics
- More examples and exercises
- Improved visualizations
- Documentation enhancements

## License

This project is for educational purposes. Feel free to use, modify, and share.

---

**Start Learning:**

```bash
# Normal distributions
python src/normal_distribution/main.py

# Uniform distributions
python src/uniform_distribution/main.py
```
