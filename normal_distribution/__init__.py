"""
Normal Distribution Learning Package

A comprehensive set of modules to learn about normal distributions
through interactive code and visualizations.

Modules:
- basics: Core concepts of normal distributions (PDF, CDF, empirical rule)
- central_limit_theorem: Demonstrations of the Central Limit Theorem
- standard_normal: Z-scores and standardization
- analysis: Testing for normality, Q-Q plots, and data transformations
- visualizer: Interactive GUI application (PySide6)

Usage:
    # Run individual modules
    python -m src.normal_distribution.basics
    python -m src.normal_distribution.central_limit_theorem
    python -m src.normal_distribution.standard_normal
    python -m src.normal_distribution.analysis
    python -m src.normal_distribution.visualizer

    # Or import in your own code
    from src.normal_distribution import basics
"""

__version__ = "0.1.0"
__author__ = "Statistical Approaches"

# Make it easy to run modules
__all__ = [
    "basics",
    "central_limit_theorem",
    "standard_normal",
    "analysis",
    "visualizer",
]
