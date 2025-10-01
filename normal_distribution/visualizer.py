"""
Interactive Normal Distribution Visualizer

An interactive GUI application that lets you explore normal distributions
by adjusting parameters (μ and σ) with sliders.

Features:
- Real-time visualization of PDF and CDF
- Interactive sliders for mean and standard deviation
- Display key statistics and probabilities
- Shaded areas showing different standard deviations
- Sample data generation and display

Run this file directly to launch the interactive visualizer:
    python visualizer.py
"""

import sys
import numpy as np
from scipy import stats
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                QHBoxLayout, QLabel, QSlider, QPushButton,
                                QGroupBox, QGridLayout, QTextEdit)
from PySide6.QtCore import Qt
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class NormalDistributionVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Normal Distribution Visualizer")
        self.setGeometry(100, 100, 1400, 800)

        # Initialize parameters
        self.mu = 0.0
        self.sigma = 1.0

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left panel: Controls
        left_panel = self.create_control_panel()
        layout.addWidget(left_panel, stretch=1)

        # Right panel: Plots
        right_panel = self.create_plot_panel()
        layout.addWidget(right_panel, stretch=3)

        # Initial plot
        self.update_plots()

    def create_control_panel(self):
        """Create the control panel with sliders and buttons."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Normal Distribution Explorer")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Mean (μ) control
        mu_group = QGroupBox("Mean (μ)")
        mu_layout = QVBoxLayout()

        self.mu_label = QLabel(f"μ = {self.mu:.2f}")
        self.mu_label.setAlignment(Qt.AlignCenter)
        self.mu_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        mu_layout.addWidget(self.mu_label)

        self.mu_slider = QSlider(Qt.Horizontal)
        self.mu_slider.setMinimum(-100)
        self.mu_slider.setMaximum(100)
        self.mu_slider.setValue(0)
        self.mu_slider.setTickPosition(QSlider.TicksBelow)
        self.mu_slider.setTickInterval(10)
        self.mu_slider.valueChanged.connect(self.update_mu)
        mu_layout.addWidget(self.mu_slider)

        mu_range_label = QLabel("Range: -10.0 to 10.0")
        mu_range_label.setAlignment(Qt.AlignCenter)
        mu_layout.addWidget(mu_range_label)

        mu_group.setLayout(mu_layout)
        layout.addWidget(mu_group)

        # Standard Deviation (σ) control
        sigma_group = QGroupBox("Standard Deviation (σ)")
        sigma_layout = QVBoxLayout()

        self.sigma_label = QLabel(f"σ = {self.sigma:.2f}")
        self.sigma_label.setAlignment(Qt.AlignCenter)
        self.sigma_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        sigma_layout.addWidget(self.sigma_label)

        self.sigma_slider = QSlider(Qt.Horizontal)
        self.sigma_slider.setMinimum(10)  # 0.1
        self.sigma_slider.setMaximum(100)  # 10.0
        self.sigma_slider.setValue(10)  # 1.0
        self.sigma_slider.setTickPosition(QSlider.TicksBelow)
        self.sigma_slider.setTickInterval(10)
        self.sigma_slider.valueChanged.connect(self.update_sigma)
        sigma_layout.addWidget(self.sigma_slider)

        sigma_range_label = QLabel("Range: 0.1 to 10.0")
        sigma_range_label.setAlignment(Qt.AlignCenter)
        sigma_layout.addWidget(sigma_range_label)

        sigma_group.setLayout(sigma_layout)
        layout.addWidget(sigma_group)

        # Statistics display
        stats_group = QGroupBox("Statistics & Probabilities")
        stats_layout = QVBoxLayout()

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(300)
        stats_layout.addWidget(self.stats_text)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Preset buttons
        presets_group = QGroupBox("Presets")
        presets_layout = QGridLayout()

        presets = [
            ("Standard Normal", 0, 1),
            ("Wide Distribution", 0, 3),
            ("Narrow Distribution", 0, 0.5),
            ("Shifted Right", 5, 1),
            ("Shifted Left", -5, 1),
            ("IQ Scores", 100, 15)
        ]

        row, col = 0, 0
        for name, mu, sigma in presets:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, m=mu, s=sigma: self.set_preset(m, s))
            presets_layout.addWidget(btn, row, col)
            col += 1
            if col > 1:
                col = 0
                row += 1

        presets_group.setLayout(presets_layout)
        layout.addWidget(presets_group)

        # Sample generation button
        sample_btn = QPushButton("Generate Random Samples")
        sample_btn.clicked.connect(self.generate_samples)
        sample_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        layout.addWidget(sample_btn)

        layout.addStretch()

        return panel

    def create_plot_panel(self):
        """Create the plotting panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        return panel

    def update_mu(self, value):
        """Update mean from slider."""
        self.mu = value / 10.0  # Scale to -10.0 to 10.0
        self.mu_label.setText(f"μ = {self.mu:.2f}")
        self.update_plots()

    def update_sigma(self, value):
        """Update standard deviation from slider."""
        self.sigma = value / 10.0  # Scale to 0.1 to 10.0
        self.sigma_label.setText(f"σ = {self.sigma:.2f}")
        self.update_plots()

    def set_preset(self, mu, sigma):
        """Set preset values for mu and sigma."""
        # Update sliders (which will trigger update_plots)
        self.mu_slider.setValue(int(mu * 10))
        self.sigma_slider.setValue(int(sigma * 10))

    def update_plots(self):
        """Update all plots with current parameters."""
        self.figure.clear()

        # Create distribution
        dist = stats.norm(self.mu, self.sigma)

        # Generate x values
        x = np.linspace(self.mu - 4*self.sigma, self.mu + 4*self.sigma, 1000)
        pdf = dist.pdf(x)
        cdf = dist.cdf(x)

        # Create subplots
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax2 = self.figure.add_subplot(2, 1, 2)

        # Plot 1: PDF with shaded regions
        ax1.plot(x, pdf, 'b-', linewidth=2, label='PDF')
        ax1.fill_between(x, pdf, alpha=0.2, color='blue')

        # Shade 68-95-99.7 regions
        # 68% (±1σ)
        x_68 = x[(x >= self.mu - self.sigma) & (x <= self.mu + self.sigma)]
        pdf_68 = pdf[(x >= self.mu - self.sigma) & (x <= self.mu + self.sigma)]
        ax1.fill_between(x_68, pdf_68, alpha=0.4, color='yellow', label='68% (±1σ)')

        # 95% (±2σ)
        x_95_low = x[(x >= self.mu - 2*self.sigma) & (x < self.mu - self.sigma)]
        pdf_95_low = pdf[(x >= self.mu - 2*self.sigma) & (x < self.mu - self.sigma)]
        x_95_high = x[(x > self.mu + self.sigma) & (x <= self.mu + 2*self.sigma)]
        pdf_95_high = pdf[(x > self.mu + self.sigma) & (x <= self.mu + 2*self.sigma)]
        ax1.fill_between(x_95_low, pdf_95_low, alpha=0.4, color='orange', label='95% (±2σ)')
        ax1.fill_between(x_95_high, pdf_95_high, alpha=0.4, color='orange')

        # Mark mean and standard deviations
        ax1.axvline(self.mu, color='red', linestyle='--', linewidth=2, label=f'μ={self.mu:.2f}')
        for i in [-2, -1, 1, 2]:
            ax1.axvline(self.mu + i*self.sigma, color='gray', linestyle=':', alpha=0.5)

        ax1.set_xlabel('x')
        ax1.set_ylabel('Probability Density')
        ax1.set_title(f'Normal Distribution PDF: N(μ={self.mu:.2f}, σ={self.sigma:.2f})')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Plot 2: CDF
        ax2.plot(x, cdf, 'g-', linewidth=2, label='CDF')
        ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50%')
        ax2.axhline(0.68, color='orange', linestyle='--', alpha=0.5, label='68%')
        ax2.axhline(0.95, color='purple', linestyle='--', alpha=0.5, label='95%')
        ax2.axvline(self.mu, color='red', linestyle='--', linewidth=2, label=f'μ={self.mu:.2f}')

        ax2.set_xlabel('x')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution Function (CDF)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

        # Update statistics
        self.update_statistics(dist)

    def update_statistics(self, dist):
        """Update the statistics text display."""
        # Calculate various statistics
        mean = self.mu
        median = self.mu
        mode = self.mu
        variance = self.sigma ** 2
        std_dev = self.sigma

        # Calculate probabilities
        prob_1sigma = dist.cdf(self.mu + self.sigma) - dist.cdf(self.mu - self.sigma)
        prob_2sigma = dist.cdf(self.mu + 2*self.sigma) - dist.cdf(self.mu - 2*self.sigma)
        prob_3sigma = dist.cdf(self.mu + 3*self.sigma) - dist.cdf(self.mu - 3*self.sigma)

        # Percentiles
        p25 = dist.ppf(0.25)
        p50 = dist.ppf(0.50)
        p75 = dist.ppf(0.75)
        p90 = dist.ppf(0.90)
        p95 = dist.ppf(0.95)
        p99 = dist.ppf(0.99)

        text = f"""
<h3>Distribution Parameters</h3>
<b>Mean (μ):</b> {mean:.4f}<br>
<b>Median:</b> {median:.4f}<br>
<b>Mode:</b> {mode:.4f}<br>
<b>Variance (σ²):</b> {variance:.4f}<br>
<b>Std Dev (σ):</b> {std_dev:.4f}<br>

<h3>Empirical Rule (68-95-99.7)</h3>
<b>P(μ - σ < X < μ + σ):</b> {prob_1sigma*100:.2f}%<br>
<b>Range:</b> [{mean - std_dev:.2f}, {mean + std_dev:.2f}]<br>
<br>
<b>P(μ - 2σ < X < μ + 2σ):</b> {prob_2sigma*100:.2f}%<br>
<b>Range:</b> [{mean - 2*std_dev:.2f}, {mean + 2*std_dev:.2f}]<br>
<br>
<b>P(μ - 3σ < X < μ + 3σ):</b> {prob_3sigma*100:.2f}%<br>
<b>Range:</b> [{mean - 3*std_dev:.2f}, {mean + 3*std_dev:.2f}]<br>

<h3>Percentiles</h3>
<b>25th:</b> {p25:.4f}<br>
<b>50th (Median):</b> {p50:.4f}<br>
<b>75th:</b> {p75:.4f}<br>
<b>90th:</b> {p90:.4f}<br>
<b>95th:</b> {p95:.4f}<br>
<b>99th:</b> {p99:.4f}<br>
        """

        self.stats_text.setHtml(text)

    def generate_samples(self):
        """Generate and visualize random samples from the current distribution."""
        # Generate samples
        n_samples = 1000
        samples = np.random.normal(self.mu, self.sigma, n_samples)

        # Create new figure for samples
        sample_figure = Figure(figsize=(10, 6))
        ax = sample_figure.add_subplot(1, 1, 1)

        # Histogram of samples
        ax.hist(samples, bins=50, density=True, alpha=0.7, color='skyblue',
                edgecolor='black', label=f'Samples (n={n_samples})')

        # Overlay theoretical PDF
        x = np.linspace(samples.min(), samples.max(), 100)
        ax.plot(x, stats.norm(self.mu, self.sigma).pdf(x), 'r-',
                linewidth=2, label='Theoretical PDF')

        # Add statistics
        sample_mean = np.mean(samples)
        sample_std = np.std(samples, ddof=1)

        stats_text = f'Sample μ = {sample_mean:.2f}\nSample σ = {sample_std:.2f}\n'
        stats_text += f'Theoretical μ = {self.mu:.2f}\nTheoretical σ = {self.sigma:.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title(f'Random Samples from N(μ={self.mu:.2f}, σ={self.sigma:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Replace main canvas temporarily
        old_canvas = self.canvas
        sample_canvas = FigureCanvas(sample_figure)

        # Find the canvas in the layout and replace it
        layout = self.centralWidget().layout()
        right_panel = layout.itemAt(1).widget()
        right_layout = right_panel.layout()

        right_layout.removeWidget(old_canvas)
        right_layout.addWidget(sample_canvas)

        self.canvas = sample_canvas
        self.figure = sample_figure
        self.canvas.draw()

        # Add button to return to main view
        return_btn = QPushButton("Return to Distribution View")
        return_btn.clicked.connect(lambda: self.return_to_main_view(old_canvas, sample_canvas, return_btn))
        return_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        right_layout.addWidget(return_btn)

    def return_to_main_view(self, old_canvas, sample_canvas, button):
        """Return from sample view to main distribution view."""
        layout = self.centralWidget().layout()
        right_panel = layout.itemAt(1).widget()
        right_layout = right_panel.layout()

        right_layout.removeWidget(sample_canvas)
        right_layout.removeWidget(button)
        right_layout.addWidget(old_canvas)

        sample_canvas.deleteLater()
        button.deleteLater()

        self.canvas = old_canvas
        # Recreate figure for old canvas
        self.figure = Figure(figsize=(10, 8))
        self.canvas.figure = self.figure
        self.update_plots()


def main():
    """Launch the interactive visualizer."""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    visualizer = NormalDistributionVisualizer()
    visualizer.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    print("="*60)
    print("LAUNCHING INTERACTIVE NORMAL DISTRIBUTION VISUALIZER")
    print("="*60)
    print("\nFeatures:")
    print("  - Adjust μ (mean) and σ (standard deviation) with sliders")
    print("  - Real-time PDF and CDF visualization")
    print("  - View probabilities and percentiles")
    print("  - Try preset distributions")
    print("  - Generate and visualize random samples")
    print("\nEnjoy exploring normal distributions!")
    print("="*60)

    main()
