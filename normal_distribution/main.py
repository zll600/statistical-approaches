"""
Statistical Approaches - Main Menu

A collection of interactive learning modules for statistics and probability.
Currently includes comprehensive normal distribution tutorials.
"""

import sys
import subprocess
import os


def print_banner():
    print("\n" + "="*70)
    print(" " * 15 + "STATISTICAL APPROACHES")
    print(" " * 10 + "Interactive Learning Modules")
    print("="*70)


def print_menu():
    print("\nNORMAL DISTRIBUTION LEARNING MODULES")
    print("-" * 70)
    print("1. Basics - Normal Distribution Fundamentals")
    print("   - PDF, CDF, and the 68-95-99.7 rule")
    print("   - Probability calculations and random sampling")
    print()
    print("2. Central Limit Theorem")
    print("   - Demonstrations with dice rolls and various distributions")
    print("   - Understanding standard error and sample size effects")
    print()
    print("3. Standard Normal & Z-Scores")
    print("   - Standardization and z-score calculations")
    print("   - Percentiles and probability lookups")
    print()
    print("4. Analysis & Testing for Normality")
    print("   - Q-Q plots and statistical tests")
    print("   - Data transformations and real-world examples")
    print()
    print("5. Interactive Visualizer (GUI)")
    print("   - Real-time exploration with interactive sliders")
    print("   - Generate samples and see statistics")
    print()
    print("6. Run ALL modules sequentially")
    print()
    print("0. Exit")
    print("-" * 70)


def run_module(module_name):
    """Run a specific module."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(script_dir, f"{module_name}.py")

    print(f"\n{'='*70}")
    print(f"Running: {module_name}")
    print(f"{'='*70}\n")

    try:
        result = subprocess.run(
            [sys.executable, module_path],
            check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Error running module: {e}")
        return False
    except FileNotFoundError:
        print(f"\n[ERROR] Module not found: {module_path}")
        return False


def run_all_modules():
    """Run all modules in sequence."""
    modules = [
        ("basics", "Normal Distribution Basics"),
        ("central_limit_theorem", "Central Limit Theorem"),
        ("standard_normal", "Standard Normal & Z-Scores"),
        ("analysis", "Analysis & Testing for Normality"),
    ]

    print("\n" + "="*70)
    print("RUNNING ALL MODULES SEQUENTIALLY")
    print("="*70)
    print("\nNote: Close each plot window to proceed to the next module.")
    print("Press Enter to continue...")
    input()

    for module_name, display_name in modules:
        print(f"\n{'='*70}")
        print(f"Module: {display_name}")
        print(f"{'='*70}")
        if not run_module(module_name):
            print(f"\n[WARNING] Stopped at {display_name} due to error.")
            return

    print("\n" + "="*70)
    print("[OK] All modules completed successfully!")
    print("="*70)


def main():
    while True:
        print_banner()
        print_menu()

        try:
            choice = input("Select an option (0-6): ").strip()

            if choice == "0":
                print("\nGoodbye! Happy learning!")
                break

            elif choice == "1":
                run_module("basics")

            elif choice == "2":
                run_module("central_limit_theorem")

            elif choice == "3":
                run_module("standard_normal")

            elif choice == "4":
                run_module("analysis")

            elif choice == "5":
                run_module("visualizer")

            elif choice == "6":
                run_all_modules()

            else:
                print("\n[ERROR] Invalid choice. Please select 0-6.")

            if choice != "0":
                print("\nPress Enter to return to menu...")
                input()

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}")
            print("\nPress Enter to continue...")
            input()


if __name__ == "__main__":
    main()
