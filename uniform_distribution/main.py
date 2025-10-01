"""
Uniform Distribution Learning - Main Menu

Interactive menu to explore uniform distribution concepts.
"""

import sys
import subprocess
import os


def print_banner():
    print("\n" + "=" * 70)
    print(" " * 12 + "UNIFORM DISTRIBUTION LEARNING")
    print(" " * 15 + "Interactive Modules")
    print("=" * 70)


def print_menu():
    print("\nUNIFORM DISTRIBUTION MODULES")
    print("-" * 70)
    print("1. Basics - Uniform Distribution Fundamentals")
    print("   - Continuous and discrete uniform")
    print("   - PDF, CDF, and properties")
    print()
    print("2. Applications - Real-World Use Cases")
    print("   - Monte Carlo simulations")
    print("   - Waiting times and random processes")
    print()
    print("3. Transformations - Generating Other Distributions")
    print("   - Inverse transform method")
    print("   - Box-Muller transform")
    print("   - Acceptance-rejection sampling")
    print()
    print("4. Run ALL modules sequentially")
    print()
    print("0. Exit")
    print("-" * 70)


def run_module(module_name):
    """Run a specific module."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(script_dir, f"{module_name}.py")

    print(f"\n{'=' * 70}")
    print(f"Running: {module_name}")
    print(f"{'=' * 70}\n")

    try:
        result = subprocess.run([sys.executable, module_path], check=True)
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
        ("basics", "Uniform Distribution Basics"),
        ("applications", "Real-World Applications"),
        ("transformations", "Distribution Transformations"),
    ]

    print("\n" + "=" * 70)
    print("RUNNING ALL MODULES SEQUENTIALLY")
    print("=" * 70)
    print("\nNote: Close each plot window to proceed to the next module.")
    print("Press Enter to continue...")
    input()

    for module_name, display_name in modules:
        print(f"\n{'=' * 70}")
        print(f"Module: {display_name}")
        print(f"{'=' * 70}")
        if not run_module(module_name):
            print(f"\n[WARNING] Stopped at {display_name} due to error.")
            return

    print("\n" + "=" * 70)
    print("[OK] All modules completed successfully!")
    print("=" * 70)


def main():
    while True:
        print_banner()
        print_menu()

        try:
            choice = input("Select an option (0-4): ").strip()

            if choice == "0":
                print("\nGoodbye! Happy learning!")
                break

            elif choice == "1":
                run_module("basics")

            elif choice == "2":
                run_module("applications")

            elif choice == "3":
                run_module("transformations")

            elif choice == "4":
                run_all_modules()

            else:
                print("\n[ERROR] Invalid choice. Please select 0-4.")

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
