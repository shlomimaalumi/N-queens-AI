import argparse
from naive_algorithm import NaiveAlgorithm
from MIN_CONFLICTS_algorithm import MinConflictsAlgorithm
from mutation_rate_fit import MutationRateFit
from Genetic_algorithm import GeneticAlgorithm


def run_naive_algorithm():
    print("Running Naive Algorithm...")
    algo = NaiveAlgorithm()
    algo.run()
    print("Naive Algorithm finished.")


def run_min_conflicts_algorithm():
    print("Running Min Conflicts Algorithm...")
    algo = MinConflictsAlgorithm()
    algo.run()
    print("Min Conflicts Algorithm finished.")


def run_genetic_algorithm():
    print("Running Genetic Algorithm...")
    algo = GeneticAlgorithm()
    algo.run()
    print("Genetic Algorithm finished.")


def run_mutation_rate_fit():
    print("Running Mutation Rate Fit...")
    mutation_rate_fitter = MutationRateFit()
    mutation_rate_fitter.run()
    print("Mutation Rate Fitting finished.")


def run_predict_best_mutation_rate():
    print("Running Adaptive Mutation Rate...")
    mutation_rate_fitter = MutationRateFit()
    best_rate = mutation_rate_fitter.predict_best_mutation_rate()
    print(f"Best Mutation Rate Predicted: {best_rate}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run different algorithms.")

    parser.add_argument(
        "--naive", action="store_true", help="Run Naive Algorithm."
    )
    parser.add_argument(
        "--min-conflicts", action="store_true", help="Run Min Conflicts Algorithm."
    )
    parser.add_argument(
        "--genetic", action="store_true", help="Run Genetic Algorithm."
    )
    parser.add_argument(
        "--mutation-fit", action="store_true", help="Run Mutation Rate Fit."
    )
    parser.add_argument(
        "--adaptive-mutation", action="store_true", help="Run Adaptive Mutation Rate Prediction."
    )

    args = parser.parse_args()

    if args.naive:
        run_naive_algorithm()
    elif args.min_conflicts:
        run_min_conflicts_algorithm()
    elif args.genetic:
        run_genetic_algorithm()
    elif args.mutation_fit:
        run_mutation_rate_fit()
    elif args.adaptive_mutation:
        run_predict_best_mutation_rate()
    else:
        print("Please specify an operation to run. Use --help for more information.")
