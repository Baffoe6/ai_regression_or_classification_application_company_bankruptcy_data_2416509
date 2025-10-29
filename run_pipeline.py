"""
Main entry point for running the optimized bankruptcy prediction pipeline.
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.config import Config, get_default_config
from src.pipeline import BankruptcyPredictor
from src.optimization import ModelOptimizer
from src.utils import setup_logging


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Optimized Bankruptcy Prediction Pipeline"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="CompanyBankruptcyData.csv",
        help="Path to the data file",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--optimize", action="store_true", help="Enable advanced optimization"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["logistic_regression", "random_forest", "neural_network"],
        default=["logistic_regression", "random_forest", "neural_network"],
        help="Models to train",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Output directory for results"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = Config.from_json(args.config)
    else:
        config = get_default_config()

    # Override with command line arguments
    config.data_path = args.data
    config.output_dir = args.output_dir

    if args.verbose:
        config.log_level = "DEBUG"

    # Initialize predictor
    predictor = BankruptcyPredictor(config)

    try:
        # Run basic pipeline
        print("🚀 Starting optimized bankruptcy prediction pipeline...")
        report = predictor.run_full_pipeline(args.data)

        # Run advanced optimization if requested
        if args.optimize:
            print("\n🔧 Running advanced optimization...")

            optimizer = ModelOptimizer(config)

            # Optimize models
            optimized_models = []
            for model_type in args.models:
                if (
                    model_type != "neural_network"
                ):  # Skip NN for command line for simplicity
                    print(f"Optimizing {model_type}...")
                    result = optimizer.optimize_single_model(
                        model_type=model_type,
                        X_train=predictor.X_train,
                        y_train=predictor.y_train,
                        X_test=predictor.X_test,
                        optimize_features=True,
                        optimize_hyperparams=True,
                    )
                    optimized_models.append(result)

            # Create ensemble
            if len(optimized_models) > 1:
                print("Creating ensemble models...")
                ensemble_results = optimizer.create_optimized_ensemble(
                    optimized_models=optimized_models,
                    X_train=predictor.X_train,
                    y_train=predictor.y_train,
                )

                # Evaluate ensemble
                from sklearn.metrics import accuracy_score, f1_score

                voting_ensemble = ensemble_results["voting_ensemble"]
                y_pred = voting_ensemble.predict(predictor.X_test)

                accuracy = accuracy_score(predictor.y_test, y_pred)
                f1 = f1_score(predictor.y_test, y_pred)

                print(f"Ensemble Performance:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  F1-Score: {f1:.4f}")

        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\nResults saved to: {config.output_dir}")
        print(f"Report:\n{report}")

        return 0

    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
