"""
Simple runner for the bankruptcy prediction project without TensorFlow.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def run_simple_pipeline():
    """Run a simplified version of the pipeline."""

    print("🚀 Starting Simple Bankruptcy Prediction Pipeline...")

    try:
        # Import required modules
        from src.config import Config, DEFAULT_MODEL_CONFIGS
        from src.data import DataProcessor
        from src.models import (
            LogisticRegressionModel,
            RandomForestModel,
            ModelEvaluator,
            ModelFactory,
        )

        # Load configuration
        config = Config()
        config.feature_selection = True
        config.max_features = 30  # Reduce for faster processing

        print("✅ Configuration loaded")

        # Initialize data processor
        processor = DataProcessor(config)

        # Load and process data
        print("📊 Loading and processing data...")
        X_train, X_test, y_train, y_test, exploration_stats = (
            processor.process_full_pipeline()
        )

        print(f"✅ Data processed successfully!")
        print(f"   - Training samples: {X_train.shape[0]}")
        print(f"   - Test samples: {X_test.shape[0]}")
        print(f"   - Features: {X_train.shape[1]}")
        print(f"   - Target distribution: {exploration_stats['target_distribution']}")

        # Train models
        print("\n🤖 Training models...")
        results = []
        models = {}

        # Train Logistic Regression models
        for model_config in DEFAULT_MODEL_CONFIGS["logistic_regression"]:
            print(f"   Training {model_config.name}...")

            model = ModelFactory.create_model("logistic_regression", model_config)
            model.fit(X_train, y_train)
            models[model_config.name] = model

            # Evaluate
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            result = ModelEvaluator.evaluate_model(
                y_test, y_pred, y_pred_proba, model_config.name
            )
            results.append(result)

            print(
                f"     ✅ {model_config.name} - F1: {result['f1_score']:.4f}, Accuracy: {result['accuracy']:.4f}"
            )

        # Train Random Forest models
        for model_config in DEFAULT_MODEL_CONFIGS["random_forest"]:
            print(f"   Training {model_config.name}...")

            model = ModelFactory.create_model("random_forest", model_config)
            model.fit(X_train, y_train)
            models[model_config.name] = model

            # Evaluate
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            result = ModelEvaluator.evaluate_model(
                y_test, y_pred, y_pred_proba, model_config.name
            )
            results.append(result)

            print(
                f"     ✅ {model_config.name} - F1: {result['f1_score']:.4f}, Accuracy: {result['accuracy']:.4f}"
            )

        # Find best model
        best_result = max(results, key=lambda x: x["f1_score"])

        print("\n" + "=" * 60)
        print("📊 RESULTS SUMMARY")
        print("=" * 60)

        print("\nModel Performance:")
        print("-" * 40)
        for result in sorted(results, key=lambda x: x["f1_score"], reverse=True):
            print(
                f"{result['model_name']:<20} | F1: {result['f1_score']:.4f} | Acc: {result['accuracy']:.4f} | AUC: {result['roc_auc']:.4f}"
            )

        print(f"\n🏆 BEST MODEL: {best_result['model_name']}")
        print(f"   F1-Score: {best_result['f1_score']:.4f}")
        print(f"   Accuracy: {best_result['accuracy']:.4f}")
        print(f"   ROC-AUC:  {best_result['roc_auc']:.4f}")

        # Feature importance for Random Forest
        print("\n🔍 Feature Importance Analysis:")
        for model_name, model in models.items():
            if (
                hasattr(model, "get_feature_importance")
                and model.get_feature_importance() is not None
            ):
                importance = model.get_feature_importance()
                top_indices = np.argsort(importance)[-10:][::-1]  # Top 10 features

                print(f"\n   Top 10 features for {model_name}:")
                feature_names = processor.feature_names or [
                    f"Feature_{i}" for i in range(len(importance))
                ]
                for i, idx in enumerate(top_indices):
                    if idx < len(feature_names):
                        print(
                            f"     {i+1}. {feature_names[idx]}: {importance[idx]:.4f}"
                        )

        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print("\n💡 Next Steps:")
        print(
            "   • Run the full optimized notebook: jupyter notebook bankruptcy_prediction_optimized.ipynb"
        )
        print("   • Start the API server: python -m uvicorn src.api:app --reload")
        print("   • Install TensorFlow for neural networks: pip install tensorflow")

        return True

    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_simple_pipeline()
    exit(0 if success else 1)
