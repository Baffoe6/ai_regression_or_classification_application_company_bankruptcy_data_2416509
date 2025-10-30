"""
Unit tests for the bankruptcy prediction modules.
"""

import os
import shutil
# Import modules to test
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.append(".")
from src.config import Config, get_default_config
from src.data import DataProcessor
from src.models import ModelEvaluator, ModelFactory
from src.pipeline import BankruptcyPredictor


class TestConfig(unittest.TestCase):
    """Test configuration management."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = get_default_config()

        self.assertIsInstance(config, Config)
        self.assertEqual(config.target_column, "Bankrupt?")
        self.assertEqual(config.test_size, 0.2)
        self.assertEqual(config.random_state, 42)

    def test_config_serialization(self):
        """Test configuration serialization to/from JSON."""
        config = Config()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config.to_json(f.name)

            # Load config back
            loaded_config = Config.from_json(f.name)

            self.assertEqual(config.target_column, loaded_config.target_column)
            self.assertEqual(config.test_size, loaded_config.test_size)

            # Cleanup
            os.unlink(f.name)


class TestDataProcessor(unittest.TestCase):
    """Test data processing functionality."""

    def setUp(self):
        """Set up test data."""
        self.config = Config()
        self.config.data_path = "test_data.csv"

        # Create sample test data
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        data = {f"X{i+1}": np.random.randn(n_samples) for i in range(n_features)}
        data["Bankrupt?"] = np.random.binomial(1, 0.2, n_samples)

        self.test_df = pd.DataFrame(data)
        self.processor = DataProcessor(self.config)

    def test_data_exploration(self):
        """Test data exploration functionality."""
        stats = self.processor.explore_data(self.test_df)

        self.assertIn("shape", stats)
        self.assertIn("target_distribution", stats)
        self.assertIn("feature_count", stats)
        self.assertEqual(stats["shape"], self.test_df.shape)
        self.assertEqual(stats["feature_count"], 10)

    def test_data_preprocessing(self):
        """Test data preprocessing."""
        X, y = self.processor.preprocess_data(self.test_df)

        self.assertEqual(X.shape[0], len(self.test_df))
        self.assertEqual(len(y), len(self.test_df))
        self.assertEqual(X.shape[1], 10)  # Number of features

    def test_data_splitting(self):
        """Test data splitting."""
        X = np.random.randn(100, 10)
        y = np.random.binomial(1, 0.2, 100)

        X_train, X_test, y_train, y_test = self.processor.split_data(X, y)

        self.assertEqual(len(X_train), 80)  # 80% for training
        self.assertEqual(len(X_test), 20)  # 20% for testing
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)


class TestModelFactory(unittest.TestCase):
    """Test model factory and model creation."""

    def setUp(self):
        """Set up test configuration."""
        from src.config import ModelConfig

        self.config = ModelConfig("test_model", {"C": 1.0})

    def test_logistic_regression_creation(self):
        """Test logistic regression model creation."""
        model = ModelFactory.create_model("logistic_regression", self.config)

        self.assertIsNotNone(model)
        self.assertEqual(model.name, "test_model")

    def test_random_forest_creation(self):
        """Test random forest model creation."""
        model = ModelFactory.create_model("random_forest", self.config)

        self.assertIsNotNone(model)
        self.assertEqual(model.name, "test_model")

    def test_neural_network_creation(self):
        """Test neural network model creation."""
        model = ModelFactory.create_model("neural_network", self.config, input_dim=10)

        self.assertIsNotNone(model)
        self.assertEqual(model.name, "test_model")

    def test_invalid_model_type(self):
        """Test invalid model type raises error."""
        with self.assertRaises(ValueError):
            ModelFactory.create_model("invalid_model", self.config)


class TestModelEvaluator(unittest.TestCase):
    """Test model evaluation functionality."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.y_true = np.random.binomial(1, 0.3, 100)
        self.y_pred = np.random.binomial(1, 0.3, 100)
        self.y_pred_proba = np.column_stack(
            [1 - np.random.beta(2, 5, 100), np.random.beta(2, 5, 100)]
        )

        self.evaluator = ModelEvaluator()

    def test_model_evaluation(self):
        """Test model evaluation metrics."""
        metrics = self.evaluator.evaluate_model(
            self.y_true, self.y_pred, self.y_pred_proba, "test_model"
        )

        required_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertGreaterEqual(metrics[metric], 0)
            self.assertLessEqual(metrics[metric], 1)


class TestBankruptcyPredictor(unittest.TestCase):
    """Test the main prediction pipeline."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

        config = Config()
        config.output_dir = self.temp_dir
        config.save_models = False  # Don't save models in tests
        config.save_plots = False  # Don't save plots in tests

        self.predictor = BankruptcyPredictor(config)

        # Create mock data
        self._create_mock_data()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def _create_mock_data(self):
        """Create mock data for testing."""
        np.random.seed(42)
        n_samples = 200
        n_features = 20

        # Create realistic-looking financial data
        X = np.random.randn(n_samples, n_features)
        # Make bankruptcy somewhat related to features
        y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.5 < -0.5).astype(
            int
        )

        self.predictor.X_train = X[:160]
        self.predictor.X_test = X[160:]
        self.predictor.y_train = y[:160]
        self.predictor.y_test = y[160:]

        # Mock exploration stats
        self.predictor.exploration_stats = {
            "shape": (n_samples, n_features + 1),
            "missing_values": 0,
            "target_distribution": {0: 150, 1: 50},
            "feature_count": n_features,
        }

    @patch("src.pipeline.DataProcessor")
    def test_data_loading_and_processing(self, mock_processor):
        """Test data loading and processing."""
        # Mock the data processor
        mock_instance = MagicMock()
        mock_processor.return_value = mock_instance
        mock_instance.process_full_pipeline.return_value = (
            self.predictor.X_train,
            self.predictor.X_test,
            self.predictor.y_train,
            self.predictor.y_test,
            self.predictor.exploration_stats,
        )

        # Test data loading
        self.predictor.load_and_process_data()

        self.assertIsNotNone(self.predictor.X_train)
        self.assertIsNotNone(self.predictor.X_test)
        self.assertIsNotNone(self.predictor.y_train)
        self.assertIsNotNone(self.predictor.y_test)

    def test_model_training(self):
        """Test model training functionality."""
        # Test with a subset of models for speed
        self.predictor.train_models(["logistic_regression"])

        self.assertGreater(len(self.predictor.models), 0)

        # Check that models are trained
        for model_name, model in self.predictor.models.items():
            self.assertTrue(model.is_trained)

    def test_model_evaluation(self):
        """Test model evaluation."""
        # First train some models
        self.predictor.train_models(["logistic_regression"])

        # Then evaluate them
        self.predictor.evaluate_models()

        self.assertGreater(len(self.predictor.results), 0)

        # Check that results contain required metrics
        for result in self.predictor.results:
            self.assertIn("accuracy", result)
            self.assertIn("f1_score", result)
            self.assertIn("model_name", result)

    def test_best_model_selection(self):
        """Test best model selection."""
        # Mock some results
        self.predictor.results = [
            {"model_name": "model1", "f1_score": 0.7, "accuracy": 0.8},
            {"model_name": "model2", "f1_score": 0.8, "accuracy": 0.75},
            {"model_name": "model3", "f1_score": 0.6, "accuracy": 0.85},
        ]

        best_name, best_result = self.predictor.get_best_model("f1_score")

        self.assertEqual(best_name, "model2")
        self.assertEqual(best_result["f1_score"], 0.8)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestConfig,
        TestDataProcessor,
        TestModelFactory,
        TestModelEvaluator,
        TestBankruptcyPredictor,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
