"""
Comprehensive test suite for data processing pipeline.
Tests data loading, preprocessing, validation, and feature engineering.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config  # noqa: E402
from src.data import DataProcessor  # noqa: E402


class TestDataProcessor:
    """Test suite for DataProcessor class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample bankruptcy data for testing."""
        np.random.seed(42)
        n_samples = 100

        # Create realistic financial ratios
        data = {
            "ROA(C)_before_interest_and_depreciation_before_interest": np.random.normal(
                0.1, 0.05, n_samples
            ),
            "ROA(A)_before_interest_and_%_after_tax": np.random.normal(
                0.08, 0.04, n_samples
            ),
            "ROA(B)_before_interest_and_depreciation_after_tax": np.random.normal(
                0.09, 0.045, n_samples
            ),
            "Operating_Gross_Margin": np.random.normal(0.25, 0.1, n_samples),
            "Realized_Sales_Gross_Margin": np.random.normal(0.2, 0.08, n_samples),
            "Net_Value_Per_Share_(B)": np.random.normal(10, 5, n_samples),
            "Net_Value_Per_Share_(A)": np.random.normal(12, 6, n_samples),
            "Working_Capital/Total_Assets": np.random.normal(0.15, 0.1, n_samples),
            "Current_Ratio": np.random.normal(1.5, 0.5, n_samples),
            "Quick_Ratio": np.random.normal(1.2, 0.4, n_samples),
            "Bankrupt?": np.random.binomial(1, 0.1, n_samples),  # 10% bankruptcy rate
        }

        # Add more features to simulate real dataset
        for i in range(85):  # Total of 95 features like original dataset
            data[f"X{i+11}"] = np.random.normal(0, 1, n_samples)

        return pd.DataFrame(data)

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            target_column="Bankrupt?",
            test_size=0.3,
            random_state=42,
            feature_selection=True,
            max_features=20,
            scale_features=True,
        )

    @pytest.fixture
    def processor(self, config):
        """Create DataProcessor instance."""
        return DataProcessor(config)

    def test_data_loading(self, processor, sample_data):
        """Test data loading functionality."""
        # Save sample data to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            # Test loading
            loaded_data = processor.load_data(temp_file)

            assert isinstance(loaded_data, pd.DataFrame)
            assert len(loaded_data) == len(sample_data)
            assert all(col in loaded_data.columns for col in sample_data.columns)

        finally:
            os.unlink(temp_file)

    def test_data_validation(self, processor, sample_data):
        """Test data validation functionality."""
        # Test valid data
        is_valid, message = processor.validate_data(sample_data)
        assert is_valid
        assert "Data validation passed" in message

        # Test missing target column
        invalid_data = sample_data.drop("Bankrupt?", axis=1)
        is_valid, message = processor.validate_data(invalid_data)
        assert not is_valid
        assert "Target column" in message

        # Test insufficient samples
        small_data = sample_data.head(5)
        is_valid, message = processor.validate_data(small_data)
        assert not is_valid
        assert "Insufficient" in message

    def test_preprocessing(self, processor, sample_data):
        """Test data preprocessing pipeline."""
        X, y = processor.preprocess_data(sample_data)

        # Check output types and shapes
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y) == len(sample_data)
        assert y.name == "Bankrupt?"

        # Check that target column is removed from features
        assert "Bankrupt?" not in X.columns

        # Check for missing values
        assert not X.isnull().any().any()
        assert not y.isnull().any()

    def test_feature_selection(self, processor, sample_data):
        """Test feature selection functionality."""
        X, y = processor.preprocess_data(sample_data)

        # Test feature selection
        X_selected = processor.select_features(X, y)

        assert isinstance(X_selected, pd.DataFrame)
        assert len(X_selected.columns) <= processor.config.max_features
        assert len(X_selected) == len(X)

        # Feature importance should be available
        assert hasattr(processor, "feature_importance_")
        assert len(processor.feature_importance_) == len(X_selected.columns)

    def test_feature_scaling(self, processor, sample_data):
        """Test feature scaling functionality."""
        X, y = processor.preprocess_data(sample_data)

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Test scaling
        X_train_scaled, X_test_scaled = processor.scale_features(X_train, X_test)

        assert isinstance(X_train_scaled, np.ndarray)
        assert isinstance(X_test_scaled, np.ndarray)
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape

        # Check that features are scaled (mean ≈ 0, std ≈ 1)
        train_means = np.mean(X_train_scaled, axis=0)
        train_stds = np.std(X_train_scaled, axis=0)

        assert np.allclose(train_means, 0, atol=1e-10)
        assert np.allclose(train_stds, 1, atol=1e-10)

    def test_data_splitting(self, processor, sample_data):
        """Test train/test data splitting."""
        X, y = processor.preprocess_data(sample_data)

        X_train, X_test, y_train, y_test = processor.split_data(X, y)

        # Check shapes
        total_samples = len(X)
        expected_test_size = int(total_samples * processor.config.test_size)

        assert len(X_test) == len(y_test) == expected_test_size
        assert len(X_train) == len(y_train) == total_samples - expected_test_size

        # Check stratification (similar class distribution)
        train_pos_ratio = y_train.mean()
        test_pos_ratio = y_test.mean()
        overall_pos_ratio = y.mean()

        assert abs(train_pos_ratio - overall_pos_ratio) < 0.05
        assert abs(test_pos_ratio - overall_pos_ratio) < 0.05

    def test_handle_missing_values(self, processor, sample_data):
        """Test missing value handling."""
        # Introduce missing values
        data_with_missing = sample_data.copy()
        data_with_missing.iloc[0:5, 1:3] = np.nan

        X, y = processor.preprocess_data(data_with_missing)

        # Check that missing values are handled
        assert not X.isnull().any().any()
        assert not y.isnull().any()

    def test_outlier_detection(self, processor, sample_data):
        """Test outlier detection and handling."""
        # Introduce outliers
        data_with_outliers = sample_data.copy()
        data_with_outliers.iloc[0, 1] = 1000  # Extreme outlier

        X, y = processor.preprocess_data(data_with_outliers)

        # Data should still be valid after outlier handling
        assert isinstance(X, pd.DataFrame)
        assert len(X) == len(sample_data)
        assert not X.isnull().any().any()

    def test_categorical_encoding(self, processor):
        """Test categorical variable encoding."""
        # Create data with categorical variables
        data = pd.DataFrame(
            {
                "category_A": ["high", "medium", "low", "high", "medium"],
                "category_B": ["yes", "no", "yes", "no", "yes"],
                "numeric": [1.0, 2.0, 3.0, 4.0, 5.0],
                "Bankrupt?": [0, 1, 0, 1, 0],
            }
        )

        X, y = processor.preprocess_data(data)

        # Check that categorical variables are encoded
        assert isinstance(X, pd.DataFrame)
        assert X.dtypes.apply(lambda x: x.kind in "biufc").all()  # All numeric types

    def test_feature_engineering(self, processor, sample_data):
        """Test feature engineering capabilities."""
        X, y = processor.preprocess_data(sample_data)

        # Check that feature engineering maintains data integrity
        assert isinstance(X, pd.DataFrame)
        assert len(X) == len(sample_data)
        assert X.dtypes.apply(lambda x: x.kind in "biufc").all()

    def test_data_leakage_prevention(self, processor, sample_data):
        """Test that preprocessing prevents data leakage."""
        X, y = processor.preprocess_data(sample_data)
        X_train, X_test, y_train, y_test = processor.split_data(X, y)

        # Fit scaler on training data only
        X_train_scaled, X_test_scaled = processor.scale_features(X_train, X_test)

        # Test set should not influence training set statistics
        scaler = processor.scaler_
        train_means = scaler.mean_

        # Means should be based on training data only
        expected_means = X_train.mean().values
        assert np.allclose(train_means, expected_means, rtol=1e-5)

    def test_reproducibility(self, processor, sample_data):
        """Test that preprocessing is reproducible."""
        # Process data twice
        X1, y1 = processor.preprocess_data(sample_data)
        X2, y2 = processor.preprocess_data(sample_data)

        # Results should be identical
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)

    def test_error_handling(self, processor):
        """Test error handling for edge cases."""
        # Test empty dataframe
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError):
            processor.preprocess_data(empty_df)

        # Test dataframe with only target column
        target_only_df = pd.DataFrame({"Bankrupt?": [0, 1, 0]})
        with pytest.raises(ValueError):
            processor.preprocess_data(target_only_df)

    def test_memory_efficiency(self, processor):
        """Test memory efficiency with large datasets."""
        # Create larger dataset
        np.random.seed(42)
        large_data = pd.DataFrame(
            {f"feature_{i}": np.random.randn(10000) for i in range(50)}
        )
        large_data["Bankrupt?"] = np.random.binomial(1, 0.1, 10000)

        # Process without memory issues
        X, y = processor.preprocess_data(large_data)

        assert isinstance(X, pd.DataFrame)
        assert len(X) == 10000
        assert (
            X.memory_usage(deep=True).sum() < large_data.memory_usage(deep=True).sum()
        )


class TestDataValidation:
    """Test suite for data validation utilities."""

    def test_schema_validation(self):
        """Test data schema validation."""
        # Valid schema
        valid_data = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [0.5, 1.5, 2.5],
                "Bankrupt?": [0, 1, 0],
            }
        )

        processor = DataProcessor(Config())
        is_valid, message = processor.validate_data(valid_data)
        assert is_valid

    def test_data_quality_checks(self):
        """Test data quality validation."""
        # Data with quality issues
        poor_quality_data = pd.DataFrame(
            {
                "feature1": [1.0, np.inf, 3.0],  # Contains infinity
                "feature2": [0.5, 1.5, -np.inf],  # Contains negative infinity
                "Bankrupt?": [0, 1, 0],
            }
        )

        processor = DataProcessor(Config())

        # Should handle infinite values
        X, y = processor.preprocess_data(poor_quality_data)
        assert not np.isinf(X.values).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
