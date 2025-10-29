"""
Data processing module for bankruptcy prediction.
Provides comprehensive data loading, preprocessing, validation, and quality analysis.
"""

# Import existing DataProcessor for backward compatibility
from .processor import DataProcessor as NewDataProcessor, DataValidator
from .validator import (
    DataSchema,
    BankruptcyDataSchema,
    DataQualityAnalyzer,
    validate_bankruptcy_data,
    analyze_data_quality,
)

# Keep existing DataProcessor for compatibility
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Any
import os

from ..utils import get_logger
from ..config import Config

logger = get_logger(__name__)


class DataProcessor:
    """Handles data loading, preprocessing, and feature engineering."""

    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler() if config.scale_features else None
        self.feature_selector = None
        self.feature_names = None

    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Load the bankruptcy dataset."""
        path = data_path or self.config.data_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path)

        # Basic data validation
        if self.config.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.config.target_column}' not found in data"
            )

        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df

    def explore_data(self, df: pd.DataFrame) -> dict:
        """Generate data exploration statistics."""
        logger.info("Performing data exploration")

        exploration_stats = {
            "shape": df.shape,
            "missing_values": df.isnull().sum().sum(),
            "target_distribution": df[self.config.target_column]
            .value_counts()
            .to_dict(),
            "feature_count": len(
                [col for col in df.columns if col != self.config.target_column]
            ),
            "duplicates": df.duplicated().sum(),
            "data_types": df.dtypes.value_counts().to_dict(),
        }

        logger.info(f"Data exploration completed: {exploration_stats}")
        return exploration_stats

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the data for modeling."""
        logger.info("Starting data preprocessing")

        # Separate features and target
        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Handle missing values if any
        if X.isnull().sum().sum() > 0:
            logger.warning("Missing values detected, filling with median")
            X = X.fillna(X.median())

        # Feature selection
        if self.config.feature_selection and self.config.max_features:
            logger.info(
                f"Performing feature selection to top {self.config.max_features} features"
            )
            self.feature_selector = SelectKBest(
                score_func=f_classif, k=self.config.max_features
            )
            X = self.feature_selector.fit_transform(X, y)

            # Update feature names
            if self.feature_selector:
                selected_indices = self.feature_selector.get_support(indices=True)
                self.feature_names = [self.feature_names[i] for i in selected_indices]

        logger.info("Data preprocessing completed")
        return X.values if hasattr(X, "values") else X, y.values

    def split_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets."""
        logger.info("Splitting data into train and test sets")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        logger.info(
            f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}"
        )
        return X_train, X_test, y_train, y_test

    def scale_features(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using StandardScaler."""
        if not self.config.scale_features:
            return X_train, X_test

        logger.info("Scaling features")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores from feature selector."""
        if self.feature_selector:
            return self.feature_selector.scores_
        return None

    def process_full_pipeline(
        self, data_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Execute the full data processing pipeline."""
        logger.info("Starting full data processing pipeline")

        # Load data
        df = self.load_data(data_path)

        # Explore data
        exploration_stats = self.explore_data(df)

        # Preprocess
        X, y = self.preprocess_data(df)

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        logger.info("Full data processing pipeline completed")

        return X_train_scaled, X_test_scaled, y_train, y_test, exploration_stats


# Export all classes and functions
__all__ = [
    "DataProcessor",
    "NewDataProcessor",
    "DataValidator",
    "DataSchema",
    "BankruptcyDataSchema",
    "DataQualityAnalyzer",
    "validate_bankruptcy_data",
    "analyze_data_quality",
]
