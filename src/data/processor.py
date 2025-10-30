"""
Data processing and validation module.
Handles data loading, cleaning, preprocessing, and validation.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Comprehensive data processing pipeline for bankruptcy prediction.
    Handles loading, validation, cleaning, preprocessing, and feature engineering.
    """

    def __init__(self, config):
        """
        Initialize DataProcessor with configuration.

        Args:
            config: Configuration object with processing parameters
        """
        self.config = config
        self.scaler_ = None
        self.feature_selector_ = None
        self.feature_importance_ = None
        self.selected_features_ = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file with error handling.

        Args:
            file_path: Path to the CSV file

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.EmptyDataError: If file is empty
            ValueError: If file format is invalid
        """
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)

            if df.empty:
                raise ValueError("Data file is empty")

            logger.info(
                f"Successfully loaded {len(df)} rows and {len(df.columns)} columns"
            )
            return df

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error("Data file is empty")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise ValueError(f"Failed to load data: {str(e)}")

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate data quality and structure.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, message)
        """
        if df is None or df.empty:
            return False, "DataFrame is empty or None"

        if len(df) < 10:
            return False, f"Insufficient data samples: {len(df)} (minimum 10 required)"

        if self.config.target_column not in df.columns:
            return (
                False,
                f"Target column '{self.config.target_column}' not found in data",
            )

        # Check for minimum feature count
        feature_columns = [
            col for col in df.columns if col != self.config.target_column
        ]
        if len(feature_columns) < 2:
            return (
                False,
                f"Insufficient features: {len(feature_columns)} (minimum 2 required)",
            )

        # Check target variable
        target_unique = df[self.config.target_column].nunique()
        if target_unique < 2:
            return (
                False,
                f"Target variable has only {target_unique} unique values (minimum 2 required)",
            )

        # Check for excessive missing values
        missing_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_percentage > 0.5:
            return (
                False,
                f"Excessive missing values: {missing_percentage:.1%} (maximum 50% allowed)",
            )

        logger.info("Data validation passed")
        return True, "Data validation passed"

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete data preprocessing pipeline.

        Args:
            df: Raw DataFrame

        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info("Starting data preprocessing")

        # Validate data first
        is_valid, message = self.validate_data(df)
        if not is_valid:
            raise ValueError(f"Data validation failed: {message}")

        # Create a copy to avoid modifying original data
        df_processed = df.copy()

        # Handle infinite values
        df_processed = self._handle_infinite_values(df_processed)

        # Handle missing values
        df_processed = self._handle_missing_values(df_processed)

        # Handle outliers
        df_processed = self._handle_outliers(df_processed)

        # Encode categorical variables
        df_processed = self._encode_categorical_variables(df_processed)

        # Separate features and target
        X = df_processed.drop(self.config.target_column, axis=1)
        y = df_processed[self.config.target_column]

        logger.info(
            f"Preprocessing completed: {len(X)} samples, {len(X.columns)} features"
        )
        return X, y

    def _handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle infinite values in the dataset."""
        logger.info("Handling infinite values")

        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using appropriate imputation strategies."""
        logger.info("Handling missing values")

        missing_counts = df.isnull().sum()
        if missing_counts.sum() == 0:
            return df

        # Log missing value statistics
        logger.info(f"Found missing values in {(missing_counts > 0).sum()} columns")

        # Separate numeric and categorical columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Remove target column from processing
        if self.config.target_column in numeric_columns:
            numeric_columns.remove(self.config.target_column)
        if self.config.target_column in categorical_columns:
            categorical_columns.remove(self.config.target_column)

        # Impute numeric columns with median
        if numeric_columns:
            numeric_imputer = SimpleImputer(strategy="median")
            df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

        # Impute categorical columns with mode
        if categorical_columns:
            categorical_imputer = SimpleImputer(strategy="most_frequent")
            df[categorical_columns] = categorical_imputer.fit_transform(
                df[categorical_columns]
            )

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method for numeric columns."""
        logger.info("Handling outliers")

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.config.target_column in numeric_columns:
            numeric_columns.remove(self.config.target_column)

        outlier_counts = {}

        for column in numeric_columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            outlier_count = outliers.sum()

            if outlier_count > 0:
                outlier_counts[column] = outlier_count
                # Cap outliers instead of removing them
                df.loc[df[column] < lower_bound, column] = lower_bound
                df.loc[df[column] > upper_bound, column] = upper_bound

        if outlier_counts:
            logger.info(f"Handled outliers in {len(outlier_counts)} columns")

        return df

    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables using appropriate methods."""
        logger.info("Encoding categorical variables")

        categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if self.config.target_column in categorical_columns:
            categorical_columns.remove(self.config.target_column)

        if not categorical_columns:
            return df

        # Use one-hot encoding for categorical variables
        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

        logger.info(f"Encoded {len(categorical_columns)} categorical columns")
        return df_encoded

    def select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Select the most important features using multiple methods.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            DataFrame with selected features
        """
        if not self.config.feature_selection:
            logger.info("Feature selection disabled")
            return X

        logger.info(f"Selecting top {self.config.max_features} features")

        if len(X.columns) <= self.config.max_features:
            logger.info(
                f"Number of features ({len(X.columns)}) already <= max_features ({self.config.max_features})"
            )
            self.selected_features_ = X.columns.tolist()
            return X

        # Use Random Forest for feature importance
        rf = RandomForestClassifier(
            n_estimators=100, random_state=self.config.random_state
        )
        rf.fit(X, y)

        # Get feature importances
        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)

        # Select top features
        top_features = feature_importance.head(self.config.max_features)[
            "feature"
        ].tolist()

        # Store feature importance and selected features
        self.feature_importance_ = feature_importance["importance"].values
        self.selected_features_ = top_features

        X_selected = X[top_features]

        logger.info(f"Selected {len(top_features)} features based on importance")
        return X_selected

    def scale_features(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler.

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            Tuple of (scaled_train, scaled_test)
        """
        if not self.config.scale_features:
            logger.info("Feature scaling disabled")
            return X_train.values, X_test.values

        logger.info("Scaling features")

        # Initialize scaler
        self.scaler_ = StandardScaler()

        # Fit on training data only
        X_train_scaled = self.scaler_.fit_transform(X_train)
        X_test_scaled = self.scaler_.transform(X_test)

        logger.info("Feature scaling completed")
        return X_train_scaled, X_test_scaled

    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data with test_size={self.config.test_size}")

        # Ensure stratification is possible
        if y.nunique() >= 2 and y.value_counts().min() >= 2:
            stratify = y
        else:
            stratify = None
            logger.warning(
                "Stratification disabled due to insufficient samples in some classes"
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=stratify,
        )

        logger.info(
            f"Data split completed: {len(X_train)} train, {len(X_test)} test samples"
        )
        return X_train, X_test, y_train, y_test

    def get_feature_names(self) -> List[str]:
        """Get the names of selected features."""
        if self.selected_features_ is not None:
            return self.selected_features_
        else:
            return []

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        return self.feature_importance_


class DataValidator:
    """
    Standalone data validation utilities.
    """

    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality assessment.

        Args:
            df: DataFrame to assess

        Returns:
            Dictionary with quality metrics
        """
        quality_report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "missing_percentage": df.isnull().sum().sum() / (len(df) * len(df.columns)),
            "duplicate_rows": df.duplicated().sum(),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(
                df.select_dtypes(include=["object", "category"]).columns
            ),
            "unique_values_per_column": df.nunique().to_dict(),
            "data_types": df.dtypes.to_dict(),
        }

        # Check for infinite values
        numeric_df = df.select_dtypes(include=[np.number])
        infinite_values = np.isinf(numeric_df).sum().sum()
        quality_report["infinite_values"] = infinite_values

        # Memory usage
        quality_report["memory_usage_mb"] = (
            df.memory_usage(deep=True).sum() / 1024 / 1024
        )

        return quality_report

    @staticmethod
    def detect_outliers(df: pd.DataFrame, method: str = "iqr") -> Dict[str, int]:
        """
        Detect outliers in numeric columns.

        Args:
            df: DataFrame to analyze
            method: Outlier detection method ('iqr' or 'zscore')

        Returns:
            Dictionary with outlier counts per column
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}

        for column in numeric_columns:
            if method == "iqr":
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            elif method == "zscore":
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                outliers = z_scores > 3
            else:
                raise ValueError(f"Unknown method: {method}")

            outlier_counts[column] = outliers.sum()

        return outlier_counts
