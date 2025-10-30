"""
Simplified models module without TensorFlow dependency for initial testing.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ..config import Config, ModelConfig
from ..utils import get_logger

logger = get_logger(__name__)


class BaseModel:
    """Base class for all models."""

    def __init__(self, name: str, config: ModelConfig):
        self.name = name
        self.config = config
        self.model = None
        self.is_trained = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)

    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model wrapper."""

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the logistic regression model."""
        logger.info(f"Training {self.name}")

        self.model = LogisticRegression(
            random_state=42, max_iter=1000, **self.config.params
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info(f"{self.name} training completed")


class RandomForestModel(BaseModel):
    """Random Forest model wrapper."""

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the random forest model."""
        logger.info(f"Training {self.name}")

        self.model = RandomForestClassifier(random_state=42, **self.config.params)

        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info(f"{self.name} training completed")

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance from the trained model."""
        if self.is_trained:
            return self.model.feature_importances_
        return None


class ModelFactory:
    """Factory class for creating models."""

    @staticmethod
    def create_model(
        model_type: str, config: ModelConfig, input_dim: Optional[int] = None, **kwargs
    ) -> BaseModel:
        """Create a model instance based on type and configuration."""

        if model_type == "logistic_regression":
            return LogisticRegressionModel(config.name, config)
        elif model_type == "random_forest":
            return RandomForestModel(config.name, config)
        elif model_type == "neural_network":
            logger.warning(
                "Neural network models require TensorFlow. Skipping for now."
            )
            return None
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class ModelEvaluator:
    """Evaluates model performance."""

    @staticmethod
    def evaluate_model(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str,
    ) -> Dict[str, float]:
        """Evaluate model performance with multiple metrics."""

        metrics = {
            "model_name": model_name,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred_proba[:, 1]),
        }

        logger.info(f"Evaluation completed for {model_name}: {metrics}")
        return metrics

    @staticmethod
    def cross_validate_model(
        model: BaseModel, X: np.ndarray, y: np.ndarray, cv_folds: int = 5
    ) -> Dict[str, float]:
        """Perform cross-validation on a model."""
        logger.info(f"Performing {cv_folds}-fold cross-validation for {model.name}")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Accuracy scores
        cv_scores = cross_val_score(model.model, X, y, cv=cv, scoring="accuracy")

        cv_results = {
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std_accuracy": cv_scores.std(),
            "cv_scores": cv_scores.tolist(),
        }

        logger.info(f"Cross-validation completed for {model.name}: {cv_results}")
        return cv_results


class ModelManager:
    """Manages model loading and lifecycle for API endpoints."""

    def __init__(self, config: Config):
        """Initialize the ModelManager with configuration."""
        self.config = config
        self.loaded_models = {}
        self.feature_names = None
        logger.info("ModelManager initialized")

    def load_models(self, models_dir: str = None) -> None:
        """Load models from disk."""
        if models_dir is None:
            models_dir = os.path.join(self.config.output_dir, "models")

        if not os.path.exists(models_dir):
            logger.warning(f"Models directory not found: {models_dir}")
            return

        for filename in os.listdir(models_dir):
            if filename.endswith(".joblib"):
                model_name = filename.replace(".joblib", "")
                model_path = os.path.join(models_dir, filename)

                try:
                    model = joblib.load(model_path)
                    self.loaded_models[model_name] = model
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {str(e)}")

        logger.info(f"Loaded {len(self.loaded_models)} models")

    def get_model(self, model_name: str = None):
        """Get a model by name, or return the first available model."""
        if not self.loaded_models:
            logger.warning("No models loaded")
            return None

        if model_name is None:
            return self.loaded_models[list(self.loaded_models.keys())[0]]

        if model_name not in self.loaded_models:
            logger.warning(f"Model {model_name} not found")
            return None

        return self.loaded_models[model_name]

    def list_models(self) -> List[str]:
        """List all loaded model names."""
        return list(self.loaded_models.keys())

    def predict(self, model_name: str, features: np.ndarray):
        """Make a prediction using a specific model."""
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not available")

        return model.predict(features)

    def predict_proba(self, model_name: str, features: np.ndarray):
        """Get prediction probabilities using a specific model."""
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not available")

        if hasattr(model, "predict_proba"):
            return model.predict_proba(features)
        else:
            logger.warning(f"Model {model_name} does not support predict_proba")
            return None
