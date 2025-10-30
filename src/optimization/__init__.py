"""
Advanced model optimization techniques including hyperparameter tuning,
ensemble methods, and feature selection.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import xgboost as xgb
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from ..config import Config
from ..models import BaseModel, ModelFactory
from ..utils import get_logger

logger = get_logger(__name__)


class HyperparameterOptimizer:
    """Handles hyperparameter optimization for different models."""

    def __init__(self, config: Config):
        self.config = config

        # Define parameter grids
        self.param_grids = {
            "logistic_regression": {
                "C": [0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2", "elasticnet"],
                "solver": ["liblinear", "saga"],
                "max_iter": [1000, 2000],
            },
            "random_forest": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": [None, "balanced"],
            },
            "xgboost": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
            },
        }

    def optimize_hyperparameters(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv_folds: int = 5,
        n_iter: int = 50,
        scoring: str = "f1",
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using RandomizedSearchCV."""
        logger.info(f"Starting hyperparameter optimization for {model_type}")

        if model_type not in self.param_grids:
            raise ValueError(f"Parameter grid not defined for {model_type}")

        # Get base model
        base_model = self._get_base_model(model_type)
        param_grid = self.param_grids[model_type]

        # Setup search
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=n_iter,
            cv=cv_folds,
            scoring=scoring,
            random_state=self.config.random_state,
            n_jobs=-1,
            verbose=1,
        )

        # Fit search
        search.fit(X_train, y_train)

        logger.info(f"Best parameters for {model_type}: {search.best_params_}")
        logger.info(f"Best {scoring} score: {search.best_score_:.4f}")

        return {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "best_estimator": search.best_estimator_,
            "cv_results": search.cv_results_,
        }

    def _get_base_model(self, model_type: str):
        """Get base model for hyperparameter optimization."""
        if model_type == "logistic_regression":
            return LogisticRegression(random_state=self.config.random_state)
        elif model_type == "random_forest":
            return RandomForestClassifier(random_state=self.config.random_state)
        elif model_type == "xgboost":
            return xgb.XGBClassifier(random_state=self.config.random_state)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class FeatureSelector:
    """Advanced feature selection techniques."""

    def __init__(self, config: Config):
        self.config = config
        self.selected_features_ = None
        self.feature_scores_ = None

    def recursive_feature_elimination(
        self,
        estimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_features: int = 50,
        cv_folds: int = 5,
    ) -> np.ndarray:
        """Perform recursive feature elimination with cross-validation."""
        logger.info(f"Performing RFE with CV to select {n_features} features")

        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=cv_folds,
            scoring="f1",
            min_features_to_select=n_features,
        )

        selector.fit(X_train, y_train)

        self.selected_features_ = selector.support_
        self.feature_scores_ = selector.ranking_

        logger.info(f"Selected {selector.n_features_} features")

        return X_train[:, selector.support_]

    def model_based_selection(
        self,
        estimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        threshold: str = "median",
    ) -> np.ndarray:
        """Select features based on importance weights."""
        logger.info(
            f"Performing model-based feature selection with threshold={threshold}"
        )

        selector = SelectFromModel(estimator=estimator, threshold=threshold)
        selector.fit(X_train, y_train)

        self.selected_features_ = selector.get_support()

        logger.info(f"Selected {selector.get_support().sum()} features")

        return selector.transform(X_train)

    def get_selected_feature_indices(self) -> np.ndarray:
        """Get indices of selected features."""
        if self.selected_features_ is None:
            raise ValueError("Feature selection must be performed first")
        return np.where(self.selected_features_)[0]


class EnsembleBuilder:
    """Build ensemble models for improved performance."""

    def __init__(self, config: Config):
        self.config = config

    def create_voting_ensemble(
        self, models: List[Tuple[str, BaseModel]], voting: str = "soft"
    ) -> VotingClassifier:
        """Create a voting ensemble from trained models."""
        logger.info(f"Creating voting ensemble with {len(models)} models")

        # Extract sklearn models for voting classifier
        estimators = []
        for name, model in models:
            if hasattr(model, "model"):
                estimators.append((name, model.model))
            else:
                estimators.append((name, model))

        ensemble = VotingClassifier(estimators=estimators, voting=voting)

        logger.info("Voting ensemble created")
        return ensemble

    def create_bagging_ensemble(
        self,
        base_estimator,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
    ) -> BaggingClassifier:
        """Create a bagging ensemble."""
        logger.info(f"Creating bagging ensemble with {n_estimators} estimators")

        ensemble = BaggingClassifier(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            random_state=self.config.random_state,
            n_jobs=-1,
        )

        logger.info("Bagging ensemble created")
        return ensemble

    def create_stacked_ensemble(self, base_models: List[BaseModel], meta_model=None):
        """Create a stacked ensemble (requires additional implementation)."""
        # This would require implementing a custom stacking classifier
        # For now, return a voting ensemble as a placeholder
        logger.warning("Stacked ensemble not fully implemented, using voting ensemble")
        return self.create_voting_ensemble(
            [(f"model_{i}", model) for i, model in enumerate(base_models)]
        )


class ModelOptimizer:
    """Main class for model optimization."""

    def __init__(self, config: Config):
        self.config = config
        self.hyperparameter_optimizer = HyperparameterOptimizer(config)
        self.feature_selector = FeatureSelector(config)
        self.ensemble_builder = EnsembleBuilder(config)

    def optimize_single_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        optimize_features: bool = True,
        optimize_hyperparams: bool = True,
    ) -> Dict[str, Any]:
        """Optimize a single model with feature selection and hyperparameter tuning."""
        logger.info(f"Optimizing {model_type}")

        results = {}
        X_train_opt = X_train.copy()
        X_test_opt = X_test.copy()

        # Feature selection
        if optimize_features and self.config.feature_selection:
            base_model = self.hyperparameter_optimizer._get_base_model(model_type)

            if model_type == "random_forest":
                # Use the model itself for feature selection
                X_train_opt = self.feature_selector.model_based_selection(
                    base_model, X_train_opt, y_train
                )
                X_test_opt = X_test_opt[:, self.feature_selector.selected_features_]
            else:
                # Use RFE for other models
                rf_selector = RandomForestClassifier(
                    n_estimators=100, random_state=self.config.random_state
                )
                X_train_opt = self.feature_selector.recursive_feature_elimination(
                    rf_selector,
                    X_train_opt,
                    y_train,
                    n_features=min(50, X_train_opt.shape[1] // 2),
                )
                X_test_opt = X_test_opt[:, self.feature_selector.selected_features_]

            results["selected_features"] = (
                self.feature_selector.get_selected_feature_indices()
            )
            results["n_selected_features"] = len(results["selected_features"])

        # Hyperparameter optimization
        if optimize_hyperparams:
            opt_results = self.hyperparameter_optimizer.optimize_hyperparameters(
                model_type, X_train_opt, y_train
            )

            results["best_params"] = opt_results["best_params"]
            results["best_cv_score"] = opt_results["best_score"]
            results["optimized_model"] = opt_results["best_estimator"]
        else:
            # Use default model
            results["optimized_model"] = self.hyperparameter_optimizer._get_base_model(
                model_type
            )
            results["optimized_model"].fit(X_train_opt, y_train)

        # Store optimized data
        results["X_train_optimized"] = X_train_opt
        results["X_test_optimized"] = X_test_opt

        logger.info(f"Optimization completed for {model_type}")
        return results

    def create_optimized_ensemble(
        self,
        optimized_models: List[Dict[str, Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict[str, Any]:
        """Create an ensemble from optimized models."""
        logger.info("Creating optimized ensemble")

        # Prepare models for ensemble
        ensemble_models = []
        for i, model_result in enumerate(optimized_models):
            model_name = f"optimized_model_{i}"
            model = model_result["optimized_model"]
            ensemble_models.append((model_name, model))

        # Create voting ensemble
        voting_ensemble = self.ensemble_builder.create_voting_ensemble(
            ensemble_models, voting="soft"
        )

        # Train ensemble
        voting_ensemble.fit(X_train, y_train)

        # Create bagging ensemble with best model
        best_model = max(optimized_models, key=lambda x: x.get("best_cv_score", 0))[
            "optimized_model"
        ]
        bagging_ensemble = self.ensemble_builder.create_bagging_ensemble(
            best_model, n_estimators=10
        )
        bagging_ensemble.fit(X_train, y_train)

        results = {
            "voting_ensemble": voting_ensemble,
            "bagging_ensemble": bagging_ensemble,
            "base_models": ensemble_models,
        }

        logger.info("Optimized ensemble created")
        return results

    def save_optimization_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Save optimization results."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save serializable parts
        serializable_results = {}
        for key, value in results.items():
            if key.endswith("_model") or key.endswith("_ensemble"):
                # Save model separately
                model_path = filepath.replace(".joblib", f"_{key}.joblib")
                joblib.dump(value, model_path)
                serializable_results[key] = model_path
            else:
                serializable_results[key] = value

        joblib.dump(serializable_results, filepath)
        logger.info(f"Optimization results saved to {filepath}")


def create_advanced_models(config: Config) -> Dict[str, Any]:
    """Create additional advanced models like XGBoost."""

    models = {}

    try:
        # XGBoost model
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.random_state,
        )
        models["xgboost"] = xgb_model
        logger.info("XGBoost model created")

    except ImportError:
        logger.warning("XGBoost not available, skipping XGBoost model")

    return models
