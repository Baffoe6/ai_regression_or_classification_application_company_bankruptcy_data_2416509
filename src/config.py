"""
Configuration management for the bankruptcy prediction project.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json


@dataclass
class ModelConfig:
    """Configuration for individual models."""

    name: str
    params: Dict[str, Any]
    enabled: bool = True


@dataclass
class Config:
    """Main configuration class for the bankruptcy prediction project."""

    # Data settings
    data_path: str = "CompanyBankruptcyData.csv"
    target_column: str = "Bankrupt?"
    test_size: float = 0.2
    random_state: int = 42

    # Feature engineering
    scale_features: bool = True
    feature_selection: bool = True
    max_features: Optional[int] = None

    # Model settings
    cross_validation_folds: int = 5
    early_stopping_patience: int = 10

    # Logging
    log_level: str = "INFO"
    log_file: str = "bankruptcy_prediction.log"

    # Output settings
    output_dir: str = "outputs"
    save_models: bool = True
    save_plots: bool = True

    @classmethod
    def from_json(cls, json_path: str) -> "Config":
        """Load configuration from JSON file."""
        with open(json_path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, json_path: str) -> None:
        """Save configuration to JSON file."""
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


# Default model configurations
DEFAULT_MODEL_CONFIGS = {
    "logistic_regression": [
        ModelConfig("LogReg_Default", {}),
        ModelConfig("LogReg_L2", {"C": 0.1, "penalty": "l2"}),
        ModelConfig("LogReg_L1", {"C": 10, "penalty": "l1", "solver": "liblinear"}),
    ],
    "random_forest": [
        ModelConfig("RF_Default", {}),
        ModelConfig("RF_Deep", {"n_estimators": 200, "max_depth": 20}),
        ModelConfig(
            "RF_Balanced",
            {"n_estimators": 150, "max_depth": 10, "class_weight": "balanced"},
        ),
    ],
}


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def setup_directories(config: Config) -> None:
    """Create necessary directories based on configuration."""
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "reports"), exist_ok=True)
