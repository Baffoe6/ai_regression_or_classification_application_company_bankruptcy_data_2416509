"""
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request model for single company bankruptcy prediction."""
    features: Dict[str, float] = Field(..., description="Financial indicators (X1-X95)")
    model_name: Optional[str] = Field(None, description="Specific model to use")
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "X1": 0.5,
                    "X2": 0.3,
                    "X3": 0.8,
                    "X4": 0.6,
                    "X5": 0.7,
                    # ... more features
                },
                "model_name": "ensemble_model"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for bankruptcy prediction."""
    prediction: int = Field(..., description="Bankruptcy prediction (0: Not Bankrupt, 1: Bankrupt)")
    probability: float = Field(..., description="Probability of bankruptcy")
    confidence: str = Field(..., description="Confidence level (Low/Medium/High)")
    model_used: str = Field(..., description="Name of the model used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 0,
                "probability": 0.25,
                "confidence": "High",
                "model_used": "ensemble_model",
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    companies: List[PredictionRequest]
    model_name: Optional[str] = Field(None, description="Specific model to use for all predictions")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction": 0,
                        "probability": 0.25,
                        "confidence": "High",
                        "model_used": "ensemble_model",
                        "timestamp": "2024-01-15T10:30:00"
                    }
                ],
                "summary": {
                    "total_companies": 100,
                    "predicted_bankruptcies": 15,
                    "bankruptcy_rate_percent": 15.0,
                    "model_used": "ensemble_model",
                    "timestamp": "2024-01-15T10:30:00"
                }
            }
        }


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="Model F1 score")
    roc_auc: float = Field(..., description="ROC AUC score")
    
    class Config:
        schema_extra = {
            "example": {
                "accuracy": 0.95,
                "precision": 0.93,
                "recall": 0.94,
                "f1_score": 0.935,
                "roc_auc": 0.98
            }
        }


class FeatureImportance(BaseModel):
    """Feature importance information."""
    feature_name: str = Field(..., description="Name of the feature")
    importance_score: float = Field(..., description="Importance score")
    rank: int = Field(..., description="Importance rank")


class ModelInfo(BaseModel):
    """Model information response."""
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type/algorithm")
    version: str = Field(..., description="Model version")
    performance_metrics: ModelMetrics = Field(..., description="Model performance metrics")
    features_used: int = Field(..., description="Number of features used")
    feature_importance: Optional[List[FeatureImportance]] = Field(None, description="Feature importance rankings")
    training_data_size: Optional[int] = Field(None, description="Size of training dataset")
    last_trained: Optional[str] = Field(None, description="Last training timestamp")
    status: str = Field(..., description="Model status (active/inactive)")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "ensemble_model",
                "type": "VotingClassifier",
                "version": "1.0.0",
                "performance_metrics": {
                    "accuracy": 0.95,
                    "precision": 0.93,
                    "recall": 0.94,
                    "f1_score": 0.935,
                    "roc_auc": 0.98
                },
                "features_used": 95,
                "training_data_size": 6819,
                "last_trained": "2024-01-15T08:00:00",
                "status": "active"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    models_loaded: int = Field(..., description="Number of models loaded")
    available_models: List[str] = Field(..., description="List of available model names")
    timestamp: str = Field(..., description="Health check timestamp")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime in seconds")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "models_loaded": 3,
                "available_models": ["ensemble_model", "xgboost_model", "random_forest_model"],
                "timestamp": "2024-01-15T10:30:00",
                "uptime_seconds": 3600.5,
                "memory_usage_mb": 512.8
            }
        }


class DataUploadResponse(BaseModel):
    """Response for data upload operations."""
    message: str = Field(..., description="Upload status message")
    file_size_mb: float = Field(..., description="Uploaded file size in MB")
    rows_processed: int = Field(..., description="Number of rows processed")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors found")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Upload timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Data uploaded and validated successfully",
                "file_size_mb": 2.5,
                "rows_processed": 1000,
                "validation_errors": [],
                "processing_time_seconds": 1.23,
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Missing required features: X1, X2, X3",
                "details": {
                    "missing_features": ["X1", "X2", "X3"],
                    "provided_features": 92
                },
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class ModelTrainingRequest(BaseModel):
    """Request model for training new models."""
    model_type: str = Field(..., description="Type of model to train")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Model hyperparameters")
    cross_validation_folds: int = Field(5, description="Number of CV folds")
    test_size: float = Field(0.2, description="Test set size")
    random_state: int = Field(42, description="Random state for reproducibility")
    
    class Config:
        schema_extra = {
            "example": {
                "model_type": "RandomForestClassifier",
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5
                },
                "cross_validation_folds": 5,
                "test_size": 0.2,
                "random_state": 42
            }
        }


class ModelTrainingResponse(BaseModel):
    """Response model for model training."""
    model_name: str = Field(..., description="Name of the trained model")
    training_status: str = Field(..., description="Training status")
    performance_metrics: ModelMetrics = Field(..., description="Training performance metrics")
    training_time_seconds: float = Field(..., description="Training time in seconds")
    model_size_mb: float = Field(..., description="Trained model size in MB")
    timestamp: str = Field(..., description="Training completion timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "random_forest_2024_01_15",
                "training_status": "completed",
                "performance_metrics": {
                    "accuracy": 0.95,
                    "precision": 0.93,
                    "recall": 0.94,
                    "f1_score": 0.935,
                    "roc_auc": 0.98
                },
                "training_time_seconds": 45.6,
                "model_size_mb": 12.8,
                "timestamp": "2024-01-15T10:30:00"
            }
        }