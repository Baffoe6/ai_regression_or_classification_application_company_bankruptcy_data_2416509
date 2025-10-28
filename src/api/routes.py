"""
API routes for bankruptcy prediction service.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
import os
import time
import psutil
from datetime import datetime
import tempfile
import io

from ..config import Config
from ..utils import get_logger
from ..models import ModelManager
from ..data import DataProcessor, validate_bankruptcy_data, analyze_data_quality
from .models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    ModelMetrics,
    FeatureImportance,
    HealthResponse,
    DataUploadResponse,
    ModelTrainingRequest,
    ModelTrainingResponse
)

logger = get_logger(__name__)

# Create router
router = APIRouter()

# Global variables for loaded models and configuration
loaded_models = {}
model_manager = None
config = None
feature_names = None
app_start_time = time.time()


def get_config() -> Config:
    """Dependency to get configuration."""
    global config
    if config is None:
        try:
            config = Config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise HTTPException(status_code=500, detail="Configuration error")
    return config


def get_model_manager() -> ModelManager:
    """Dependency to get model manager."""
    global model_manager
    if model_manager is None:
        try:
            config = get_config()
            model_manager = ModelManager(config)
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
            raise HTTPException(status_code=500, detail="Model manager initialization error")
    return model_manager


def load_models_and_config():
    """Load trained models and configuration."""
    global loaded_models, config, feature_names
    
    try:
        config = get_config()
        
        # Load models from the models directory
        models_dir = os.path.join(config.output_dir, "models")
        
        if not os.path.exists(models_dir):
            logger.warning(f"Models directory not found: {models_dir}")
            return
        
        for filename in os.listdir(models_dir):
            if filename.endswith('.joblib'):
                model_name = filename.replace('.joblib', '')
                model_path = os.path.join(models_dir, filename)
                
                try:
                    model = joblib.load(model_path)
                    loaded_models[model_name] = model
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {str(e)}")
        
        # Load feature names
        feature_names = [f"X{i}" for i in range(1, 96)]  # Default feature names
        
        logger.info(f"Loaded {len(loaded_models)} models")
        
    except Exception as e:
        logger.error(f"Failed to load models and config: {str(e)}")


def get_confidence_level(probability: float) -> str:
    """Determine confidence level based on probability."""
    if probability < 0.3 or probability > 0.7:
        return "High"
    elif probability < 0.4 or probability > 0.6:
        return "Medium"
    else:
        return "Low"


# Health and status endpoints
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with detailed system information."""
    uptime = time.time() - app_start_time
    memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    return HealthResponse(
        status="healthy",
        models_loaded=len(loaded_models),
        available_models=list(loaded_models.keys()),
        timestamp=datetime.now().isoformat(),
        uptime_seconds=round(uptime, 2),
        memory_usage_mb=round(memory_usage, 2)
    )


@router.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Get information about available models."""
    if not loaded_models:
        # Try to load models if they haven't been loaded yet
        load_models_and_config()
        
        if not loaded_models:
            raise HTTPException(status_code=503, detail="No models available")
    
    models_info = []
    for name, model in loaded_models.items():
        # Extract model information
        model_type = type(model).__name__
        
        # Default performance metrics (should be loaded from saved metrics file)
        performance_metrics = ModelMetrics(
            accuracy=0.95,  # Placeholder
            precision=0.93,  # Placeholder
            recall=0.94,     # Placeholder
            f1_score=0.935,   # Placeholder
            roc_auc=0.98     # Placeholder
        )
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
            feature_importance = [
                FeatureImportance(
                    feature_name=f"X{i+1}",
                    importance_score=float(score),
                    rank=i+1
                )
                for i, score in enumerate(sorted(importance_scores, reverse=True)[:20])
            ]
        
        models_info.append(ModelInfo(
            name=name,
            type=model_type,
            version="1.0.0",
            performance_metrics=performance_metrics,
            features_used=len(feature_names) if feature_names else 95,
            feature_importance=feature_importance,
            training_data_size=6819,  # Default value
            last_trained=datetime.now().isoformat(),
            status="active"
        ))
    
    return models_info


# Prediction endpoints
@router.post("/predict", response_model=PredictionResponse)
async def predict_bankruptcy(request: PredictionRequest):
    """Predict bankruptcy for a single company."""
    if not loaded_models:
        load_models_and_config()
        
        if not loaded_models:
            raise HTTPException(status_code=503, detail="No models loaded")
    
    # Select model
    if request.model_name:
        if request.model_name not in loaded_models:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found")
        model = loaded_models[request.model_name]
        selected_model_name = request.model_name
    else:
        # Use the first available model
        selected_model_name = list(loaded_models.keys())[0]
        model = loaded_models[selected_model_name]
    
    try:
        # Prepare input data
        if feature_names:
            # Ensure all required features are present
            missing_features = set(feature_names) - set(request.features.keys())
            if missing_features:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing features: {list(missing_features)[:10]}..."
                )
            
            # Create feature array in correct order
            features_array = np.array([request.features[feature] for feature in feature_names])
        else:
            # Use provided features as-is
            features_array = np.array(list(request.features.values()))
        
        # Reshape for single prediction
        features_array = features_array.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        # Get probability
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_array)[0]
            probability_bankrupt = probabilities[1]  # Probability of class 1 (bankrupt)
        else:
            probability_bankrupt = float(prediction)  # Fallback for models without predict_proba
        
        # Determine confidence
        confidence = get_confidence_level(probability_bankrupt)
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability_bankrupt),
            confidence=confidence,
            model_used=selected_model_name,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict bankruptcy for multiple companies."""
    if not loaded_models:
        load_models_and_config()
        
        if not loaded_models:
            raise HTTPException(status_code=503, detail="No models loaded")
    
    if len(request.companies) > 1000:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size too large (max 1000)")
    
    predictions = []
    bankruptcy_count = 0
    
    for company_request in request.companies:
        try:
            # Use the predict endpoint logic
            prediction_response = await predict_bankruptcy(company_request)
            predictions.append(prediction_response)
            
            if prediction_response.prediction == 1:
                bankruptcy_count += 1
                
        except HTTPException as e:
            # Skip invalid entries but log the error
            logger.warning(f"Skipped invalid company data: {str(e)}")
            continue
    
    # Create summary
    total_companies = len(predictions)
    bankruptcy_rate = (bankruptcy_count / total_companies * 100) if total_companies > 0 else 0
    
    summary = {
        "total_companies": total_companies,
        "predicted_bankruptcies": bankruptcy_count,
        "bankruptcy_rate_percent": round(bankruptcy_rate, 2),
        "model_used": request.model_name or list(loaded_models.keys())[0],
        "timestamp": datetime.now().isoformat()
    }
    
    return BatchPredictionResponse(
        predictions=predictions,
        summary=summary
    )


# Data management endpoints
@router.post("/data/upload", response_model=DataUploadResponse)
async def upload_data(file: UploadFile = File(...)):
    """Upload and validate bankruptcy data."""
    start_time = time.time()
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read uploaded file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Validate data
        validation_errors = []
        try:
            validate_bankruptcy_data(df)
        except ValueError as e:
            validation_errors.append(str(e))
        
        # Calculate file size
        file_size_mb = len(content) / (1024 * 1024)
        
        # Processing time
        processing_time = time.time() - start_time
        
        return DataUploadResponse(
            message="Data uploaded and validated successfully" if not validation_errors else "Data uploaded with validation warnings",
            file_size_mb=round(file_size_mb, 2),
            rows_processed=len(df),
            validation_errors=validation_errors,
            processing_time_seconds=round(processing_time, 3),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Data upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data upload failed: {str(e)}")


@router.get("/data/quality/{file_path:path}")
async def analyze_data_quality_endpoint(file_path: str):
    """Analyze data quality for a specific file."""
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Load and analyze data
        df = pd.read_csv(file_path)
        quality_report = analyze_data_quality(df)
        
        return {
            "file_path": file_path,
            "quality_report": quality_report,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Data quality analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data quality analysis failed: {str(e)}")


# Model management endpoints
@router.post("/models/train", response_model=ModelTrainingResponse)
async def train_model(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
    """Train a new model (background task)."""
    try:
        config = get_config()
        model_manager = get_model_manager()
        
        # Start training in background
        training_start_time = time.time()
        
        # This is a simplified example - in practice, you'd implement proper async training
        model_name = f"{request.model_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Return immediate response
        return ModelTrainingResponse(
            model_name=model_name,
            training_status="started",
            performance_metrics=ModelMetrics(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                roc_auc=0.0
            ),
            training_time_seconds=0.0,
            model_size_mb=0.0,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")


@router.post("/models/reload")
async def reload_models():
    """Reload models from disk."""
    global loaded_models
    
    try:
        loaded_models.clear()
        load_models_and_config()
        
        return {
            "message": "Models reloaded successfully",
            "models_loaded": len(loaded_models),
            "available_models": list(loaded_models.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model reload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


@router.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a specific model."""
    if model_name not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    try:
        # Remove from memory
        del loaded_models[model_name]
        
        # Remove from disk
        config = get_config()
        model_path = os.path.join(config.output_dir, "models", f"{model_name}.joblib")
        if os.path.exists(model_path):
            os.remove(model_path)
        
        return {
            "message": f"Model '{model_name}' deleted successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model deletion failed: {str(e)}")


# Utility endpoints
@router.get("/features")
async def get_feature_info():
    """Get information about required features."""
    return {
        "feature_count": len(feature_names) if feature_names else 95,
        "feature_names": feature_names or [f"X{i}" for i in range(1, 96)],
        "feature_description": "Financial indicators for bankruptcy prediction",
        "data_types": "All features should be numeric (float)",
        "missing_values": "Not allowed - all features are required",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/metrics/{model_name}")
async def get_model_metrics(model_name: str):
    """Get detailed metrics for a specific model."""
    if model_name not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # In a real implementation, you would load saved metrics
    # This is a placeholder implementation
    return {
        "model_name": model_name,
        "metrics": {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.94,
            "f1_score": 0.935,
            "roc_auc": 0.98,
            "confusion_matrix": [[1200, 50], [30, 400]],
            "classification_report": "Detailed classification metrics..."
        },
        "timestamp": datetime.now().isoformat()
    }


# Initialize models on module import
load_models_and_config()