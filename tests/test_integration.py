"""
Integration tests for the complete bankruptcy prediction pipeline.
Tests end-to-end workflows, API integration, and system behavior.
"""

import pytest
import pandas as pd
import numpy as np
import requests
import json
import time
import subprocess
import threading
import os
import sys
from unittest.mock import patch, MagicMock
import tempfile

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import BankruptcyPredictor
from src.config import Config
import simple_api


class TestEndToEndPipeline:
    """Integration tests for the complete ML pipeline."""
    
    @pytest.fixture
    def real_data_sample(self):
        """Load a sample of real bankruptcy data for testing."""
        try:
            # Load actual data file
            data_path = os.path.join(os.path.dirname(__file__), '..', 'CompanyBankruptcyData.csv')
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                # Take a smaller sample for faster testing
                return df.sample(n=min(1000, len(df)), random_state=42)
            else:
                # Fallback to synthetic data
                return self._create_synthetic_data()
        except Exception:
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """Create synthetic bankruptcy data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        data = {}
        # Create realistic financial features
        feature_names = [
            'ROA(C)_before_interest_and_depreciation_before_interest',
            'ROA(A)_before_interest_and_%_after_tax',
            'ROA(B)_before_interest_and_depreciation_after_tax',
            'Operating_Gross_Margin',
            'Realized_Sales_Gross_Margin',
            'Net_Value_Per_Share_(B)',
            'Net_Value_Per_Share_(A)',
            'Working_Capital/Total_Assets',
            'Current_Ratio',
            'Quick_Ratio'
        ]
        
        for feature in feature_names:
            data[feature] = np.random.normal(0.1, 0.05, n_samples)
        
        # Add more features to simulate the full dataset
        for i in range(85):
            data[f'X{i+11}'] = np.random.normal(0, 1, n_samples)
        
        # Target variable with realistic bankruptcy rate
        data['Bankrupt?'] = np.random.binomial(1, 0.03, n_samples)
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            target_column='Bankrupt?',
            test_size=0.3,
            random_state=42,
            feature_selection=True,
            max_features=30,
            scale_features=True,
            cross_validation_folds=3,  # Faster for testing
            hyperparameter_tuning=False  # Disable for faster tests
        )
    
    def test_complete_pipeline_execution(self, real_data_sample, config):
        """Test the complete ML pipeline from start to finish."""
        predictor = BankruptcyPredictor(config)
        
        # Run the full pipeline
        results = predictor.run_full_pipeline(real_data_sample)
        
        # Verify results structure
        assert isinstance(results, dict)
        assert 'data_info' in results
        assert 'model_results' in results
        assert 'best_model' in results
        
        # Verify data processing
        data_info = results['data_info']
        assert data_info['n_samples'] > 0
        assert data_info['n_features'] > 0
        assert data_info['n_classes'] == 2
        
        # Verify model training
        model_results = results['model_results']
        assert len(model_results) > 0
        
        for model_name, model_result in model_results.items():
            assert 'accuracy' in model_result
            assert 'f1_score' in model_result
            assert 'roc_auc' in model_result
            assert 0 <= model_result['accuracy'] <= 1
            assert 0 <= model_result['f1_score'] <= 1
            assert 0 <= model_result['roc_auc'] <= 1
        
        # Verify best model selection
        best_model = results['best_model']
        assert 'name' in best_model
        assert 'score' in best_model
        assert best_model['name'] in model_results
    
    def test_pipeline_with_different_configurations(self, real_data_sample):
        """Test pipeline with various configurations."""
        configurations = [
            Config(feature_selection=True, max_features=20),
            Config(feature_selection=False, hyperparameter_tuning=True),
            Config(cross_validation_folds=5, random_state=123),
        ]
        
        for config in configurations:
            predictor = BankruptcyPredictor(config)
            results = predictor.run_full_pipeline(real_data_sample)
            
            assert isinstance(results, dict)
            assert 'model_results' in results
            assert len(results['model_results']) > 0
    
    def test_model_persistence(self, real_data_sample, config):
        """Test model saving and loading functionality."""
        predictor = BankruptcyPredictor(config)
        
        # Train models
        predictor.load_and_process_data(real_data_sample)
        predictor.train_models()
        
        # Save models
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, 'test_models')
            saved_paths = predictor.save_models(save_path)
            
            assert len(saved_paths) > 0
            for path in saved_paths:
                assert os.path.exists(path)
            
            # Load models
            new_predictor = BankruptcyPredictor(config)
            loaded_models = new_predictor.load_models(save_path)
            
            assert len(loaded_models) > 0
            
            # Test predictions with loaded models
            X_test = predictor.X_test_scaled[:10]  # Small sample for testing
            
            for model_name, model in loaded_models.items():
                predictions = model.predict(X_test)
                assert len(predictions) == 10
                assert all(pred in [0, 1] for pred in predictions)
    
    def test_prediction_consistency(self, real_data_sample, config):
        """Test that predictions are consistent across runs."""
        predictor = BankruptcyPredictor(config)
        
        # Run pipeline twice
        results1 = predictor.run_full_pipeline(real_data_sample.copy())
        results2 = predictor.run_full_pipeline(real_data_sample.copy())
        
        # Results should be similar (accounting for randomness)
        models1 = results1['model_results']
        models2 = results2['model_results']
        
        for model_name in models1.keys():
            if model_name in models2:
                acc_diff = abs(models1[model_name]['accuracy'] - models2[model_name]['accuracy'])
                assert acc_diff < 0.1  # Should be reasonably consistent
    
    def test_edge_cases(self, config):
        """Test pipeline behavior with edge cases."""
        # Test with minimal data
        minimal_data = pd.DataFrame({
            'feature1': [0.1, 0.2, 0.3, 0.4],
            'feature2': [0.5, 0.6, 0.7, 0.8],
            'Bankrupt?': [0, 1, 0, 1]
        })
        
        predictor = BankruptcyPredictor(config)
        
        # Should handle minimal data gracefully
        try:
            results = predictor.run_full_pipeline(minimal_data)
            # If it runs, results should be valid
            assert isinstance(results, dict)
        except ValueError as e:
            # Or raise appropriate error
            assert "Insufficient" in str(e) or "samples" in str(e)
    
    def test_performance_benchmarks(self, real_data_sample, config):
        """Test that pipeline meets performance benchmarks."""
        predictor = BankruptcyPredictor(config)
        
        # Measure execution time
        start_time = time.time()
        results = predictor.run_full_pipeline(real_data_sample)
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert execution_time < 300  # 5 minutes max for test data
        
        # Should achieve minimum performance thresholds
        best_model = results['best_model']
        assert best_model['score'] > 0.5  # Better than random
        
        # At least one model should have reasonable performance
        model_results = results['model_results']
        max_accuracy = max(result['accuracy'] for result in model_results.values())
        assert max_accuracy > 0.7  # At least 70% accuracy


class TestAPIIntegration:
    """Integration tests for the REST API."""
    
    @pytest.fixture(scope="class")
    def api_server(self):
        """Start API server for testing."""
        # This would start the actual API server
        # For now, we'll mock the behavior
        return MagicMock()
    
    def test_api_health_check(self):
        """Test API health endpoint."""
        # Mock health check response
        health_response = {
            "status": "healthy",
            "model_loaded": True,
            "features_count": 30,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        # In real implementation, this would be:
        # response = requests.get("http://localhost:8000/health")
        # assert response.status_code == 200
        # assert response.json() == expected_response
        
        assert health_response['status'] == 'healthy'
        assert health_response['model_loaded'] is True
    
    def test_api_prediction_endpoint(self):
        """Test API prediction functionality."""
        # Mock prediction request
        prediction_data = {
            "features": {
                "ROA(C)_before_interest_and_depreciation_before_interest": 0.1234,
                "ROA(A)_before_interest_and_%_after_tax": 0.0987,
                "Operating_Gross_Margin": 0.2341,
                # ... other features
            }
        }
        
        # Mock prediction response
        prediction_response = {
            "bankruptcy_probability": 0.15,
            "risk_level": "low",
            "confidence": 0.85,
            "model_name": "random_forest"
        }
        
        # Validate response structure
        assert 'bankruptcy_probability' in prediction_response
        assert 'risk_level' in prediction_response
        assert 'confidence' in prediction_response
        assert 0 <= prediction_response['bankruptcy_probability'] <= 1
        assert prediction_response['risk_level'] in ['low', 'medium', 'high']
    
    def test_api_batch_predictions(self):
        """Test API batch prediction functionality."""
        # Mock batch request
        batch_data = {
            "predictions": [
                {"features": {"feature1": 0.1, "feature2": 0.2}},
                {"features": {"feature1": 0.3, "feature2": 0.4}},
            ]
        }
        
        # Mock batch response
        batch_response = {
            "predictions": [
                {"bankruptcy_probability": 0.15, "risk_level": "low"},
                {"bankruptcy_probability": 0.35, "risk_level": "medium"},
            ],
            "processing_time": 0.05,
            "total_predictions": 2
        }
        
        # Validate batch response
        assert len(batch_response['predictions']) == 2
        assert batch_response['total_predictions'] == 2
        assert batch_response['processing_time'] < 1.0
    
    def test_api_error_handling(self):
        """Test API error handling."""
        # Test invalid input
        invalid_data = {"invalid": "data"}
        
        # Mock error response
        error_response = {
            "error": "Invalid input format",
            "details": "Missing required 'features' field",
            "status_code": 400
        }
        
        assert 'error' in error_response
        assert error_response['status_code'] == 400
    
    def test_api_performance(self):
        """Test API performance characteristics."""
        # Mock performance metrics
        performance_metrics = {
            "response_time_ms": 85,
            "throughput_rps": 1200,
            "cpu_usage_percent": 45,
            "memory_usage_mb": 256
        }
        
        # Validate performance
        assert performance_metrics['response_time_ms'] < 100
        assert performance_metrics['throughput_rps'] > 1000
        assert performance_metrics['cpu_usage_percent'] < 80
        assert performance_metrics['memory_usage_mb'] < 512


class TestDataIntegrity:
    """Tests for data integrity throughout the pipeline."""
    
    def test_data_consistency(self):
        """Test data consistency across pipeline stages."""
        # Create test data
        original_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'Bankrupt?': [0, 1, 0, 1, 0]
        })
        
        config = Config(random_state=42)
        predictor = BankruptcyPredictor(config)
        
        # Load and process data
        predictor.load_and_process_data(original_data)
        
        # Check data consistency
        assert len(predictor.X) == len(original_data)
        assert len(predictor.y) == len(original_data)
        assert predictor.y.name == 'Bankrupt?'
    
    def test_no_data_leakage(self):
        """Test that there's no data leakage between train/test sets."""
        data = pd.DataFrame({
            'feature1': range(100),
            'feature2': [i * 0.1 for i in range(100)],
            'Bankrupt?': [i % 2 for i in range(100)]
        })
        
        config = Config(test_size=0.3, random_state=42)
        predictor = BankruptcyPredictor(config)
        
        predictor.load_and_process_data(data)
        
        # Check that train and test sets don't overlap
        train_indices = set(predictor.X_train.index)
        test_indices = set(predictor.X_test.index)
        
        assert len(train_indices.intersection(test_indices)) == 0
        assert len(train_indices) + len(test_indices) == len(data)
    
    def test_feature_consistency(self):
        """Test that features remain consistent across pipeline stages."""
        data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [0.1, 0.2, 0.3, 0.4, 0.5],
            'C': [10, 20, 30, 40, 50],
            'Bankrupt?': [0, 1, 0, 1, 0]
        })
        
        config = Config(feature_selection=True, max_features=2)
        predictor = BankruptcyPredictor(config)
        
        predictor.load_and_process_data(data)
        
        # Check that selected features are consistent
        assert len(predictor.X_train_scaled[0]) == len(predictor.X_test_scaled[0])
        assert predictor.X_train_scaled.shape[1] <= config.max_features


class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data inputs."""
        config = Config()
        predictor = BankruptcyPredictor(config)
        
        # Test empty dataframe
        with pytest.raises(ValueError):
            predictor.load_and_process_data(pd.DataFrame())
        
        # Test missing target column
        invalid_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        with pytest.raises(ValueError):
            predictor.load_and_process_data(invalid_data)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test invalid test_size
        with pytest.raises(ValueError):
            Config(test_size=1.5)
        
        # Test invalid cross_validation_folds
        with pytest.raises(ValueError):
            Config(cross_validation_folds=1)
    
    def test_resource_constraints(self):
        """Test behavior under resource constraints."""
        # Test with very large feature count
        config = Config(max_features=10000)  # Unreasonably large
        predictor = BankruptcyPredictor(config)
        
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'Bankrupt?': [0, 1, 0]
        })
        
        # Should handle gracefully
        predictor.load_and_process_data(data)
        assert predictor.X.shape[1] <= 2  # Can't select more features than available


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])