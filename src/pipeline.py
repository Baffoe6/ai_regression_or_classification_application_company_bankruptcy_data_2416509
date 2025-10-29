"""
Main training and evaluation pipeline for bankruptcy prediction.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import json
import os

from .config import Config, DEFAULT_MODEL_CONFIGS, setup_directories
from .utils import setup_logging, get_logger
from .data import DataProcessor
from .models import ModelFactory, ModelEvaluator, BaseModel
from .visualization import Visualizer


class BankruptcyPredictor:
    """Main class for bankruptcy prediction pipeline."""
    
    def __init__(self, config: Config = None):
        """Initialize the predictor with configuration."""
        self.config = config or Config()
        setup_directories(self.config)
        
        # Setup logging
        self.logger = setup_logging(self.config)
        
        # Initialize components
        self.data_processor = DataProcessor(self.config)
        self.visualizer = Visualizer(self.config)
        self.model_evaluator = ModelEvaluator()
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.exploration_stats = None
        
        # Model storage
        self.models = {}
        self.results = []
        
        self.logger.info("BankruptcyPredictor initialized")
    
    def load_and_process_data(self, data_path: str = None) -> None:
        """Load and process the data."""
        self.logger.info("Starting data loading and processing")
        
        # Process data
        self.X_train, self.X_test, self.y_train, self.y_test, self.exploration_stats = \
            self.data_processor.process_full_pipeline(data_path)
        
        self.logger.info(f"Data processing completed. Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
    
    def visualize_data_exploration(self) -> None:
        """Create data exploration visualizations."""
        if self.exploration_stats is None:
            self.logger.warning("No exploration stats available. Run load_and_process_data first.")
            return
        
        # Load original data for visualization
        df = self.data_processor.load_data(self.config.data_path)
        # Plot target distribution
        self.visualizer.plot_target_distribution(df, target_column=self.config.target_column)
    
    def train_models(self, model_types: List[str] = None) -> None:
        """Train all configured models."""
        self.logger.info("Starting model training")
        
        if self.X_train is None:
            raise ValueError("Data must be loaded first. Call load_and_process_data().")
        
        model_types = model_types or ['logistic_regression', 'random_forest', 'neural_network']
        
        for model_type in model_types:
            if model_type not in DEFAULT_MODEL_CONFIGS:
                self.logger.warning(f"Unknown model type: {model_type}")
                continue
            
            model_configs = DEFAULT_MODEL_CONFIGS[model_type]
            
            for config in model_configs:
                if not config.enabled:
                    continue
                
                try:
                    # Create model
                    if model_type == 'neural_network':
                        model = ModelFactory.create_model(
                            model_type, config, 
                            input_dim=self.X_train.shape[1],
                            early_stopping_patience=self.config.early_stopping_patience
                        )
                    else:
                        model = ModelFactory.create_model(model_type, config)
                    
                    # Train model
                    model.fit(self.X_train, self.y_train)
                    
                    # Store model
                    self.models[config.name] = model
                    
                    # Save model if configured
                    if self.config.save_models:
                        model_path = os.path.join(self.config.output_dir, "models", f"{config.name}.joblib")
                        model.save_model(model_path)
                    
                    self.logger.info(f"Successfully trained {config.name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to train {config.name}: {str(e)}")
        
        self.logger.info("Model training completed")
    
    def evaluate_models(self) -> None:
        """Evaluate all trained models."""
        self.logger.info("Starting model evaluation")
        
        if not self.models:
            raise ValueError("No models trained. Call train_models() first.")
        
        self.results = []
        roc_data = []
        confusion_data = []
        
        for model_name, model in self.models.items():
            try:
                # Make predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)
                
                # Evaluate model
                results = self.model_evaluator.evaluate_model(
                    self.y_test, y_pred, y_pred_proba, model_name
                )
                
                self.results.append(results)
                
                # Store data for visualizations
                roc_data.append((model_name, self.y_test, y_pred_proba))
                confusion_data.append((model_name, self.y_test, y_pred))
                
                # Cross-validation (if applicable)
                cv_results = self.model_evaluator.cross_validate_model(
                    model, self.X_train, self.y_train, self.config.cross_validation_folds
                )
                
                if cv_results:
                    results.update(cv_results)
                
                self.logger.info(f"Evaluated {model_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_name}: {str(e)}")
        
        # Create visualizations
        # Convert results to dict format for visualization
        results_dict = {r['model_name']: r for r in self.results}
        self.visualizer.plot_model_performance_comparison(results_dict)
        
        # Note: ROC curves and confusion matrices plotting requires additional data structure
        # These visualizations can be added later if needed
        
        self.logger.info("Model evaluation completed")
    
    def analyze_feature_importance(self) -> None:
        """Analyze feature importance for applicable models."""
        self.logger.info("Starting feature importance analysis")
        
        feature_names = self.data_processor.feature_names
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'get_feature_importance'):
                    importance = model.get_feature_importance()
                    if importance is not None:
                        self.visualizer.plot_feature_importance(
                            feature_names, importance, model_name
                        )
                
            except Exception as e:
                self.logger.error(f"Failed to analyze feature importance for {model_name}: {str(e)}")
    
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, Dict[str, Any]]:
        """Get the best performing model based on a specific metric."""
        if not self.results:
            raise ValueError("No evaluation results available. Call evaluate_models() first.")
        
        best_result = max(self.results, key=lambda x: x.get(metric, 0))
        best_model_name = best_result['model_name']
        
        self.logger.info(f"Best model by {metric}: {best_model_name} ({best_result[metric]:.4f})")
        
        return best_model_name, best_result
    
    def save_results(self, filepath: str = None) -> None:
        """Save evaluation results to JSON file."""
        if not self.results:
            self.logger.warning("No results to save")
            return
        
        filepath = filepath or os.path.join(self.config.output_dir, "reports", "evaluation_results.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare results for JSON serialization
        serializable_results = []
        for result in self.results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, (list, np.ndarray)):
                    serializable_result[key] = list(value) if hasattr(value, '__iter__') else [value]
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
        
        with open(filepath, 'w') as f:
            json.dump({
                'config': self.config.__dict__,
                'exploration_stats': self.exploration_stats,
                'results': serializable_results
            }, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive text report."""
        if not self.results:
            return "No results available. Run evaluation first."
        
        report = []
        report.append("=" * 60)
        report.append("BANKRUPTCY PREDICTION MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Data summary
        if self.exploration_stats:
            report.append("DATA SUMMARY:")
            report.append(f"  Total samples: {self.exploration_stats['shape'][0]}")
            report.append(f"  Features: {self.exploration_stats['feature_count']}")
            report.append(f"  Missing values: {self.exploration_stats['missing_values']}")
            report.append(f"  Target distribution: {self.exploration_stats['target_distribution']}")
            report.append("")
        
        # Model results
        report.append("MODEL PERFORMANCE:")
        report.append("-" * 40)
        
        # Sort results by F1-score
        sorted_results = sorted(self.results, key=lambda x: x.get('f1_score', 0), reverse=True)
        
        for result in sorted_results:
            report.append(f"Model: {result['model_name']}")
            report.append(f"  Accuracy:  {result.get('accuracy', 0):.4f}")
            report.append(f"  Precision: {result.get('precision', 0):.4f}")
            report.append(f"  Recall:    {result.get('recall', 0):.4f}")
            report.append(f"  F1-Score:  {result.get('f1_score', 0):.4f}")
            report.append(f"  ROC-AUC:   {result.get('roc_auc', 0):.4f}")
            
            if 'cv_mean_accuracy' in result:
                report.append(f"  CV Accuracy: {result['cv_mean_accuracy']:.4f} ± {result['cv_std_accuracy']:.4f}")
            
            report.append("")
        
        # Best model summary
        best_model_name, best_result = self.get_best_model('f1_score')
        report.append("BEST MODEL:")
        report.append(f"  {best_model_name} (F1-Score: {best_result['f1_score']:.4f})")
        report.append("")
        
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = os.path.join(self.config.output_dir, "reports", "evaluation_report.txt")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        self.logger.info(f"Report saved to {report_path}")
        
        return report_text
    
    def run_full_pipeline(self, data_path: str = None) -> str:
        """Run the complete pipeline from data loading to evaluation."""
        self.logger.info("Starting full pipeline")
        
        try:
            # Load and process data
            self.load_and_process_data(data_path)
            
            # Visualize data exploration
            self.visualize_data_exploration()
            
            # Train models
            self.train_models()
            
            # Evaluate models
            self.evaluate_models()
            
            # Analyze feature importance
            self.analyze_feature_importance()
            
            # Save results
            self.save_results()
            
            # Generate report
            report = self.generate_report()
            
            self.logger.info("Full pipeline completed successfully")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Main function to run the bankruptcy prediction pipeline."""
    # Load configuration
    config = Config()
    
    # Create predictor
    predictor = BankruptcyPredictor(config)
    
    # Run full pipeline
    report = predictor.run_full_pipeline()
    
    print(report)


if __name__ == "__main__":
    main()