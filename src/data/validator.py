"""
Data validation utilities and schema enforcement.
Provides comprehensive data validation, quality checks, and schema validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
import json
import warnings

logger = logging.getLogger(__name__)


class DataSchema:
    """
    Define and enforce data schemas for bankruptcy prediction.
    """
    
    def __init__(self):
        self.required_columns = []
        self.column_types = {}
        self.value_ranges = {}
        self.categorical_values = {}
        self.nullable_columns = set()
        
    def add_column(self, name: str, dtype: str, nullable: bool = True, 
                   value_range: Optional[Tuple[float, float]] = None,
                   allowed_values: Optional[List[Any]] = None):
        """
        Add a column definition to the schema.
        
        Args:
            name: Column name
            dtype: Data type ('int', 'float', 'string', 'category')
            nullable: Whether column can contain null values
            value_range: Tuple of (min, max) for numeric columns
            allowed_values: List of allowed values for categorical columns
        """
        self.required_columns.append(name)
        self.column_types[name] = dtype
        
        if nullable:
            self.nullable_columns.add(name)
            
        if value_range:
            self.value_ranges[name] = value_range
            
        if allowed_values:
            self.categorical_values[name] = allowed_values
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame against schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check extra columns
        extra_columns = set(df.columns) - set(self.required_columns)
        if extra_columns:
            errors.append(f"Unexpected columns: {extra_columns}")
        
        # Validate each column
        for column in self.required_columns:
            if column not in df.columns:
                continue
                
            # Check nullability
            if column not in self.nullable_columns and df[column].isnull().any():
                errors.append(f"Column '{column}' contains null values but is not nullable")
            
            # Check data types
            expected_dtype = self.column_types[column]
            if expected_dtype in ['int', 'float']:
                if not pd.api.types.is_numeric_dtype(df[column]):
                    errors.append(f"Column '{column}' should be numeric but is {df[column].dtype}")
            elif expected_dtype == 'string':
                if not pd.api.types.is_string_dtype(df[column]) and not pd.api.types.is_object_dtype(df[column]):
                    errors.append(f"Column '{column}' should be string but is {df[column].dtype}")
            
            # Check value ranges
            if column in self.value_ranges and pd.api.types.is_numeric_dtype(df[column]):
                min_val, max_val = self.value_ranges[column]
                out_of_range = (df[column] < min_val) | (df[column] > max_val)
                if out_of_range.any():
                    errors.append(f"Column '{column}' has values outside range [{min_val}, {max_val}]")
            
            # Check categorical values
            if column in self.categorical_values:
                invalid_values = set(df[column].dropna().unique()) - set(self.categorical_values[column])
                if invalid_values:
                    errors.append(f"Column '{column}' has invalid values: {invalid_values}")
        
        return len(errors) == 0, errors


class BankruptcyDataSchema(DataSchema):
    """
    Predefined schema for bankruptcy prediction data.
    """
    
    def __init__(self):
        super().__init__()
        self._define_bankruptcy_schema()
    
    def _define_bankruptcy_schema(self):
        """Define the standard bankruptcy prediction schema."""
        # Target variable
        self.add_column('Bankrupt?', 'int', nullable=False, 
                       value_range=(0, 1), allowed_values=[0, 1])
        
        # Key financial ratios
        financial_ratios = [
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
        
        for ratio in financial_ratios:
            self.add_column(ratio, 'float', nullable=True, value_range=(-10, 10))


class DataQualityAnalyzer:
    """
    Comprehensive data quality analysis and reporting.
    """
    
    def __init__(self):
        self.quality_checks = {
            'completeness': self._check_completeness,
            'uniqueness': self._check_uniqueness,
            'validity': self._check_validity,
            'consistency': self._check_consistency,
            'accuracy': self._check_accuracy
        }
    
    def analyze(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """
        Perform comprehensive data quality analysis.
        
        Args:
            df: DataFrame to analyze
            target_column: Name of target column
            
        Returns:
            Dictionary with quality analysis results
        """
        logger.info("Starting data quality analysis")
        
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': self._get_dataset_info(df),
            'quality_scores': {},
            'issues_found': [],
            'recommendations': []
        }
        
        # Run quality checks
        for check_name, check_function in self.quality_checks.items():
            try:
                score, issues, recommendations = check_function(df, target_column)
                analysis_results['quality_scores'][check_name] = score
                analysis_results['issues_found'].extend(issues)
                analysis_results['recommendations'].extend(recommendations)
            except Exception as e:
                logger.error(f"Error in {check_name} check: {str(e)}")
                analysis_results['quality_scores'][check_name] = 0.0
        
        # Calculate overall quality score
        overall_score = np.mean(list(analysis_results['quality_scores'].values()))
        analysis_results['overall_quality_score'] = overall_score
        
        logger.info(f"Data quality analysis completed. Overall score: {overall_score:.2f}")
        
        return analysis_results
    
    def _get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information."""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns)
        }
    
    def _check_completeness(self, df: pd.DataFrame, target_column: str = None) -> Tuple[float, List[str], List[str]]:
        """Check data completeness (missing values)."""
        issues = []
        recommendations = []
        
        # Calculate completeness score
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness_score = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        
        # Check for high missing value columns
        missing_by_column = df.isnull().sum() / len(df)
        high_missing_columns = missing_by_column[missing_by_column > 0.5].index.tolist()
        
        if high_missing_columns:
            issues.append(f"Columns with >50% missing values: {high_missing_columns}")
            recommendations.append("Consider removing or imputing high missing value columns")
        
        # Check target column completeness
        if target_column and target_column in df.columns:
            target_missing = df[target_column].isnull().sum()
            if target_missing > 0:
                issues.append(f"Target column '{target_column}' has {target_missing} missing values")
                recommendations.append("Remove rows with missing target values")
        
        return completeness_score, issues, recommendations
    
    def _check_uniqueness(self, df: pd.DataFrame, target_column: str = None) -> Tuple[float, List[str], List[str]]:
        """Check data uniqueness (duplicates)."""
        issues = []
        recommendations = []
        
        # Calculate uniqueness score
        duplicate_rows = df.duplicated().sum()
        uniqueness_score = 1 - (duplicate_rows / len(df)) if len(df) > 0 else 0
        
        if duplicate_rows > 0:
            issues.append(f"Found {duplicate_rows} duplicate rows")
            recommendations.append("Remove or investigate duplicate rows")
        
        # Check for columns with low uniqueness
        for column in df.columns:
            unique_ratio = df[column].nunique() / len(df)
            if unique_ratio < 0.01 and column != target_column:
                issues.append(f"Column '{column}' has very low uniqueness ({unique_ratio:.1%})")
                recommendations.append(f"Consider removing low-variance column '{column}'")
        
        return uniqueness_score, issues, recommendations
    
    def _check_validity(self, df: pd.DataFrame, target_column: str = None) -> Tuple[float, List[str], List[str]]:
        """Check data validity (format, ranges, types)."""
        issues = []
        recommendations = []
        validity_issues = 0
        total_checks = 0
        
        # Check for infinite values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            total_checks += 1
            if np.isinf(df[column]).any():
                validity_issues += 1
                issues.append(f"Column '{column}' contains infinite values")
                recommendations.append(f"Handle infinite values in '{column}'")
        
        # Check for negative values where they shouldn't exist
        potential_ratio_columns = [col for col in df.columns if any(keyword in col.lower() 
                                 for keyword in ['ratio', 'margin', 'percentage', '%'])]
        
        for column in potential_ratio_columns:
            if column in numeric_columns:
                total_checks += 1
                negative_count = (df[column] < 0).sum()
                if negative_count > len(df) * 0.1:  # More than 10% negative
                    validity_issues += 1
                    issues.append(f"Column '{column}' has many negative values ({negative_count})")
                    recommendations.append(f"Investigate negative values in ratio column '{column}'")
        
        # Check target variable validity
        if target_column and target_column in df.columns:
            total_checks += 1
            unique_targets = df[target_column].dropna().unique()
            if len(unique_targets) < 2:
                validity_issues += 1
                issues.append(f"Target variable has only {len(unique_targets)} unique values")
                recommendations.append("Ensure target variable has at least 2 classes")
            elif not all(val in [0, 1] for val in unique_targets if pd.notna(val)):
                validity_issues += 1
                issues.append("Target variable contains non-binary values")
                recommendations.append("Ensure target variable is properly encoded as 0/1")
        
        validity_score = 1 - (validity_issues / total_checks) if total_checks > 0 else 1
        
        return validity_score, issues, recommendations
    
    def _check_consistency(self, df: pd.DataFrame, target_column: str = None) -> Tuple[float, List[str], List[str]]:
        """Check data consistency across columns."""
        issues = []
        recommendations = []
        consistency_issues = 0
        total_checks = 0
        
        # Check for consistent data types within columns
        for column in df.columns:
            total_checks += 1
            if df[column].dtype == 'object':
                # Check if supposed to be numeric
                try:
                    pd.to_numeric(df[column], errors='raise')
                except (ValueError, TypeError):
                    pass  # It's actually categorical
                else:
                    consistency_issues += 1
                    issues.append(f"Column '{column}' appears to be numeric but stored as object")
                    recommendations.append(f"Convert '{column}' to appropriate numeric type")
        
        # Check for outliers that might indicate inconsistency
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            if column != target_column:
                total_checks += 1
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[column] < Q1 - 3 * IQR) | (df[column] > Q3 + 3 * IQR)).sum()
                
                if outliers > len(df) * 0.05:  # More than 5% outliers
                    consistency_issues += 1
                    issues.append(f"Column '{column}' has many outliers ({outliers})")
                    recommendations.append(f"Investigate outliers in '{column}'")
        
        consistency_score = 1 - (consistency_issues / total_checks) if total_checks > 0 else 1
        
        return consistency_score, issues, recommendations
    
    def _check_accuracy(self, df: pd.DataFrame, target_column: str = None) -> Tuple[float, List[str], List[str]]:
        """Check data accuracy based on domain knowledge."""
        issues = []
        recommendations = []
        accuracy_issues = 0
        total_checks = 0
        
        # Check for reasonable ranges in financial ratios
        financial_checks = {
            'current_ratio': (0, 50),  # Current ratio should be reasonable
            'roa': (-1, 1),            # ROA should be between -100% and 100%
            'margin': (-1, 1),         # Margins should be between -100% and 100%
            'ratio': (0, 100)          # Most ratios should be positive and reasonable
        }
        
        for column in df.columns:
            column_lower = column.lower()
            for check_type, (min_val, max_val) in financial_checks.items():
                if check_type in column_lower and pd.api.types.is_numeric_dtype(df[column]):
                    total_checks += 1
                    out_of_range = ((df[column] < min_val) | (df[column] > max_val)).sum()
                    
                    if out_of_range > len(df) * 0.01:  # More than 1% out of range
                        accuracy_issues += 1
                        issues.append(f"Column '{column}' has values outside expected range [{min_val}, {max_val}]")
                        recommendations.append(f"Validate data source for '{column}'")
                    break
        
        # Check class balance for target variable
        if target_column and target_column in df.columns:
            total_checks += 1
            class_counts = df[target_column].value_counts()
            if len(class_counts) >= 2:
                minority_ratio = class_counts.min() / class_counts.sum()
                if minority_ratio < 0.01:  # Less than 1% minority class
                    accuracy_issues += 1
                    issues.append(f"Severe class imbalance: minority class is {minority_ratio:.1%}")
                    recommendations.append("Consider data balancing techniques")
        
        accuracy_score = 1 - (accuracy_issues / total_checks) if total_checks > 0 else 1
        
        return accuracy_score, issues, recommendations
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a human-readable quality report."""
        report = []
        report.append("=" * 60)
        report.append("DATA QUALITY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {analysis_results['timestamp']}")
        report.append()
        
        # Dataset info
        info = analysis_results['dataset_info']
        report.append("DATASET INFORMATION:")
        report.append(f"  Rows: {info['rows']:,}")
        report.append(f"  Columns: {info['columns']}")
        report.append(f"  Memory Usage: {info['memory_usage_mb']:.2f} MB")
        report.append(f"  Numeric Columns: {info['numeric_columns']}")
        report.append(f"  Categorical Columns: {info['categorical_columns']}")
        report.append()
        
        # Quality scores
        report.append("QUALITY SCORES:")
        for metric, score in analysis_results['quality_scores'].items():
            status = "GOOD" if score >= 0.8 else "FAIR" if score >= 0.6 else "POOR"
            report.append(f"  {metric.title()}: {score:.2f} ({status})")
        
        overall_score = analysis_results['overall_quality_score']
        overall_status = "GOOD" if overall_score >= 0.8 else "FAIR" if overall_score >= 0.6 else "POOR"
        report.append(f"  Overall Quality: {overall_score:.2f} ({overall_status})")
        report.append()
        
        # Issues
        if analysis_results['issues_found']:
            report.append("ISSUES FOUND:")
            for i, issue in enumerate(analysis_results['issues_found'], 1):
                report.append(f"  {i}. {issue}")
            report.append()
        
        # Recommendations
        if analysis_results['recommendations']:
            report.append("RECOMMENDATIONS:")
            for i, rec in enumerate(analysis_results['recommendations'], 1):
                report.append(f"  {i}. {rec}")
        
        report.append("=" * 60)
        
        return '\n'.join(report)


def validate_bankruptcy_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Quick validation function for bankruptcy prediction data.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    schema = BankruptcyDataSchema()
    return schema.validate(df)


def analyze_data_quality(df: pd.DataFrame, target_column: str = 'Bankrupt?') -> Dict[str, Any]:
    """
    Quick data quality analysis function.
    
    Args:
        df: DataFrame to analyze
        target_column: Name of target column
        
    Returns:
        Quality analysis results
    """
    analyzer = DataQualityAnalyzer()
    return analyzer.analyze(df, target_column)