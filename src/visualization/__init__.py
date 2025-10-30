"""
Visualization utilities for bankruptcy prediction analysis.
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ..config import Config
from ..utils import get_logger

logger = get_logger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set style for matplotlib
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class BankruptcyVisualizer:
    """Comprehensive visualization tools for bankruptcy prediction analysis."""

    def __init__(self, config: Config = None, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.

        Args:
            config: Configuration object
            figsize: Default figure size for matplotlib plots
        """
        self.config = config
        self.figsize = figsize
        self.colors = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "success": "#2ca02c",
            "danger": "#d62728",
            "warning": "#ff7f0e",
            "info": "#17a2b8",
        }

        if config:
            self.output_dir = os.path.join(config.output_dir, "plots")
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.output_dir = "plots"

    def plot_target_distribution(
        self,
        df: pd.DataFrame,
        target_column: str = "Bankrupt?",
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Plot the distribution of the target variable.

        Args:
            df: DataFrame containing the data
            target_column: Name of the target column
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        target_counts = df[target_column].value_counts()
        target_percentages = df[target_column].value_counts(normalize=True) * 100

        fig = go.Figure(
            data=[
                go.Bar(
                    x=["Not Bankrupt", "Bankrupt"],
                    y=target_counts.values,
                    text=[
                        f"{count}<br>({percentage:.1f}%)"
                        for count, percentage in zip(
                            target_counts.values, target_percentages.values
                        )
                    ],
                    textposition="auto",
                    marker_color=[self.colors["success"], self.colors["danger"]],
                )
            ]
        )

        fig.update_layout(
            title={
                "text": "Distribution of Bankruptcy Status",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20},
            },
            xaxis_title="Bankruptcy Status",
            yaxis_title="Number of Companies",
            template="plotly_white",
            height=500,
        )

        if save_path:
            fig.write_html(save_path)
        elif self.config and self.config.save_plots:
            save_path = os.path.join(self.output_dir, "target_distribution.html")
            fig.write_html(save_path)
            logger.info(f"Target distribution plot saved to {save_path}")

        return fig

    def plot_feature_correlation_heatmap(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Plot correlation heatmap of features.

        Args:
            df: DataFrame containing the data
            features: List of features to include (if None, uses top 20 by variance)
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        if features is None:
            # Select top 20 features by variance
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "Bankrupt?" in numeric_cols:
                numeric_cols.remove("Bankrupt?")

            variances = df[numeric_cols].var().sort_values(ascending=False)
            features = variances.head(20).index.tolist()

        corr_matrix = df[features].corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu",
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 8},
                hoverongaps=False,
            )
        )

        fig.update_layout(
            title={
                "text": f"Feature Correlation Heatmap (Top {len(features)} Features)",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 16},
            },
            template="plotly_white",
            height=600,
            width=800,
        )

        if save_path:
            fig.write_html(save_path)
        elif self.config and self.config.save_plots:
            save_path = os.path.join(self.output_dir, "correlation_heatmap.html")
            fig.write_html(save_path)
            logger.info(f"Correlation heatmap saved to {save_path}")

        return fig

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_scores: np.ndarray,
        top_n: int = 20,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Plot feature importance scores.

        Args:
            feature_names: List of feature names
            importance_scores: Array of importance scores
            top_n: Number of top features to display
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        # Create DataFrame and sort by importance
        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importance_scores})
            .sort_values("importance", ascending=False)
            .head(top_n)
        )

        fig = go.Figure(
            data=[
                go.Bar(
                    x=importance_df["importance"],
                    y=importance_df["feature"],
                    orientation="h",
                    marker_color=self.colors["primary"],
                    text=np.round(importance_df["importance"], 4),
                    textposition="auto",
                )
            ]
        )

        fig.update_layout(
            title={
                "text": f"Top {top_n} Feature Importance Scores",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 18},
            },
            xaxis_title="Importance Score",
            yaxis_title="Features",
            template="plotly_white",
            height=600,
            yaxis={"categoryorder": "total ascending"},
        )

        if save_path:
            fig.write_html(save_path)
        elif self.config and self.config.save_plots:
            save_path = os.path.join(self.output_dir, "feature_importance.html")
            fig.write_html(save_path)
            logger.info(f"Feature importance plot saved to {save_path}")

        return fig

    def plot_model_performance_comparison(
        self,
        model_results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Plot comparison of multiple models' performance metrics.

        Args:
            model_results: Dict with model names as keys and metrics as values
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        model_names = list(model_results.keys())

        fig = go.Figure()

        for metric in metrics:
            values = [model_results[model].get(metric, 0) for model in model_names]
            fig.add_trace(
                go.Bar(
                    name=metric.replace("_", " ").title(),
                    x=model_names,
                    y=values,
                    text=[f"{v:.3f}" for v in values],
                    textposition="auto",
                )
            )

        fig.update_layout(
            title={
                "text": "Model Performance Comparison",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 18},
            },
            xaxis_title="Models",
            yaxis_title="Score",
            barmode="group",
            template="plotly_white",
            height=500,
            yaxis={"range": [0, 1.1]},
        )

        if save_path:
            fig.write_html(save_path)
        elif self.config and self.config.save_plots:
            save_path = os.path.join(self.output_dir, "model_comparison.html")
            fig.write_html(save_path)
            logger.info(f"Model comparison plot saved to {save_path}")

        return fig

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = None,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        cm = confusion_matrix(y_true, y_pred)

        if class_names is None:
            class_names = ["Not Bankrupt", "Bankrupt"]

        # Calculate percentages
        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        # Create annotations
        annotations = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f"{cm[i, j]}<br>({cm_percent[i, j]:.1f}%)",
                        showarrow=False,
                        font=dict(
                            color="white" if cm[i, j] > cm.max() / 2 else "black",
                            size=14,
                        ),
                    )
                )

        fig = go.Figure(
            data=go.Heatmap(
                z=cm, x=class_names, y=class_names, colorscale="Blues", showscale=True
            )
        )

        fig.update_layout(
            title={
                "text": "Confusion Matrix",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 18},
            },
            xaxis_title="Predicted",
            yaxis_title="Actual",
            annotations=annotations,
            template="plotly_white",
            height=500,
            width=500,
        )

        if save_path:
            fig.write_html(save_path)
        elif self.config and self.config.save_plots:
            save_path = os.path.join(self.output_dir, "confusion_matrix.html")
            fig.write_html(save_path)
            logger.info(f"Confusion matrix plot saved to {save_path}")

        return fig


# Legacy Visualizer class for backward compatibility
class Visualizer(BankruptcyVisualizer):
    """Legacy visualizer class - use BankruptcyVisualizer instead."""

    def __init__(self, config: Config):
        super().__init__(config)
        logger.warning(
            "Visualizer class is deprecated. Use BankruptcyVisualizer instead."
        )


# Export main classes and functions
__all__ = ["BankruptcyVisualizer", "Visualizer"]  # Legacy support
