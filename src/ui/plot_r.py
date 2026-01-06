"""
R-style plotting functions for the Linear Regression Guide.

This module contains functions that mimic R's statistical plotting capabilities.
"""

from typing import Optional, List, Any
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from ..config import get_logger
from ..data import safe_scalar as _safe_scalar

logger = get_logger(__name__)


def create_r_output_display(model: Any, feature_names: List[str] = None) -> str:
    """
    Create R-style model summary output.

    Args:
        model: Statistical model object
        feature_names: List of feature names

    Returns:
        Formatted string for display
    """
    if not hasattr(model, 'summary'):
        return "Model summary not available"

    try:
        # Try to get summary from statsmodels-like model
        summary = model.summary()

        # Convert to string if needed
        if hasattr(summary, 'as_text'):
            return summary.as_text()
        elif hasattr(summary, 'as_latex'):
            # For LaTeX output, we'd need additional processing
            return str(summary)
        else:
            return str(summary)

    except Exception as e:
        logger.warning(f"Could not create R-style summary: {e}")
        return f"Model summary unavailable: {str(e)}"


def create_r_output_figure(
    model: Any,
    figure_type: str = "residuals",
    feature_names: Optional[List[str]] = None,
    **kwargs
) -> Optional[Any]:
    """
    Create R-style diagnostic figures.

    Args:
        model: Statistical model object
        figure_type: Type of figure ("residuals", "qqplot", "scale_location", etc.)
        feature_names: Feature names for labeling
        **kwargs: Additional arguments

    Returns:
        Plotly figure object or None if not supported
    """
    try:
        if figure_type == "residuals":
            return _create_residuals_vs_fitted_plot(model, **kwargs)
        elif figure_type == "qqplot":
            return _create_qq_plot(model, **kwargs)
        elif figure_type == "scale_location":
            return _create_scale_location_plot(model, **kwargs)
        elif figure_type == "residuals_leverage":
            return _create_residuals_leverage_plot(model, **kwargs)
        else:
            logger.warning(f"Unknown figure type: {figure_type}")
            # Return empty figure instead of None to avoid crash
            fig = go.Figure()
            fig.add_annotation(text=f"Unknown plot type: {figure_type}", showarrow=False)
            return fig

    except Exception as e:
        logger.error(f"Could not create {figure_type} plot: {e}")
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating plot: {e}", showarrow=False)
        return fig


def _create_residuals_vs_fitted_plot(model: Any, **kwargs) -> go.Figure:
    """Create Residuals vs Fitted plot (R-style diagnostic plot 1)."""
    try:
        # Extract fitted values and residuals
        if hasattr(model, 'fittedvalues') and hasattr(model, 'resid'):
            fitted = model.fittedvalues
            residuals = model.resid
        elif isinstance(model, dict):
            # Fallback for mock models
            fitted = model.get('fittedvalues', np.zeros(10))
            residuals = model.get('resid', np.zeros(10))
        else:
            raise ValueError("Model does not have fittedvalues or resid")

        fig = go.Figure()
        
        # Add points
        fig.add_trace(go.Scatter(
            x=fitted, y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='rgba(31, 119, 180, 0.6)', size=8)
        ))
        
        # Add zero line
        fig.add_trace(go.Scatter(
            x=[min(fitted), max(fitted)], y=[0, 0],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Zero Line',
            showlegend=False
        ))
        
        fig.update_layout(
            title="Residuals vs Fitted",
            xaxis_title="Fitted values",
            yaxis_title="Residuen",
            template="plotly_white"
        )
        return fig
    except Exception as e:
        logger.warning(f"Could not create Residuals vs Fitted plot: {e}")
        fig = go.Figure()
        fig.add_annotation(text="Diagnostics plot unavailable", showarrow=False)
        return fig


def _create_qq_plot(model: Any, **kwargs) -> Any:
    """Create Q-Q plot (R-style diagnostic plot 2)."""
    fig = go.Figure()
    fig.add_annotation(text="Q-Q Plot Placeholder", showarrow=False)
    fig.update_layout(title="Normal Q-Q", template="plotly_white")
    return fig


def _create_scale_location_plot(model: Any, **kwargs) -> Any:
    """Create Scale-Location plot (R-style diagnostic plot 3)."""
    fig = go.Figure()
    fig.add_annotation(text="Scale-Location Placeholder", showarrow=False)
    fig.update_layout(title="Scale-Location", template="plotly_white")
    return fig


def _create_residuals_leverage_plot(model: Any, **kwargs) -> Any:
    """Create Residuals vs Leverage plot (R-style diagnostic plot 5)."""
    fig = go.Figure()
    fig.add_annotation(text="Residuals vs Leverage Placeholder", showarrow=False)
    fig.update_layout(title="Residuals vs Leverage", template="plotly_white")
    return fig