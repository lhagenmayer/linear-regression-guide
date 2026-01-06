"""
3D plotting functions for the Linear Regression Guide.

This module contains functions for creating 3D visualizations.
"""

from typing import Optional, Union, List, Tuple, Any
import numpy as np
import plotly.graph_objects as go
from ..config import get_logger

logger = get_logger(__name__)


def create_regression_mesh(
    x_data: np.ndarray,
    y_data: np.ndarray,
    coeffs: List[float],
    resolution: int = 30
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a mesh grid for 3D regression surface plotting.

    Args:
        x_data: X-axis original data (to determine range)
        y_data: Y-axis original data (to determine range)
        coeffs: Regression coefficients [intercept, b1, b2]
        resolution: Number of points in each dimension

    Returns:
        Tuple of (X_mesh, Y_mesh, Z_mesh) for plotting
    """
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_data), np.max(y_data)
    
    # Add small margin
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    
    x_lin = np.linspace(x_min - x_margin, x_max + x_margin, resolution)
    y_lin = np.linspace(y_min - y_margin, y_max + y_margin, resolution)
    X_mesh, Y_mesh = np.meshgrid(x_lin, y_lin)
    
    # Calculate Z based on coefficients: Z = b0 + b1*X + b2*Y
    if len(coeffs) >= 3:
        Z_mesh = coeffs[0] + coeffs[1] * X_mesh + coeffs[2] * Y_mesh
    elif len(coeffs) == 2:
        Z_mesh = coeffs[0] * X_mesh + coeffs[1] * Y_mesh
    else:
        Z_mesh = X_mesh * 0
        
    return X_mesh, Y_mesh, Z_mesh


def get_3d_layout_config(x_title: str, y_title: str, z_title: str, height: int = 600) -> dict:
    """Standardized 3D layout configuration."""
    return {
        "scene": {
            "xaxis": {"title": x_title},
            "yaxis": {"title": y_title},
            "zaxis": {"title": z_title},
        },
        "height": height,
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        "showlegend": True,
    }


def create_zero_plane(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_value: float = 0,
    color: str = "lightgray",
    opacity: float = 0.3
) -> go.Surface:
    """Create a zero plane for 3D plots."""
    x_vals = np.linspace(x_range[0], x_range[1], 10)
    y_vals = np.linspace(y_range[0], y_range[1], 10)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.full_like(X, z_value)

    return go.Surface(
        x=X, y=Y, z=Z,
        colorscale=[[0, color], [1, color]],
        opacity=opacity,
        showscale=False,
        name="Zero Plane"
    )


def create_plotly_3d_scatter(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    name: str = "Data Points",
    color: str = "blue",
    size: Optional[Union[int, np.ndarray]] = 4,
    opacity: float = 0.8
) -> go.Scatter3d:
    """Create a 3D scatter plot."""
    return go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        name=name,
        marker=dict(size=size, color=color, opacity=opacity)
    )


def create_plotly_3d_surface(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    x_scatter: Optional[np.ndarray] = None,
    y_scatter: Optional[np.ndarray] = None,
    z_scatter: Optional[np.ndarray] = None,
    name: str = "Regression Surface",
    colorscale: str = "Viridis",
    opacity: float = 0.7,
    **kwargs
) -> go.Figure:
    """
    Create a 3D surface plot with optional scatter points and full figure return.
    Includes backward compatibility for extra arguments from UI tabs.
    """
    fig = go.Figure()

    # Add surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        name=name,
        colorscale=colorscale,
        opacity=opacity,
        showscale=False
    ))

    # Add scatter points if provided
    if x_scatter is not None and y_scatter is not None and z_scatter is not None:
        fig.add_trace(go.Scatter3d(
            x=x_scatter, y=y_scatter, z=z_scatter,
            mode='markers',
            name=kwargs.get('scatter_name', "Observations"),
            marker=dict(size=4, color='red', opacity=0.8)
        ))

    # Apply layout and title
    title = kwargs.get('title', name)
    fig.update_layout(
        title=title, 
        template="plotly_white",
        scene=dict(
            xaxis_title=kwargs.get('x1_label', 'X1'),
            yaxis_title=kwargs.get('x2_label', 'X2'),
            zaxis_title=kwargs.get('y_label', 'Y')
        )
    )

    return fig