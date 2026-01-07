"""
Streamlit Adapter - All Streamlit-specific code.

This module contains:
- StreamlitRenderer: Main app controller
- Educational tabs with full Streamlit UI components
"""

from .app import StreamlitRenderer, create_streamlit_app
from .simple_regression_educational import render_simple_regression_educational
from .multiple_regression_educational import render_multiple_regression_educational

__all__ = [
    "StreamlitRenderer",
    "create_streamlit_app",
    "render_simple_regression_educational",
    "render_multiple_regression_educational",
]
