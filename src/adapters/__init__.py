"""
Adapters Module - Framework-specific frontends.

This module provides the bridge between our framework-agnostic content
and specific UI frameworks (Streamlit, Flask).

Architecture:
    ContentBuilder (content/) → produces → EducationalContent (data)
                                                  ↓
                           ┌─────────────────────┴─────────────────────┐
                           ↓                                           ↓
                StreamlitContentRenderer                       HTMLContentRenderer
                   (st.markdown, etc.)                        (HTML/Jinja2)
                           ↓                                           ↓
                    Streamlit App                                Flask App
"""

from .detector import FrameworkDetector, Framework
from .base import BaseRenderer, RenderContext

# Lazy imports for framework-specific components
def get_streamlit_app():
    """Get Streamlit app function."""
    from .streamlit.app import run_streamlit_app
    return run_streamlit_app

def get_flask_app():
    """Get Flask app creator."""
    from .flask_app import create_flask_app, run_flask
    return create_flask_app, run_flask

__all__ = [
    "FrameworkDetector",
    "Framework",
    "BaseRenderer",
    "RenderContext",
    "get_streamlit_app",
    "get_flask_app",
]
