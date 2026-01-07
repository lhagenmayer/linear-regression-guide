"""
Framework Adapters - Frontend-agnostic rendering layer.

Structure:
- base.py: Abstract renderer interface (RenderContext, BaseRenderer)
- detector.py: Framework auto-detection

- streamlit/: Streamlit-specific implementation
  - app.py: StreamlitRenderer
  - simple_regression_educational.py: Educational content with st.* components
  - multiple_regression_educational.py: Educational content with st.* components

- flask_app.py: Flask implementation
- templates/: Flask HTML templates

Auto-detection chooses the right framework at runtime.
"""

from .detector import FrameworkDetector, Framework
from .base import BaseRenderer, RenderContext

__all__ = [
    "FrameworkDetector",
    "Framework",
    "BaseRenderer",
    "RenderContext",
]


# Lazy imports for framework-specific modules
def get_streamlit_renderer():
    """Get StreamlitRenderer (lazy import)."""
    from .streamlit import StreamlitRenderer
    return StreamlitRenderer


def get_flask_renderer():
    """Get FlaskRenderer (lazy import)."""
    from .flask_app import FlaskRenderer
    return FlaskRenderer
