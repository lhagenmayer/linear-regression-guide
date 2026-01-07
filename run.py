#!/usr/bin/env python3
"""
Unified Entry Point - Framework-Agnostic Application.

This module detects the runtime environment and launches the appropriate frontend.

Usage:
    # Auto-detect and run
    python run.py
    
    # Force Streamlit
    streamlit run run.py
    
    # Force Flask
    FLASK_APP=run.py flask run
    # or
    python run.py --flask
    
Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                         run.py                               │
    │                    (Auto-Detection)                          │
    └──────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ↓                                 ↓
    ┌─────────────────────┐           ┌─────────────────────┐
    │   Streamlit App     │           │     Flask App       │
    │  (st.* rendering)   │           │  (HTML rendering)   │
    └─────────────────────┘           └─────────────────────┘
              │                                 │
              └────────────────┬────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────┐
    │              ContentBuilder (Framework-Agnostic)             │
    │        SimpleRegressionContent / MultipleRegressionContent   │
    └─────────────────────────────────────────────────────────────┘
              │                                 │
              ↓                                 ↓
    ┌─────────────────────┐           ┌─────────────────────┐
    │ StreamlitRenderer   │           │   HTMLRenderer      │
    │  (interprets →st.*) │           │ (interprets →HTML)  │
    └─────────────────────┘           └─────────────────────┘
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def detect_framework() -> str:
    """
    Detect which framework to use.
    
    Returns:
        'streamlit', 'flask', or 'unknown'
    """
    # Check command line arguments
    if '--flask' in sys.argv:
        return 'flask'
    if '--streamlit' in sys.argv:
        return 'streamlit'
    
    # Check environment variables
    if os.environ.get('FLASK_APP'):
        return 'flask'
    
    # Check if running in Streamlit
    try:
        import streamlit.runtime.scriptrunner as sr
        ctx = sr.get_script_run_ctx()
        if ctx is not None:
            return 'streamlit'
    except (ImportError, Exception):
        pass
    
    # Check if imported by Streamlit CLI
    if any('streamlit' in arg.lower() for arg in sys.argv):
        return 'streamlit'
    
    # Default to Streamlit for interactive use
    return 'streamlit'


def run_streamlit():
    """Run Streamlit application."""
    from src.adapters.streamlit.app import run_streamlit_app
    run_streamlit_app()


def run_flask():
    """Run Flask application."""
    from src.adapters.flask_app import run_flask as flask_run
    flask_run()


def create_app():
    """Create Flask app for WSGI servers."""
    from src.adapters.flask_app import create_flask_app
    return create_flask_app()


# Main execution
framework = detect_framework()

if framework == 'flask':
    # For direct Python execution
    if __name__ == '__main__':
        run_flask()
    else:
        # For WSGI servers (gunicorn, etc.)
        app = create_app()
else:
    # Streamlit mode - execute the app
    run_streamlit()
