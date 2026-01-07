"""
AI Module - Perplexity AI Integration.

Provides AI-powered explanations and insights for regression analysis.
Framework-agnostic: works with both Streamlit and Flask.
"""

from .perplexity_client import PerplexityClient, PerplexityConfig
from .prompts import RegressionPrompts
from .cache import AICache

__all__ = [
    "PerplexityClient",
    "PerplexityConfig", 
    "RegressionPrompts",
    "AICache",
]
