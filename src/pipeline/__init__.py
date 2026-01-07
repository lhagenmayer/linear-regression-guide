"""
Pipeline Package - Simple 4-step data processing pipeline.

The pipeline follows a clear flow:
    1. GET      → Fetch/generate data
    2. CALCULATE → Compute statistics & fit models
    3. PLOT     → Create visualizations
    4. DISPLAY  → Render in UI

Usage:
    from src.pipeline import RegressionPipeline
    
    pipeline = RegressionPipeline()
    result = pipeline.run(dataset="electronics", n=50)
"""

from .get_data import DataFetcher
from .calculate import StatisticsCalculator
from .plot import PlotBuilder
from .display import UIRenderer
from .regression_pipeline import RegressionPipeline, PipelineResult

__all__ = [
    # Core pipeline
    'RegressionPipeline',
    'PipelineResult',
    # Individual steps
    'DataFetcher',
    'StatisticsCalculator', 
    'PlotBuilder',
    'UIRenderer',
]
