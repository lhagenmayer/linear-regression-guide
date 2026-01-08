"""
Pipeline Package - Einfache 4-stufige Datenverarbeitungspipeline.
Migriert in den Infrastructure-Layer.

Die Pipeline folgt einem klaren Ablauf:
    1. GET      → Daten abrufen oder generieren
    2. CALCULATE → Statistiken berechnen & Modelle trainieren
    3. PLOT     → Visualisierungen erstellen
    4. DISPLAY  → In der Benutzeroberfläche rendern

Nutzung:
    from src.infrastructure import RegressionPipeline
    
    pipeline = RegressionPipeline()
    result = pipeline.run(dataset="electronics", n=50)
"""

# Kernkomponenten aus dem migrierten Infrastructure-Layer
from .data.generators import DataFetcher
from .services.calculate import StatisticsCalculator

# Lazy Imports für Komponenten mit externen Abhängigkeiten (Performance/Modularität)
def get_plot_builder():
    """Importiert den PlotBuilder verzögert (benötigt Plotly)."""
    from .services.plot import PlotBuilder
    return PlotBuilder

# Komfort-Imports für die vollständige Pipeline-Struktur
try:
    from .services.plot import PlotBuilder, PlotCollection
    from .regression_pipeline import RegressionPipeline, PipelineResult
except ImportError:
    # Fallback, falls Abhängigkeiten fehlen
    PlotBuilder = None
    PlotCollection = None
    RegressionPipeline = None
    PipelineResult = None

__all__ = [
    # Haupt-Pipeline
    'RegressionPipeline',
    'PipelineResult',
    # Einzelne Schritte
    'DataFetcher',
    'StatisticsCalculator', 
    'PlotBuilder',
    'PlotCollection',
    # Lazy Loaders
    'get_plot_builder',
]
