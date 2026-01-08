"""
Basis-Renderer - Abstraktes Interface für framework-agnostisches Rendering.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json

from ..infrastructure import DataResult, MultipleRegressionDataResult
from ..infrastructure import RegressionResult, MultipleRegressionResult
from ..infrastructure import PlotCollection


@dataclass
class RenderContext:
    """
    Framework-agnostischer Kontext für das Rendering.
    
    Enthält alle Daten, die zur Darstellung einer Regressionsanalyse benötigt werden,
    ohne Abhängigkeiten zu spezifischen UI-Frameworks (Modularität).
    """
    # Analysetyp ("simple" oder "multiple")
    analysis_type: str
    
    # Daten- und Statistik-Objekte aus der Domäne/Infrastruktur
    data: Any  # DataResult oder MultipleRegressionDataResult
    stats: Any  # RegressionResult oder MultipleRegressionResult
    
    # Plots als JSON (serialisierte Plotly-Figuren für den Transport)
    plots_json: Dict[str, str] = field(default_factory=dict)
    
    # Anzeige-Optionen für das UI
    show_formulas: bool = True
    show_true_line: bool = False
    compact_mode: bool = False
    
    # Dynamische Inhalte und Formeln
    content: Dict[str, Any] = field(default_factory=dict)
    formulas: Dict[str, str] = field(default_factory=dict)
    
    # Metadaten
    dataset_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert den Kontext in ein Dictionary für Template-Engines (z.B. Jinja2)."""
        return {
            "analysis_type": self.analysis_type,
            "dataset_name": self.dataset_name,
            "show_formulas": self.show_formulas,
            "compact_mode": self.compact_mode,
            "plots": self.plots_json,
            "content": self.content,
            "formulas": self.formulas,
            "stats": self._stats_to_dict(),
            "data": self._data_to_dict(),
        }
    
    def _stats_to_dict(self) -> Dict[str, Any]:
        """Überführt Statistik-Domain-Objekte in ein flaches Dictionary-Format."""
        if isinstance(self.stats, RegressionResult):
            return {
                "type": "simple",
                "intercept": self.stats.intercept,
                "slope": self.stats.slope,
                "r_squared": self.stats.r_squared,
                "r_squared_adj": self.stats.r_squared_adj,
                "p_slope": self.stats.p_slope,
                "t_slope": self.stats.t_slope,
                "se_slope": self.stats.se_slope,
                "n": self.stats.n,
                "df": self.stats.df,
                "mse": self.stats.mse,
                "sse": self.stats.sse,
                "sst": self.stats.sst,
                "ssr": self.stats.ssr,
            }
        elif isinstance(self.stats, MultipleRegressionResult):
            return {
                "type": "multiple",
                "intercept": self.stats.intercept,
                "coefficients": list(self.stats.coefficients),
                "r_squared": self.stats.r_squared,
                "r_squared_adj": self.stats.r_squared_adj,
                "f_statistic": self.stats.f_statistic,
                "f_pvalue": self.stats.f_pvalue,
                "p_values": list(self.stats.p_values),
                "t_values": list(self.stats.t_values),
                "n": self.stats.n,
                "k": self.stats.k,
            }
        return {}
    
    def _data_to_dict(self) -> Dict[str, Any]:
        """Überführt Datensatz-Informationen in ein Dictionary-Format."""
        if isinstance(self.data, DataResult):
            return {
                "type": "simple",
                "x_label": self.data.x_label,
                "y_label": self.data.y_label,
                "x_unit": self.data.x_unit,
                "y_unit": self.data.y_unit,
                "context_title": self.data.context_title,
                "context_description": self.data.context_description,
                "n": len(self.data.x),
            }
        elif isinstance(self.data, MultipleRegressionDataResult):
            return {
                "type": "multiple",
                "x1_label": self.data.x1_label,
                "x2_label": self.data.x2_label,
                "y_label": self.data.y_label,
                "n": len(self.data.y),
            }
        return {}


class BaseRenderer(ABC):
    """
    Abstrakte Basisklasse für framework-spezifische Renderer.
    
    Subklassen implementieren die tatsächliche Darstellungslogik für
    Streamlit, Flask oder andere UI-Technologien.
    """
    
    @abstractmethod
    def render(self, context: RenderContext) -> Any:
        """
        Führt das Rendering der Regressionsanalyse aus.
        """
        pass
    
    @abstractmethod
    def render_simple_regression(self, context: RenderContext) -> Any:
        """Spezifisches Rendering für einfache Regression."""
        pass
    
    @abstractmethod
    def render_multiple_regression(self, context: RenderContext) -> Any:
        """Spezifisches Rendering für multiple Regression."""
        pass
    
    @abstractmethod
    def run(self, host: str = "0.0.0.0", port: int = 8501, debug: bool = False) -> None:
        """
        Startet den Applikations-Server des jeweiligen Frameworks.
        """
        pass
    
    def serialize_plots(self, plots: PlotCollection) -> Dict[str, str]:
        """
        Serialisiert Plotly-Figuren in JSON-Strings für den framework-agnostischen Transport.
        
        Args:
            plots: PlotCollection Objekt aus der Pipeline.
            
        Returns:
            Dictionary mit Plot-Namen als Key und JSON-Strings als Value.
        """
        result = {}
        
        if plots.scatter is not None:
            result["scatter"] = plots.scatter.to_json()
        if plots.residuals is not None:
            result["residuals"] = plots.residuals.to_json()
        if plots.diagnostics is not None:
            result["diagnostics"] = plots.diagnostics.to_json()
        
        # Iteration über zusätzliche/benutzerdefinierte Plots
        for name, fig in plots.extra.items():
            if fig is not None:
                result[name] = fig.to_json()
        
        return result
