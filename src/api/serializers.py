"""
Serializers - Convert all data structures to JSON.

100% platform agnostic. Pure Python, no framework dependencies.
All outputs are JSON-serializable dictionaries.
"""

from typing import Dict, Any, List, Union, Optional
import json
import numpy as np


def _to_list(arr: Any) -> List:
    """Convert numpy array or any iterable to list."""
    if arr is None:
        return []
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    if hasattr(arr, 'tolist'):
        return arr.tolist()
    return list(arr)


def _to_float(val: Any) -> Optional[float]:
    """Convert to float, handle NaN."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


class DataSerializer:
    """
    Serialize DataResult to JSON.
    
    Converts numpy arrays to lists for JSON compatibility.
    """
    
    @staticmethod
    def serialize_simple(data) -> Dict[str, Any]:
        """
        Serialize simple regression data.
        
        Args:
            data: DataResult from pipeline
            
        Returns:
            JSON-serializable dictionary
        """
        return {
            "type": "simple_regression_data",
            "x": _to_list(data.x),
            "y": _to_list(data.y),
            "n": int(data.n),
            "x_label": str(data.x_label),
            "y_label": str(data.y_label),
            "x_unit": str(getattr(data, 'x_unit', '')),
            "y_unit": str(getattr(data, 'y_unit', '')),
            "context": {
                "title": str(getattr(data, 'context_title', '')),
                "description": str(getattr(data, 'context_description', '')),
            },
            "extra": data.extra if data.extra else {},
        }
    
    @staticmethod
    def serialize_multiple(data) -> Dict[str, Any]:
        """
        Serialize multiple regression data.
        
        Args:
            data: MultipleRegressionDataResult from pipeline
            
        Returns:
            JSON-serializable dictionary
        """
        return {
            "type": "multiple_regression_data",
            "x1": _to_list(data.x1),
            "x2": _to_list(data.x2),
            "y": _to_list(data.y),
            "n": int(data.n),
            "x1_label": str(data.x1_label),
            "x2_label": str(data.x2_label),
            "y_label": str(data.y_label),
            "extra": data.extra if data.extra else {},
        }


class StatsSerializer:
    """
    Serialize regression results to JSON.
    
    All statistics are converted to JSON-safe types.
    """
    
    @staticmethod
    def serialize_simple(result) -> Dict[str, Any]:
        """
        Serialize simple regression result.
        
        Args:
            result: RegressionResult from pipeline
            
        Returns:
            JSON-serializable dictionary
        """
        return {
            "type": "simple_regression_stats",
            
            # Coefficients
            "coefficients": {
                "intercept": _to_float(result.intercept),
                "slope": _to_float(result.slope),
            },
            
            # Model fit
            "model_fit": {
                "r_squared": _to_float(result.r_squared),
                "r_squared_adj": _to_float(result.r_squared_adj),
            },
            
            # Standard errors
            "standard_errors": {
                "intercept": _to_float(result.se_intercept),
                "slope": _to_float(result.se_slope),
            },
            
            # Test statistics
            "t_tests": {
                "intercept": {
                    "t_value": _to_float(result.t_intercept),
                    "p_value": _to_float(result.p_intercept),
                },
                "slope": {
                    "t_value": _to_float(result.t_slope),
                    "p_value": _to_float(result.p_slope),
                },
            },
            
            # Sum of squares
            "sum_of_squares": {
                "sse": _to_float(result.sse),
                "sst": _to_float(result.sst),
                "ssr": _to_float(result.ssr),
                "mse": _to_float(result.mse),
            },
            
            # Sample info
            "sample": {
                "n": int(result.n),
                "df": int(result.df),
            },
            
            # Predictions & Residuals (can be large, optional)
            "predictions": _to_list(result.y_pred),
            "residuals": _to_list(result.residuals),
            
            # Extra stats
            "extra": {
                k: _to_float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in (result.extra or {}).items()
            },
        }
    
    @staticmethod
    def serialize_multiple(result) -> Dict[str, Any]:
        """
        Serialize multiple regression result.
        
        Args:
            result: MultipleRegressionResult from pipeline
            
        Returns:
            JSON-serializable dictionary
        """
        return {
            "type": "multiple_regression_stats",
            
            # Coefficients
            "coefficients": {
                "intercept": _to_float(result.intercept),
                "slopes": [_to_float(c) for c in result.coefficients],
            },
            
            # Model fit
            "model_fit": {
                "r_squared": _to_float(result.r_squared),
                "r_squared_adj": _to_float(result.r_squared_adj),
                "f_statistic": _to_float(result.f_statistic),
                "f_p_value": _to_float(result.f_pvalue),
            },
            
            # Standard errors
            "standard_errors": [_to_float(se) for se in result.se_coefficients],
            
            # Test statistics for each coefficient
            "t_tests": {
                "t_values": [_to_float(t) for t in result.t_values],
                "p_values": [_to_float(p) for p in result.p_values],
            },
            
            # Sum of squares
            "sum_of_squares": {
                "sse": _to_float(result.sse),
                "sst": _to_float(result.sst),
                "ssr": _to_float(result.ssr),
            },
            
            # Sample info
            "sample": {
                "n": int(result.n),
                "k": int(result.k),
            },
            
            # Predictions & Residuals
            "predictions": _to_list(result.y_pred),
            "residuals": _to_list(result.residuals),
            
            # Extra
            "extra": {
                k: _to_float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in (result.extra or {}).items()
            },
        }
    
    @staticmethod
    def to_flat_dict(result, data=None) -> Dict[str, Any]:
        """
        Convert to flat dictionary (for backward compatibility).
        
        Useful for templates and simple consumers.
        """
        if hasattr(result, 'slope'):
            # Simple regression
            flat = {
                "intercept": _to_float(result.intercept),
                "slope": _to_float(result.slope),
                "r_squared": _to_float(result.r_squared),
                "r_squared_adj": _to_float(result.r_squared_adj),
                "se_intercept": _to_float(result.se_intercept),
                "se_slope": _to_float(result.se_slope),
                "t_intercept": _to_float(result.t_intercept),
                "t_slope": _to_float(result.t_slope),
                "p_intercept": _to_float(result.p_intercept),
                "p_slope": _to_float(result.p_slope),
                "sse": _to_float(result.sse),
                "sst": _to_float(result.sst),
                "ssr": _to_float(result.ssr),
                "mse": _to_float(result.mse),
                "n": int(result.n),
                "df": int(result.df),
                "residuals": _to_list(result.residuals),
            }
            if result.extra:
                flat.update({
                    k: _to_float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in result.extra.items()
                })
        else:
            # Multiple regression
            flat = {
                "intercept": _to_float(result.intercept),
                "b1": _to_float(result.coefficients[0]) if result.coefficients else None,
                "b2": _to_float(result.coefficients[1]) if len(result.coefficients) > 1 else None,
                "r_squared": _to_float(result.r_squared),
                "r_squared_adj": _to_float(result.r_squared_adj),
                "f_statistic": _to_float(result.f_statistic),
                "p_f": _to_float(result.f_pvalue),
                "n": int(result.n),
                "k": int(result.k),
                "residuals": _to_list(result.residuals),
            }
            if result.extra:
                flat.update({
                    k: _to_float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in result.extra.items()
                })
        
        # Add data context if provided
        if data:
            if hasattr(data, 'x_label'):
                flat["x_label"] = str(data.x_label)
                flat["y_label"] = str(data.y_label)
                flat["context_title"] = str(getattr(data, 'context_title', ''))
                flat["context_description"] = str(getattr(data, 'context_description', ''))
            else:
                flat["x1_label"] = str(data.x1_label)
                flat["x2_label"] = str(data.x2_label)
                flat["y_label"] = str(data.y_label)
        
        return flat


class PlotSerializer:
    """
    Serialize Plotly figures to JSON.
    
    Plotly figures are natively JSON-serializable.
    Any frontend can render them using plotly.js.
    """
    
    @staticmethod
    def serialize_figure(fig) -> Dict[str, Any]:
        """
        Serialize a single Plotly figure.
        
        Args:
            fig: plotly.graph_objects.Figure
            
        Returns:
            JSON-serializable dictionary (Plotly JSON format)
        """
        if fig is None:
            return None
        return json.loads(fig.to_json())
    
    @staticmethod
    def serialize_collection(plots) -> Dict[str, Any]:
        """
        Serialize PlotCollection to JSON.
        
        Args:
            plots: PlotCollection from pipeline
            
        Returns:
            Dictionary of JSON-serialized plots
        """
        result = {
            "scatter": PlotSerializer.serialize_figure(plots.scatter),
            "residuals": PlotSerializer.serialize_figure(plots.residuals),
            "diagnostics": PlotSerializer.serialize_figure(plots.diagnostics),
        }
        
        # Add extra plots
        if plots.extra:
            result["extra"] = {
                key: PlotSerializer.serialize_figure(fig)
                for key, fig in plots.extra.items()
            }
        
        return result


class ContentSerializer:
    """
    Serialize educational content to JSON.
    
    ContentElements already have to_dict() methods.
    """
    
    @staticmethod
    def serialize(content) -> Dict[str, Any]:
        """
        Serialize EducationalContent.
        
        Args:
            content: EducationalContent from ContentBuilder
            
        Returns:
            JSON-serializable dictionary
        """
        return content.to_dict()


class PipelineSerializer:
    """
    Serialize complete pipeline result to JSON.
    
    Combines all serializers for a full API response.
    """
    
    @staticmethod
    def serialize(pipeline_result, include_predictions: bool = True) -> Dict[str, Any]:
        """
        Serialize complete PipelineResult.
        
        Args:
            pipeline_result: PipelineResult from RegressionPipeline
            include_predictions: Whether to include y_pred and residuals
            
        Returns:
            Complete JSON-serializable response
        """
        # Determine regression type
        is_simple = pipeline_result.pipeline_type == "simple"
        
        # Serialize data
        if is_simple:
            data = DataSerializer.serialize_simple(pipeline_result.data)
            stats = StatsSerializer.serialize_simple(pipeline_result.stats)
        else:
            data = DataSerializer.serialize_multiple(pipeline_result.data)
            stats = StatsSerializer.serialize_multiple(pipeline_result.stats)
        
        # Optionally remove large arrays
        if not include_predictions:
            stats.pop("predictions", None)
            stats.pop("residuals", None)
        
        # Serialize plots
        plots = PlotSerializer.serialize_collection(pipeline_result.plots)
        
        return {
            "type": pipeline_result.pipeline_type,
            "data": data,
            "stats": stats,
            "plots": plots,
            "params": pipeline_result.params,
        }
    
    @staticmethod
    def serialize_minimal(pipeline_result) -> Dict[str, Any]:
        """
        Minimal serialization (no predictions/residuals).
        
        Useful for list views or summaries.
        """
        return PipelineSerializer.serialize(pipeline_result, include_predictions=False)
