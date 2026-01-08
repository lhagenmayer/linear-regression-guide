"""
Serializers - Konvertierung aller Datenstrukturen in JSON.

100% plattformunabhängig. Reines Python, keine Framework-Abhängigkeiten.
Alle Ausgaben sind JSON-serialisierbare Dictionaries.
"""

from typing import Dict, Any, List, Union, Optional
import json
import numpy as np


def _to_list(arr: Any) -> List:
    """Konvertiert NumPy-Arrays oder Iterables in Standard-Listen."""
    if arr is None:
        return []
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    if hasattr(arr, 'tolist'):
        return arr.tolist()
    return list(arr)


def _to_float(val: Any) -> Optional[float]:
    """Konvertiert Werte in Float und behandelt NaN/Inf."""
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
    Serialisiert DataResult-Objekte für die API.
    Konvertiert NumPy-Arrays in Listen für die JSON-Kompatibilität.
    """
    
    @staticmethod
    def serialize_simple(data) -> Dict[str, Any]:
        """
        Serialisiert Daten der einfachen Regression.
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
        Serialisiert Daten der multiplen Regression.
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
    Serialisiert Regressionsergebnisse für JSON.
    Alle statistischen Kennzahlen werden in JSON-sichere Typen konvertiert.
    """
    
    @staticmethod
    def serialize_simple(result) -> Dict[str, Any]:
        """
        Serialisiert Ergebnisse der einfachen Regression.
        """
        return {
            "type": "simple_regression_stats",
            
            # Koeffizienten (Intercept & Steigung)
            "coefficients": {
                "intercept": _to_float(result.intercept),
                "slope": _to_float(result.slope),
            },
            
            # Modell-Güte (Bestimmtheitsmaß)
            "model_fit": {
                "r_squared": _to_float(result.r_squared),
                "r_squared_adj": _to_float(result.r_squared_adj),
            },
            
            # Standardfehler
            "standard_errors": {
                "intercept": _to_float(result.se_intercept),
                "slope": _to_float(result.se_slope),
            },
            
            # t-Tests für Signifikanz
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
            
            # Quadratsummen (SSE, SST, SSR, MSE)
            "sum_of_squares": {
                "sse": _to_float(result.sse),
                "sst": _to_float(result.sst),
                "ssr": _to_float(result.ssr),
                "mse": _to_float(result.mse),
            },
            
            # Stichproben-Informationen
            "sample": {
                "n": int(result.n),
                "df": int(result.df),
            },
            
            # Vorhersagen & Residuen (Vektordaten)
            "predictions": _to_list(result.y_pred),
            "residuals": _to_list(result.residuals),
            
            # Zusätzliche Statistiken
            "extra": {
                k: _to_float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in (result.extra or {}).items()
            },
        }
    
    @staticmethod
    def serialize_multiple(result) -> Dict[str, Any]:
        """
        Serialisiert Ergebnisse der multiplen Regression.
        """
        return {
            "type": "multiple_regression_stats",
            
            # Koeffizienten-Vektor
            "coefficients": {
                "intercept": _to_float(result.intercept),
                "slopes": [_to_float(c) for c in result.coefficients],
            },
            
            # Modell-Güte & F-Statistik
            "model_fit": {
                "r_squared": _to_float(result.r_squared),
                "r_squared_adj": _to_float(result.r_squared_adj),
                "f_statistic": _to_float(result.f_statistic),
                "f_p_value": _to_float(result.f_pvalue),
            },
            
            # Standardfehler der Koeffizienten
            "standard_errors": [_to_float(se) for se in result.se_coefficients],
            
            # t-Tests für alle Koeffizienten
            "t_tests": {
                "t_values": [_to_float(t) for t in result.t_values],
                "p_values": [_to_float(p) for p in result.p_values],
            },
            
            # Quadratsummen
            "sum_of_squares": {
                "sse": _to_float(result.sse),
                "sst": _to_float(result.sst),
                "ssr": _to_float(result.ssr),
            },
            
            # Stichproben-Info
            "sample": {
                "n": int(result.n),
                "k": int(result.k),
            },
            
            # Vorhersagen & Residuen
            "predictions": _to_list(result.y_pred),
            "residuals": _to_list(result.residuals),
            
            # Extra-Felder
            "extra": {
                k: _to_float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in (result.extra or {}).items()
            },
        }
    
    @staticmethod
    def to_flat_dict(result, data=None) -> Dict[str, Any]:
        """
        Konvertiert das Ergebnis in ein flaches Dictionary für den ContentBuilder.
        Dieses Format wird von SimpleRegressionContent und MultipleRegressionContent 
        erwartet, um edukative Texte zu generieren.
        """
        # Lazy Loading der Inhalts-Registry, um zirkuläre Abhängigkeiten zu vermeiden
        from ..data.content import get_simple_regression_content, get_multiple_regression_descriptions
        
        if hasattr(result, 'slope'):
            # Fall: Einfache Regression
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
                "y_pred": _to_list(result.y_pred),
            }
            
            # Zusätzliche Metriken hinzufügen
            if result.extra:
                flat.update({
                    k: _to_float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in result.extra.items()
                })
            
            # Berechnete F-Statistik (für edukative Zwecke)
            msr = flat["ssr"] / 1 if flat["ssr"] else 0
            mse = flat["mse"] if flat["mse"] else 1
            flat["f_statistic"] = msr / mse if mse > 0 else 0
            flat["p_f"] = flat["p_slope"] 
            
            # Datenkontext und deskriptive Statistiken hinzufügen
            if data:
                x = np.array(data.x) if hasattr(data.x, '__iter__') else data.x
                y = np.array(data.y) if hasattr(data.y, '__iter__') else data.y
                
                flat["x_label"] = str(data.x_label)
                flat["y_label"] = str(data.y_label)
                flat["x_unit"] = str(getattr(data, 'x_unit', ''))
                flat["y_unit"] = str(getattr(data, 'y_unit', ''))
                
                # Versuch, reichhaltige Inhalte aus der Registry zu laden
                dataset_id = getattr(data, 'extra', {}).get('dataset')
                context_title = str(getattr(data, 'context_title', 'Regressionsanalyse'))
                context_desc = str(getattr(data, 'context_description', ''))
                
                if dataset_id:
                    try:
                        rich_content = get_simple_regression_content(dataset_id, flat["x_label"])
                        if rich_content.get("context_title"):
                            context_title = rich_content["context_title"]
                        if rich_content.get("context_description"):
                            context_desc = rich_content["context_description"]
                    except Exception:
                        pass # Rückfall auf Basis-Metadaten
                
                flat["context_title"] = context_title
                flat["context_description"] = context_desc
                
                # Deskriptive Statistiken (Mittelwert, Std-Abw, etc.)
                if len(x) > 0:
                    flat["x_mean"] = float(np.mean(x))
                    flat["x_std"] = float(np.std(x, ddof=1)) if len(x) > 1 else 0
                    flat["x_min"] = float(np.min(x))
                    flat["x_max"] = float(np.max(x))
                    flat["y_mean"] = float(np.mean(y))
                    flat["y_std"] = float(np.std(y, ddof=1)) if len(y) > 1 else 0
                    flat["y_min"] = float(np.min(y))
                    flat["y_max"] = float(np.max(y))
                    
                    # Korrelationskoeffizienten
                    if len(x) > 1:
                        flat["covariance"] = float(np.cov(x, y, ddof=1)[0, 1])
                        from scipy import stats as scipy_stats
                        corr = np.corrcoef(x, y)[0, 1]
                        flat["correlation"] = float(corr)
                        # t-Test für Korrelation
                        if abs(corr) < 1:
                            t_corr = corr * np.sqrt((len(x) - 2) / (1 - corr**2))
                            flat["t_correlation"] = float(t_corr)
                            flat["p_correlation"] = float(2 * (1 - scipy_stats.t.cdf(abs(t_corr), df=len(x)-2)))
                        # Spearman-Rangkorrelation
                        spearman_r, spearman_p = scipy_stats.spearmanr(x, y)
                        flat["spearman_r"] = float(spearman_r)
                        flat["spearman_p"] = float(spearman_p)
        else:
            # Fall: Multiple Regression
            coeffs = result.coefficients if result.coefficients else [0, 0]
            se_coeffs = result.se_coefficients if result.se_coefficients else [0, 0, 0]
            t_vals = result.t_values if result.t_values else [0, 0, 0]
            p_vals = result.p_values if result.p_values else [1, 1, 1]
            
            flat = {
                "intercept": _to_float(result.intercept),
                "b1": _to_float(coeffs[0]) if len(coeffs) > 0 else 0,
                "b2": _to_float(coeffs[1]) if len(coeffs) > 1 else 0,
                "beta1": _to_float(coeffs[0]) if len(coeffs) > 0 else 0,
                "beta2": _to_float(coeffs[1]) if len(coeffs) > 1 else 0,
                "se_intercept": _to_float(se_coeffs[0]) if len(se_coeffs) > 0 else 0,
                "se_beta1": _to_float(se_coeffs[1]) if len(se_coeffs) > 1 else 0,
                "se_beta2": _to_float(se_coeffs[2]) if len(se_coeffs) > 2 else 0,
                "t_intercept": _to_float(t_vals[0]) if len(t_vals) > 0 else 0,
                "t_beta1": _to_float(t_vals[1]) if len(t_vals) > 1 else 0,
                "t_beta2": _to_float(t_vals[2]) if len(t_vals) > 2 else 0,
                "p_intercept": _to_float(p_vals[0]) if len(p_vals) > 0 else 1,
                "p_beta1": _to_float(p_vals[1]) if len(p_vals) > 1 else 1,
                "p_beta2": _to_float(p_vals[2]) if len(p_vals) > 2 else 1,
                "r_squared": _to_float(result.r_squared),
                "r_squared_adj": _to_float(result.r_squared_adj),
                "f_statistic": _to_float(result.f_statistic),
                "f_p_value": _to_float(result.f_pvalue),
                "n": int(result.n),
                "k": int(result.k),
                "df": int(result.n - result.k - 1),
                "sse": _to_float(result.sse),
                "sst": _to_float(result.sst),
                "ssr": _to_float(result.ssr),
                "residuals": _to_list(result.residuals),
                "y_pred": _to_list(result.y_pred),
            }
            
            if result.extra:
                flat.update({
                    k: _to_float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in result.extra.items()
                })
            
            # Datenkontext für multiple Regression
            if data:
                x1 = np.array(data.x1) if hasattr(data.x1, '__iter__') else data.x1
                x2 = np.array(data.x2) if hasattr(data.x2, '__iter__') else data.x2
                
                flat["x1_label"] = str(data.x1_label)
                flat["x2_label"] = str(data.x2_label)
                flat["y_label"] = str(data.y_label)
                
                dataset_id = getattr(data, 'extra', {}).get('dataset')
                context_title = "Multiple Regression"
                context_desc = f"Analyse von {data.y_label} mit {data.x1_label} und {data.x2_label}"
                
                if dataset_id:
                     try:
                        rich_info = get_multiple_regression_descriptions(dataset_id)
                        if rich_info.get("main"):
                            context_desc = rich_info["main"]
                            # Dynamische Titelvergabe basierend auf dataset_id
                            if "cantons" in dataset_id.lower() or "Schweizer" in context_desc:
                                context_title = "Schweizer Kantone (Multipel)"
                            elif "weather" in dataset_id.lower():
                                context_title = "Schweizer Wetter (Multipel)"
                            elif "cities" in dataset_id.lower():
                                context_title = "Städte-Umsatzstudie"
                            elif "houses" in dataset_id.lower():
                                context_title = "Häuserpreise"
                            elif "world_bank" in dataset_id.lower():
                                context_title = "World Bank Development"
                            elif "fred" in dataset_id.lower():
                                context_title = "US Economy (FRED)"
                            elif "who" in dataset_id.lower():
                                context_title = "WHO Global Health"
                            elif "eurostat" in dataset_id.lower():
                                context_title = "Eurostat Analysis"
                            elif "nasa" in dataset_id.lower():
                                context_title = "NASA Agro-Climatology"
                                
                     except Exception:
                        pass
                
                flat["context_title"] = context_title
                flat["context_description"] = context_desc
                
                # Multikollinearität (VIF - Variance Inflation Factor)
                if len(x1) > 1:
                    corr_x1_x2 = float(np.corrcoef(x1, x2)[0, 1])
                    flat["corr_x1_x2"] = corr_x1_x2
                    r2_x = corr_x1_x2 ** 2
                    vif = 1 / (1 - r2_x) if r2_x < 1 else float('inf')
                    flat["vif_x1"] = vif
                    flat["vif_x2"] = vif
                
                # Durbin-Watson Platzhalter (für zukünftige Autokorrelations-Tests)
                flat["durbin_watson"] = 2.0
        
        return flat


class PlotSerializer:
    """
    Serialisiert Plotly-Figuren für JSON.
    Plotly-Figuren sind von Natur aus JSON-serialisierbar.
    Jedes Frontend kann sie mittels plotly.js rendern.
    """
    
    @staticmethod
    def serialize_figure(fig) -> Dict[str, Any]:
        """Konvertiert eine einzelne Plotly-Figur in ein Dictionary."""
        if fig is None:
            return {}
        return fig.to_dict()
    
    @staticmethod
    def serialize_collection(plots) -> Dict[str, Any]:
        """
        Serialisiert eine komplette Sammlung von Plots.
        """
        serialized = {
            "scatter": PlotSerializer.serialize_figure(plots.scatter),
            "residuals": PlotSerializer.serialize_figure(plots.residuals),
            "diagnostics": PlotSerializer.serialize_figure(plots.diagnostics),
        }
        
        # Zusätzliche Plots hinzufügen, falls vorhanden
        if plots.extra:
            for k, v in plots.extra.items():
                serialized[k] = PlotSerializer.serialize_figure(v)
                
        return serialized


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


class ClassificationSerializer:
    """
    Serialize classification results.
    """
    
    @staticmethod
    def serialize(dto) -> Dict[str, Any]:
        """
        Serialize ClassificationResponseDTO.
        """
        return {
            "success": dto.success,
            "method": dto.method,
            "data": {
                "X": _to_list(dto.X_data),
                "y": _to_list(dto.y_data),
                "feature_names": _to_list(dto.feature_names),
                "target_names": _to_list(dto.target_names),
            },
            "results": {
                "classes": _to_list(dto.classes),
                "predictions": _to_list(dto.predictions),
                "probabilities": _to_list(dto.probabilities) if dto.probabilities else None,
                "metrics": dto.metrics,
                "params": dto.parameters,
            },
            "metadata": {
                "dataset": dto.dataset_name,
                "description": dto.dataset_description,
            }
        }

    @staticmethod
    def to_flat_dict(dto) -> Dict[str, Any]:
        """
        Convert DTO to flat dictionary for ContentBuilders.
        """
        stats = {
            # Metadata
            "dataset_name": dto.dataset_name,
            "dataset_description": dto.dataset_description,
            "x_label": dto.feature_names[0] if dto.feature_names else "X",
            "y_label": "Class",
            "target_names": dto.target_names,
            "feature_names": dto.feature_names,
            
            # Metrics (Flattened)
            "accuracy": dto.metrics.get("accuracy"),
            "precision": dto.metrics.get("precision"),
            "recall": dto.metrics.get("recall"),
            "f1": dto.metrics.get("f1") or dto.metrics.get("f1_score"),
            "auc": dto.metrics.get("auc"),
            "confusion_matrix": dto.metrics.get("confusion_matrix"),
            
            # Model Params
            "method": dto.method,
            "k": dto.parameters.get("k"),
            "intercept": dto.parameters.get("intercept"),
            "coefficients": dto.parameters.get("coefficients"),
        }
        return stats
