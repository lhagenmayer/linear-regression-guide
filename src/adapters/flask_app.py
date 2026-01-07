"""
Flask Application - Universal Frontend.

Uses ContentBuilder + HTMLContentRenderer for truly frontend-agnostic
educational content rendering.

The same content structure is used by Streamlit - only the renderer differs.
"""

import numpy as np
from flask import Flask, render_template, request, jsonify
from typing import Dict, Any, Optional

from ..config import get_logger
from ..pipeline import RegressionPipeline
from ..content import SimpleRegressionContent, MultipleRegressionContent
from .renderers import HTMLContentRenderer

logger = get_logger(__name__)


def create_flask_app() -> Flask:
    """Create and configure Flask application."""
    app = Flask(__name__, template_folder='templates')
    
    # Initialize pipeline
    pipeline = RegressionPipeline()
    
    @app.route('/')
    def index():
        """Landing page."""
        return render_template('index.html')
    
    @app.route('/simple')
    def simple_regression():
        """Simple regression analysis page."""
        # Get parameters
        dataset = request.args.get('dataset', 'Bildung & Einkommen')
        n_points = int(request.args.get('n', 50))
        
        # Get data config
        config = _get_simple_data_config(dataset, n_points)
        
        # Map dataset name to pipeline dataset name
        dataset_map = {
            "Bildung & Einkommen": "electronics",
            "Grösse & Gewicht": "temperature",
            "Temperatur & Eisverkauf": "temperature",
        }
        
        # Run pipeline
        pipeline_result = pipeline.run_simple(
            dataset=dataset_map.get(dataset, "electronics"),
            n=n_points,
            seed=42
        )
        
        data = pipeline_result.data
        # Override labels from our config
        data.x_label = config["x_label"]
        data.y_label = config["y_label"]
        data.context_title = config["title"]
        data.context_description = config["description"]
        
        stats_result = pipeline_result.stats
        
        # Prepare stats dictionary
        stats_dict = _prepare_simple_stats(data, stats_result)
        
        # Build content using framework-agnostic ContentBuilder
        content_builder = SimpleRegressionContent(stats_dict, {})
        content = content_builder.build()
        
        # Render using HTML renderer
        renderer = HTMLContentRenderer(
            plots={},
            data={
                "x": data.x.tolist(),
                "y": data.y.tolist(),
                "x_label": data.x_label,
                "y_label": data.y_label,
            },
            stats=stats_dict
        )
        
        # Get rendered content
        content_dict = renderer.render_to_dict(content)
        
        return render_template(
            'educational_content.html',
            title=content.title,
            subtitle=content.subtitle,
            content_html=content_dict['full_html'],
            chapters=content_dict['chapters'],
            analysis_type='simple',
            dataset=dataset,
            n_points=n_points,
            stats=stats_dict
        )
    
    @app.route('/multiple')
    def multiple_regression():
        """Multiple regression analysis page."""
        # Get parameters
        dataset = request.args.get('dataset', 'Immobilienpreise')
        n_points = int(request.args.get('n', 75))
        
        # Get data config
        config = _get_multiple_data_config(dataset, n_points)
        
        # Map dataset name to pipeline dataset name
        dataset_map = {
            "Immobilienpreise": "houses",
            "Autoverbrauch": "cities",
            "Marketing-Mix": "cities",
        }
        
        # Run pipeline
        pipeline_result = pipeline.run_multiple(
            dataset=dataset_map.get(dataset, "cities"),
            n=n_points,
            seed=42
        )
        
        data = pipeline_result.data
        stats_result = pipeline_result.stats
        
        x1 = data.x1
        x2 = data.x2
        y = data.y
        
        # Prepare stats dictionary
        stats_dict = _prepare_multiple_stats(config, stats_result, x1, x2, y, n_points)
        
        # Build content
        content_builder = MultipleRegressionContent(stats_dict, {})
        content = content_builder.build()
        
        # Render
        renderer = HTMLContentRenderer(
            plots={},
            data={
                "x1": x1.tolist(),
                "x2": x2.tolist(),
                "y": y.tolist(),
                "x1_label": config["x1_label"],
                "x2_label": config["x2_label"],
                "y_label": config["y_label"],
            },
            stats=stats_dict
        )
        
        content_dict = renderer.render_to_dict(content)
        
        return render_template(
            'educational_content.html',
            title=content.title,
            subtitle=content.subtitle,
            content_html=content_dict['full_html'],
            chapters=content_dict['chapters'],
            analysis_type='multiple',
            dataset=dataset,
            n_points=n_points,
            stats=stats_dict
        )
    
    @app.route('/api/analyze', methods=['POST'])
    def api_analyze():
        """API endpoint for analysis."""
        data = request.get_json()
        
        analysis_type = data.get('type', 'simple')
        x = np.array(data.get('x', []))
        y = np.array(data.get('y', []))
        
        if analysis_type == 'simple':
            result = pipeline.calculate_simple(x, y)
            return jsonify({
                'intercept': result.intercept,
                'slope': result.slope,
                'r_squared': result.r_squared,
                'p_slope': result.p_slope,
            })
        else:
            x1 = x
            x2 = np.array(data.get('x2', []))
            result = pipeline.calculate_multiple(x1, x2, y)
            return jsonify({
                'intercept': result.intercept,
                'betas': list(result.betas) if hasattr(result, 'betas') else [],
                'r_squared': result.r_squared,
            })
    
    return app


def _get_simple_data_config(dataset: str, n: int) -> Dict[str, Any]:
    """Get data configuration for simple regression."""
    configs = {
        "Bildung & Einkommen": {
            "x_label": "Bildungsjahre",
            "y_label": "Jahreseinkommen (CHF)",
            "title": "Bildung und Einkommen",
            "description": "Untersucht den Zusammenhang zwischen Bildungsjahren und Einkommen.",
            "y_unit": "CHF"
        },
        "Grösse & Gewicht": {
            "x_label": "Körpergrösse (cm)",
            "y_label": "Körpergewicht (kg)",
            "title": "Körpergrösse und Gewicht",
            "description": "Analysiert den Zusammenhang zwischen Körpergrösse und Gewicht.",
            "y_unit": "kg"
        },
        "Temperatur & Eisverkauf": {
            "x_label": "Temperatur (°C)",
            "y_label": "Eisverkauf (Einheiten)",
            "title": "Temperatur und Eisverkauf",
            "description": "Untersucht wie die Temperatur den Eisverkauf beeinflusst.",
            "y_unit": "Einheiten"
        },
    }
    return configs.get(dataset, {
        "x_label": "X",
        "y_label": "Y",
        "title": "Benutzerdefinierte Daten",
        "description": "Analyse mit benutzerdefinierten Daten.",
        "y_unit": "Einheiten"
    })


def _get_multiple_data_config(dataset: str, n: int) -> Dict[str, Any]:
    """Get data configuration for multiple regression."""
    configs = {
        "Immobilienpreise": {
            "x1_label": "Wohnfläche (m²)",
            "x2_label": "Zimmer",
            "y_label": "Preis (CHF)",
            "title": "Immobilienpreise",
            "description": "Preis basierend auf Fläche und Zimmeranzahl.",
            "x1_range": (50, 200),
            "x2_range": (2, 6),
            "true_intercept": 100000,
            "true_beta1": 3000,
            "true_beta2": 50000,
            "noise": 50000
        },
        "Autoverbrauch": {
            "x1_label": "Gewicht (kg)",
            "x2_label": "PS",
            "y_label": "Verbrauch (L/100km)",
            "title": "Autoverbrauch",
            "description": "Kraftstoffverbrauch basierend auf Gewicht und Leistung.",
            "x1_range": (1000, 2500),
            "x2_range": (80, 300),
            "true_intercept": 2,
            "true_beta1": 0.003,
            "true_beta2": 0.02,
            "noise": 1
        },
        "Marketing-Mix": {
            "x1_label": "TV-Budget (TCHF)",
            "x2_label": "Online-Budget (TCHF)",
            "y_label": "Umsatz (TCHF)",
            "title": "Marketing-Mix Analyse",
            "description": "Umsatz basierend auf Werbeausgaben.",
            "x1_range": (10, 100),
            "x2_range": (5, 50),
            "true_intercept": 50,
            "true_beta1": 1.5,
            "true_beta2": 2.0,
            "noise": 20
        },
    }
    return configs.get(dataset, {
        "x1_label": "X₁",
        "x2_label": "X₂",
        "y_label": "Y",
        "title": "Benutzerdefinierte Daten",
        "description": "Multiple Regression mit benutzerdefinierten Daten.",
        "x1_range": (0, 100),
        "x2_range": (0, 100),
        "true_intercept": 10,
        "true_beta1": 2,
        "true_beta2": 3,
        "noise": 10
    })


def _prepare_simple_stats(data, stats_result) -> Dict[str, Any]:
    """Prepare statistics dictionary for simple regression content."""
    from scipy import stats as scipy_stats
    
    x, y = data.x, data.y
    n = len(x)
    
    corr = np.corrcoef(x, y)[0, 1]
    t_corr = corr * np.sqrt((n - 2) / (1 - corr**2)) if abs(corr) < 1 else 0
    p_corr = 2 * (1 - scipy_stats.t.cdf(abs(t_corr), df=n-2)) if abs(corr) < 1 else 0
    
    spearman_r, spearman_p = scipy_stats.spearmanr(x, y)
    
    msr = stats_result.ssr / 1 if stats_result.ssr else 0
    mse = stats_result.sse / stats_result.df if stats_result.sse and stats_result.df else 1
    f_stat = msr / mse if mse else 0
    p_f = 1 - scipy_stats.f.cdf(f_stat, dfn=1, dfd=stats_result.df) if f_stat else 1
    
    return {
        "context_title": data.context_title,
        "context_description": data.context_description,
        "x_label": data.x_label,
        "y_label": data.y_label,
        "y_unit": getattr(data, 'y_unit', ''),
        "n": n,
        "x_mean": float(np.mean(x)),
        "x_std": float(np.std(x, ddof=1)),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y, ddof=1)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "correlation": float(corr),
        "covariance": float(np.cov(x, y, ddof=1)[0, 1]),
        "t_correlation": float(t_corr),
        "p_correlation": float(p_corr),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "intercept": float(stats_result.intercept),
        "slope": float(stats_result.slope),
        "se_intercept": float(stats_result.se_intercept),
        "se_slope": float(stats_result.se_slope),
        "t_intercept": float(stats_result.t_intercept),
        "t_slope": float(stats_result.t_slope),
        "p_intercept": float(stats_result.p_intercept),
        "p_slope": float(stats_result.p_slope),
        "r_squared": float(stats_result.r_squared),
        "r_squared_adj": float(stats_result.r_squared_adj),
        "mse": float(stats_result.mse),
        "sse": float(stats_result.sse),
        "ssr": float(stats_result.ssr),
        "sst": float(stats_result.sst),
        "df": int(stats_result.df),
        "f_statistic": float(f_stat),
        "p_f": float(p_f),
        "residuals": stats_result.residuals.tolist(),
        "y_pred": stats_result.y_pred.tolist(),
    }


def _prepare_multiple_stats(
    config: Dict[str, Any],
    stats_result,
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    n: int
) -> Dict[str, Any]:
    """Prepare statistics dictionary for multiple regression content."""
    corr_x1_x2 = float(np.corrcoef(x1, x2)[0, 1])
    r2_x1 = corr_x1_x2**2
    vif = 1 / (1 - r2_x1) if r2_x1 < 1 else float('inf')
    
    return {
        "context_title": config["title"],
        "context_description": config["description"],
        "x1_label": config["x1_label"],
        "x2_label": config["x2_label"],
        "y_label": config["y_label"],
        "n": n,
        "k": 2,
        "intercept": float(stats_result.intercept),
        "beta1": float(stats_result.betas[0]) if hasattr(stats_result, 'betas') else 0,
        "beta2": float(stats_result.betas[1]) if hasattr(stats_result, 'betas') else 0,
        "se_intercept": float(stats_result.se_intercept) if hasattr(stats_result, 'se_intercept') else 0,
        "se_beta1": float(stats_result.se_betas[0]) if hasattr(stats_result, 'se_betas') else 0,
        "se_beta2": float(stats_result.se_betas[1]) if hasattr(stats_result, 'se_betas') else 0,
        "t_intercept": float(stats_result.t_intercept) if hasattr(stats_result, 't_intercept') else 0,
        "t_beta1": float(stats_result.t_betas[0]) if hasattr(stats_result, 't_betas') else 0,
        "t_beta2": float(stats_result.t_betas[1]) if hasattr(stats_result, 't_betas') else 0,
        "p_intercept": float(stats_result.p_intercept) if hasattr(stats_result, 'p_intercept') else 1,
        "p_beta1": float(stats_result.p_betas[0]) if hasattr(stats_result, 'p_betas') else 1,
        "p_beta2": float(stats_result.p_betas[1]) if hasattr(stats_result, 'p_betas') else 1,
        "r_squared": float(stats_result.r_squared),
        "r_squared_adj": float(stats_result.r_squared_adj),
        "f_statistic": float(stats_result.f_statistic) if hasattr(stats_result, 'f_statistic') else 0,
        "p_f": float(stats_result.p_f) if hasattr(stats_result, 'p_f') else 1,
        "df": n - 3,
        "corr_x1_x2": corr_x1_x2,
        "vif_x1": float(vif),
        "vif_x2": float(vif),
        "durbin_watson": 2.0,
        "residuals": stats_result.residuals.tolist() if hasattr(stats_result, 'residuals') else [],
    }


def run_flask():
    """Run Flask application."""
    app = create_flask_app()
    app.run(debug=True, host='0.0.0.0', port=5000)


if __name__ == "__main__":
    run_flask()
