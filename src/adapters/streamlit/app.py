"""
Streamlit Application - Universal Frontend.

Uses ContentBuilder + StreamlitContentRenderer for truly frontend-agnostic
educational content rendering.

The same content structure is used by Flask - only the renderer differs.
"""

import streamlit as st
import numpy as np
from typing import Dict, Any, Optional

from ...config import get_logger
from ...pipeline import RegressionPipeline
from ...content import SimpleRegressionContent, MultipleRegressionContent
from ..renderers import StreamlitContentRenderer

logger = get_logger(__name__)


def run_streamlit_app():
    """Main Streamlit application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="📊 Regression Analysis",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .section-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #1f77b4;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .metric-row {
            display: flex;
            gap: 1rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize pipeline
    pipeline = RegressionPipeline()
    
    # Sidebar controls
    with st.sidebar:
        st.title("⚙️ Einstellungen")
        
        analysis_type = st.radio(
            "Analyse-Typ",
            ["Einfache Regression", "Multiple Regression"],
            key="analysis_type"
        )
        
        st.markdown("---")
        
        # Dataset selection
        st.subheader("📊 Datensatz")
        
        if analysis_type == "Einfache Regression":
            dataset = st.selectbox(
                "Wähle Datensatz:",
                ["Bildung & Einkommen", "Grösse & Gewicht", "Temperatur & Eisverkauf", "Custom"],
                key="dataset_simple"
            )
            n_points = st.slider("Anzahl Datenpunkte", 20, 200, 50, key="n_simple")
        else:
            dataset = st.selectbox(
                "Wähle Datensatz:",
                ["Immobilienpreise", "Autoverbrauch", "Marketing-Mix", "Custom"],
                key="dataset_multiple"
            )
            n_points = st.slider("Anzahl Datenpunkte", 30, 200, 75, key="n_multiple")
        
        st.markdown("---")
        
        # Display options
        st.subheader("📖 Anzeige")
        show_formulas = st.checkbox("Formeln anzeigen", value=True)
        show_code = st.checkbox("R-Code anzeigen", value=False)
    
    # Main content
    if analysis_type == "Einfache Regression":
        render_simple_regression(pipeline, dataset, n_points, show_formulas)
    else:
        render_multiple_regression(pipeline, dataset, n_points, show_formulas)


def render_simple_regression(
    pipeline: RegressionPipeline, 
    dataset: str, 
    n_points: int,
    show_formulas: bool
):
    """Render simple regression analysis."""
    # Get data based on dataset selection
    data_config = _get_simple_data_config(dataset, n_points)
    
    # Map dataset name to pipeline dataset name
    dataset_map = {
        "Bildung & Einkommen": "electronics",
        "Grösse & Gewicht": "temperature",
        "Temperatur & Eisverkauf": "temperature",
        "Custom": "electronics",
    }
    
    # Run pipeline
    pipeline_result = pipeline.run_simple(
        dataset=dataset_map.get(dataset, "electronics"),
        n=n_points,
        seed=42
    )
    
    data = pipeline_result.data
    # Override labels from our config
    data.x_label = data_config["x_label"]
    data.y_label = data_config["y_label"]
    data.context_title = data_config["title"]
    data.context_description = data_config["description"]
    
    stats_result = pipeline_result.stats
    
    # Prepare stats dictionary for content builder
    stats_dict = _prepare_simple_stats(data, stats_result)
    
    # Build content using framework-agnostic ContentBuilder
    content_builder = SimpleRegressionContent(stats_dict, {})
    content = content_builder.build()
    
    # Render using Streamlit-specific renderer
    renderer = StreamlitContentRenderer(
        plots={},  # Plots are generated interactively
        data={
            "x": data.x,
            "y": data.y,
            "x_label": data.x_label,
            "y_label": data.y_label,
        },
        stats=stats_dict
    )
    
    renderer.render(content)
    
    # AI Interpretation Section
    _render_ai_interpretation(stats_dict)


def render_multiple_regression(
    pipeline: RegressionPipeline,
    dataset: str,
    n_points: int,
    show_formulas: bool
):
    """Render multiple regression analysis."""
    # Get data based on dataset selection
    data_config = _get_multiple_data_config(dataset, n_points)
    
    # Map dataset name to pipeline dataset name
    dataset_map = {
        "Immobilienpreise": "houses",
        "Autoverbrauch": "cities",
        "Marketing-Mix": "cities",
        "Custom": "cities",
    }
    
    # Run pipeline
    pipeline_result = pipeline.run_multiple(
        dataset=dataset_map.get(dataset, "cities"),
        n=n_points,
        seed=42
    )
    
    data = pipeline_result.data
    stats_result = pipeline_result.stats
    
    # Use data from pipeline
    x1 = data.x1
    x2 = data.x2
    y = data.y
    
    # Prepare stats dictionary
    stats_dict = _prepare_multiple_stats(data_config, stats_result, x1, x2, y, n_points)
    
    # Build content using framework-agnostic ContentBuilder
    content_builder = MultipleRegressionContent(stats_dict, {})
    content = content_builder.build()
    
    # Render using Streamlit-specific renderer
    renderer = StreamlitContentRenderer(
        plots={},
        data={
            "x1": x1,
            "x2": x2,
            "y": y,
            "x1_label": data_config["x1_label"],
            "x2_label": data_config["x2_label"],
            "y_label": data_config["y_label"],
        },
        stats=stats_dict
    )
    
    renderer.render(content)
    
    # AI Interpretation Section
    _render_ai_interpretation(stats_dict)


def _render_ai_interpretation(stats_dict: Dict[str, Any]):
    """Render AI interpretation section."""
    from ...ai import PerplexityClient
    
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); 
                padding: 1.5rem; border-radius: 1rem; color: white; margin: 1rem 0;">
        <h3 style="margin: 0; display: flex; align-items: center; gap: 0.5rem;">
            🤖 AI-Interpretation des R-Outputs
        </h3>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            Lass dir alle statistischen Werte gesamtheitlich von Perplexity AI erklären.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    client = PerplexityClient()
    
    # Status
    if client.is_configured:
        st.success("✅ Perplexity API verbunden")
    else:
        st.warning("⚠️ Kein API-Key konfiguriert - Fallback-Interpretation wird verwendet")
        st.caption("Setze `PERPLEXITY_API_KEY` als Umgebungsvariable oder in `.streamlit/secrets.toml`")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        Klicke auf **"R-Output interpretieren"** um eine vollständige, 
        gesamtheitliche Erklärung aller Werte zu erhalten:
        - Zusammenfassung des Modells
        - Interpretation der Koeffizienten
        - Bewertung der Modellgüte
        - Signifikanz-Erklärung
        - Praktische Bedeutung
        """)
    
    with col2:
        interpret_clicked = st.button(
            "🔍 R-Output interpretieren",
            type="primary",
            use_container_width=True,
            key="ai_interpret_btn"
        )
    
    # Session state for interpretation
    if "ai_interpretation_result" not in st.session_state:
        st.session_state.ai_interpretation_result = None
    
    if interpret_clicked:
        with st.spinner("🤖 AI analysiert den R-Output..."):
            response = client.interpret_r_output(stats_dict)
            st.session_state.ai_interpretation_result = response
    
    # Display interpretation
    if st.session_state.ai_interpretation_result:
        response = st.session_state.ai_interpretation_result
        
        st.markdown("### 📊 Interpretation")
        
        # Main content
        st.markdown(response.content)
        
        # Metadata
        if not response.error:
            meta_cols = st.columns(4)
            meta_cols[0].caption(f"📡 {response.model}")
            meta_cols[1].caption(f"⏱️ {response.latency_ms:.0f}ms")
            if response.usage:
                meta_cols[2].caption(f"📝 {response.usage.get('total_tokens', 'N/A')} Tokens")
            meta_cols[3].caption(f"💾 {'Cached' if response.cached else 'Live'}")
        
        # Citations
        if response.citations:
            with st.expander("📚 Quellen"):
                for i, citation in enumerate(response.citations, 1):
                    st.markdown(f"{i}. [{citation}]({citation})")


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
        "Custom": {
            "x_label": "X",
            "y_label": "Y",
            "title": "Benutzerdefinierte Daten",
            "description": "Analyse mit benutzerdefinierten Daten.",
            "y_unit": "Einheiten"
        }
    }
    return configs.get(dataset, configs["Custom"])


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
            "title": "Autoerbrauch",
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
        "Custom": {
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
        }
    }
    return configs.get(dataset, configs["Custom"])


def _prepare_simple_stats(data, stats_result) -> Dict[str, Any]:
    """Prepare statistics dictionary for simple regression content."""
    from scipy import stats as scipy_stats
    
    x, y = data.x, data.y
    n = len(x)
    
    # Correlation test
    corr = np.corrcoef(x, y)[0, 1]
    t_corr = corr * np.sqrt((n - 2) / (1 - corr**2)) if abs(corr) < 1 else 0
    p_corr = 2 * (1 - scipy_stats.t.cdf(abs(t_corr), df=n-2)) if abs(corr) < 1 else 0
    
    # Spearman
    spearman_r, spearman_p = scipy_stats.spearmanr(x, y)
    
    # F-statistic
    msr = stats_result.ssr / 1 if stats_result.ssr else 0
    mse = stats_result.sse / stats_result.df if stats_result.sse and stats_result.df else 1
    f_stat = msr / mse if mse else 0
    p_f = 1 - scipy_stats.f.cdf(f_stat, dfn=1, dfd=stats_result.df) if f_stat else 1
    
    return {
        # Context
        "context_title": data.context_title,
        "context_description": data.context_description,
        "x_label": data.x_label,
        "y_label": data.y_label,
        "y_unit": getattr(data, 'y_unit', ''),
        
        # Sample info
        "n": n,
        
        # Descriptive stats
        "x_mean": np.mean(x),
        "x_std": np.std(x, ddof=1),
        "x_min": np.min(x),
        "x_max": np.max(x),
        "y_mean": np.mean(y),
        "y_std": np.std(y, ddof=1),
        "y_min": np.min(y),
        "y_max": np.max(y),
        
        # Correlation
        "correlation": corr,
        "covariance": np.cov(x, y, ddof=1)[0, 1],
        "t_correlation": t_corr,
        "p_correlation": p_corr,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        
        # Regression coefficients
        "intercept": stats_result.intercept,
        "slope": stats_result.slope,
        "se_intercept": stats_result.se_intercept,
        "se_slope": stats_result.se_slope,
        "t_intercept": stats_result.t_intercept,
        "t_slope": stats_result.t_slope,
        "p_intercept": stats_result.p_intercept,
        "p_slope": stats_result.p_slope,
        
        # Model fit
        "r_squared": stats_result.r_squared,
        "r_squared_adj": stats_result.r_squared_adj,
        "mse": stats_result.mse,
        "sse": stats_result.sse,
        "ssr": stats_result.ssr,
        "sst": stats_result.sst,
        "df": stats_result.df,
        
        # F-test
        "f_statistic": f_stat,
        "p_f": p_f,
        
        # Residuals
        "residuals": stats_result.residuals,
        "y_pred": stats_result.y_pred,
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
    from scipy import stats as scipy_stats
    
    # VIF calculation
    corr_x1_x2 = np.corrcoef(x1, x2)[0, 1]
    r2_x1 = corr_x1_x2**2
    vif = 1 / (1 - r2_x1) if r2_x1 < 1 else float('inf')
    
    return {
        # Context
        "context_title": config["title"],
        "context_description": config["description"],
        "x1_label": config["x1_label"],
        "x2_label": config["x2_label"],
        "y_label": config["y_label"],
        
        # Sample info
        "n": n,
        "k": 2,
        
        # Coefficients
        "intercept": stats_result.intercept,
        "beta1": stats_result.betas[0] if hasattr(stats_result, 'betas') else 0,
        "beta2": stats_result.betas[1] if hasattr(stats_result, 'betas') else 0,
        "se_intercept": stats_result.se_intercept if hasattr(stats_result, 'se_intercept') else 0,
        "se_beta1": stats_result.se_betas[0] if hasattr(stats_result, 'se_betas') else 0,
        "se_beta2": stats_result.se_betas[1] if hasattr(stats_result, 'se_betas') else 0,
        "t_intercept": stats_result.t_intercept if hasattr(stats_result, 't_intercept') else 0,
        "t_beta1": stats_result.t_betas[0] if hasattr(stats_result, 't_betas') else 0,
        "t_beta2": stats_result.t_betas[1] if hasattr(stats_result, 't_betas') else 0,
        "p_intercept": stats_result.p_intercept if hasattr(stats_result, 'p_intercept') else 1,
        "p_beta1": stats_result.p_betas[0] if hasattr(stats_result, 'p_betas') else 1,
        "p_beta2": stats_result.p_betas[1] if hasattr(stats_result, 'p_betas') else 1,
        
        # Model fit
        "r_squared": stats_result.r_squared,
        "r_squared_adj": stats_result.r_squared_adj,
        "f_statistic": stats_result.f_statistic if hasattr(stats_result, 'f_statistic') else 0,
        "p_f": stats_result.p_f if hasattr(stats_result, 'p_f') else 1,
        "df": n - 3,
        
        # Multicollinearity
        "corr_x1_x2": corr_x1_x2,
        "vif_x1": vif,
        "vif_x2": vif,
        
        # Durbin-Watson
        "durbin_watson": 2.0,  # Placeholder
        
        # Residuals
        "residuals": stats_result.residuals if hasattr(stats_result, 'residuals') else [],
    }


if __name__ == "__main__":
    run_streamlit_app()
