"""
🎓 Leitfaden zur Linearen Regression
====================================

Ein didaktisches Tool mit klarer 4-Stufen-Pipeline:
    1. GET      → Daten holen
    2. CALCULATE → Statistiken berechnen
    3. PLOT     → Visualisierungen erstellen
    4. DISPLAY  → Im UI anzeigen

Start: streamlit run src/app.py
"""

import warnings
import streamlit as st

# Suppress warnings
warnings.filterwarnings('ignore')

# Path setup for imports
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for path_dir in [current_dir, parent_dir]:
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)

# Import our pipeline
from .pipeline import RegressionPipeline
from .config import get_logger

logger = get_logger(__name__)


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="📖 Leitfaden Lineare Regression",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; }
    .section-header { font-size: 1.6rem; font-weight: bold; color: #2c3e50; border-bottom: 2px solid #1f77b4; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR - Parameters
# =============================================================================
st.sidebar.markdown("# ⚙️ Parameter")

# Tab selection
analysis_type = st.sidebar.radio(
    "Analyse-Typ",
    ["Einfache Regression", "Multiple Regression"],
    horizontal=True,
)

st.sidebar.markdown("---")

if analysis_type == "Einfache Regression":
    # Simple regression parameters
    st.sidebar.markdown("### 📈 Einfache Regression")
    
    dataset = st.sidebar.selectbox(
        "Datensatz",
        ["electronics", "advertising", "temperature"],
        format_func=lambda x: {
            "electronics": "🏪 Elektronikmarkt",
            "advertising": "📢 Werbestudie",
            "temperature": "🍦 Eisverkauf",
        }.get(x, x)
    )
    
    n = st.sidebar.slider("Stichprobengrösse (n)", 10, 100, 50)
    noise = st.sidebar.slider("Rauschen (σ)", 0.1, 2.0, 0.4, 0.1)
    seed = st.sidebar.number_input("Random Seed", 1, 9999, 42)
    
    with st.sidebar.expander("🎯 Wahre Parameter"):
        true_intercept = st.slider("β₀ (Intercept)", -2.0, 3.0, 0.6, 0.1)
        true_slope = st.slider("β₁ (Steigung)", 0.1, 2.0, 0.52, 0.01)
        show_true_line = st.checkbox("Wahre Linie zeigen", True)
    
    show_formulas = st.sidebar.checkbox("Formeln anzeigen", True)

else:
    # Multiple regression parameters
    st.sidebar.markdown("### 📊 Multiple Regression")
    
    dataset = st.sidebar.selectbox(
        "Datensatz",
        ["cities", "houses"],
        format_func=lambda x: {
            "cities": "🏙️ Städte-Umsatzstudie",
            "houses": "🏠 Häuserpreise",
        }.get(x, x)
    )
    
    n = st.sidebar.slider("Stichprobengrösse (n)", 20, 200, 75)
    noise = st.sidebar.slider("Rauschen (σ)", 1.0, 10.0, 3.5, 0.5)
    seed = st.sidebar.number_input("Random Seed", 1, 9999, 42)
    show_formulas = st.sidebar.checkbox("Formeln anzeigen", True)
    
    # Set defaults for simple regression params
    true_intercept = 0
    true_slope = 0
    show_true_line = False


# =============================================================================
# MAIN CONTENT - Run Pipeline
# =============================================================================

# Initialize pipeline
pipeline = RegressionPipeline()

# Header
st.markdown('<p class="main-header">📖 Leitfaden zur Linearen Regression</p>', unsafe_allow_html=True)
st.markdown("### Von der Frage zur validierten Erkenntnis")

# Show pipeline steps
with st.expander("🔄 Pipeline-Schritte", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    col1.info("**1. GET**\n\nDaten holen")
    col2.info("**2. CALCULATE**\n\nStatistiken berechnen")
    col3.info("**3. PLOT**\n\nVisualisierungen")
    col4.info("**4. DISPLAY**\n\nIm UI anzeigen")

st.markdown("---")

# Run the appropriate pipeline
try:
    if analysis_type == "Einfache Regression":
        # Run simple regression pipeline
        result = pipeline.run_simple(
            dataset=dataset,
            n=n,
            noise=noise,
            seed=seed,
            true_intercept=true_intercept,
            true_slope=true_slope,
            show_true_line=show_true_line,
        )
        
        # Display results
        pipeline.display(result, show_formulas=show_formulas)
        
    else:
        # Run multiple regression pipeline
        result = pipeline.run_multiple(
            dataset=dataset,
            n=n,
            noise=noise,
            seed=seed,
        )
        
        # Display results
        pipeline.display(result, show_formulas=show_formulas)

except Exception as e:
    logger.error(f"Pipeline error: {e}")
    st.error(f"❌ Fehler in der Pipeline: {str(e)}")
    st.info("💡 Versuchen Sie andere Parameter oder laden Sie die Seite neu.")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    📖 Leitfaden zur Linearen Regression | 
    Simple 4-Step Pipeline: GET → CALCULATE → PLOT → DISPLAY
</div>
""", unsafe_allow_html=True)

logger.info("Application rendered successfully")
