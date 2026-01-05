"""
R-style output display for regression models.

This module provides functions to render R-style statistical output
for regression models in the Streamlit application.
"""

import streamlit as st
from typing import Optional, List, Any

from .plots import create_r_output_figure
from .logger import get_logger

logger = get_logger(__name__)


def render_r_output_section(
    model: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
    figsize: tuple = (18, 13)
) -> None:
    """
    Render the R output section with model summary and explanation.
    
    Args:
        model: Fitted regression model (statsmodels)
        feature_names: List of feature names for the model
        figsize: Figure size as (width, height)
    """
    logger.debug(f"Rendering R output section (model={'present' if model else 'absent'})")
    
    st.markdown("---")
    
    # Create two columns: R output on left, explanation on right
    col_r_output, col_r_explanation = st.columns([3, 2])
    
    with col_r_output:
        st.markdown("### 📊 R Output (Automatisch aktualisiert)")
        
        # Display R output based on current model
        try:
            if model is not None and feature_names is not None:
                fig_r = create_r_output_figure(
                    model, 
                    feature_names=feature_names, 
                    figsize=figsize
                )
                st.plotly_chart(fig_r, use_container_width=True)
            else:
                st.info("ℹ️ Wählen Sie einen Datensatz und Parameter aus, um das R Output zu sehen.")
        except Exception as e:
            st.warning(f"R Output konnte nicht geladen werden: {str(e)}")
            logger.error(f"Error rendering R output: {e}", exc_info=True)
    
    with col_r_explanation:
        _render_r_output_explanation()
    
    st.markdown("---")


def _render_r_output_explanation() -> None:
    """
    Render the explanation panel for R output sections.
    
    This is a private helper function that provides detailed explanations
    of each section in the R output.
    """
    with st.expander("📖 Erklärung der R Output Abschnitte", expanded=False):
        st.markdown("""
        #### Erklärung der Abschnitte (kurz, präzise)
        • **Call**: zeigt die verwendete Modellformel und das Datenset; nützlich zur Reproduzierbarkeit.

        • **Residuals**: fünf‑Zahlen‑Zusammenfassung der Residuen (Min, 1Q, Median, 3Q, Max) zur schnellen Beurteilung von Schiefe/Ausreißern.

        • **Coefficients**: vier Spalten: Estimate, Std. Error, t value, Pr(>|t|); jede Zeile ist ein Prädiktor (Intercept inklusive). Signifikanzsterne werden darunter erklärt.

        • **Residual standard error und degrees of freedom**: Schätzung der Fehlerstreuung und Freiheitsgrade für Tests.

        • **Multiple R-squared / Adjusted R-squared**: erklärte Varianz und bereinigte Version (bestraft unnötige Prädiktoren).

        • **F-statistic**: globaler Test, ob mindestens ein Prädiktor das Modell signifikant verbessert; p‑value dazu wird angezeigt.

        ---
        #### Wichtige Hinweise, Entscheidungen und Risiken
        • **Interpretation der Koeffizienten**: Ein Estimate ist die geschätzte Änderung in der Zielvariable pro Einheitenänderung des Prädiktors bei konstanten anderen Variablen; Pr(>|t|) gibt die zweiseitige p‑Wert‑Signifikanz an.

        • **Achtung bei Multikollinearität**: hohe Standardfehler oder aliasing können Koeffizienten unzuverlässig machen; summary() zeigt aliased coefficients nicht, Details in summary.lm‑Dokumentation.
        """)


def render_r_output_from_session_state() -> None:
    """
    Render R output using model and feature names from session state.
    
    This is a convenience function that retrieves the current model
    from session state and renders the R output section.
    """
    logger.debug("Rendering R output from session state")
    
    # Get model from session state
    model = st.session_state.get("current_model")
    feature_names = st.session_state.get("current_feature_names")
    
    # Render the R output section
    render_r_output_section(model=model, feature_names=feature_names)
