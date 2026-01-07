"""
Step 4: DISPLAY

This module handles UI rendering.
It displays plots and statistics in Streamlit.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd
import numpy as np

from ..config import get_logger
from .get_data import DataResult, MultipleRegressionDataResult
from .calculate import RegressionResult, MultipleRegressionResult
from .plot import PlotCollection

logger = get_logger(__name__)


def get_signif_stars(p: float) -> str:
    """Get significance stars like R."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


class UIRenderer:
    """
    Step 4: DISPLAY
    
    Renders all UI components in Streamlit.
    
    Example:
        renderer = UIRenderer()
        renderer.simple_regression(data, result, plots)
    """
    
    def __init__(self):
        logger.info("UIRenderer initialized")
    
    def simple_regression(
        self,
        data: DataResult,
        result: RegressionResult,
        plots: PlotCollection,
        show_formulas: bool = True,
    ) -> None:
        """
        Display complete simple regression analysis.
        
        Args:
            data: Original data
            result: Regression results
            plots: Plot collection
            show_formulas: Whether to show mathematical formulas
        """
        # Header
        st.markdown("# 📈 Einfache Lineare Regression")
        st.markdown(f"### {data.context_title}")
        st.info(data.context_description)
        
        # Key metrics
        self._display_key_metrics_simple(result)
        
        # Main scatter plot
        st.plotly_chart(plots.scatter, use_container_width=True, key="main_scatter")
        
        # Model equation
        self._display_equation_simple(result, data, show_formulas)
        
        # Coefficient table
        self._display_coefficient_table_simple(result, data)
        
        # Residual analysis
        st.markdown("---")
        st.markdown("## 📊 Residuenanalyse")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plots.residuals, use_container_width=True, key="residuals")
        with col2:
            if plots.diagnostics:
                st.plotly_chart(plots.diagnostics, use_container_width=True, key="diagnostics")
        
        # Data table
        with st.expander("📋 Daten anzeigen"):
            df = pd.DataFrame({
                data.x_label: data.x,
                data.y_label: data.y,
                "ŷ (Predicted)": result.y_pred,
                "Residuum": result.residuals,
            })
            st.dataframe(df.style.format("{:.4f}"), use_container_width=True)
    
    def multiple_regression(
        self,
        data: MultipleRegressionDataResult,
        result: MultipleRegressionResult,
        plots: PlotCollection,
        show_formulas: bool = True,
    ) -> None:
        """
        Display complete multiple regression analysis.
        
        Args:
            data: Original data
            result: Multiple regression results
            plots: Plot collection
            show_formulas: Whether to show mathematical formulas
        """
        # Header
        st.markdown("# 📊 Multiple Regression")
        
        # Key metrics
        self._display_key_metrics_multiple(result)
        
        # 3D plot
        st.plotly_chart(plots.scatter, use_container_width=True, key="scatter_3d")
        
        # Model equation
        self._display_equation_multiple(result, data, show_formulas)
        
        # Coefficient table
        self._display_coefficient_table_multiple(result, data)
        
        # Residual analysis
        st.markdown("---")
        st.markdown("## 📊 Residuenanalyse")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plots.residuals, use_container_width=True, key="resid_mult")
        with col2:
            if plots.diagnostics:
                st.plotly_chart(plots.diagnostics, use_container_width=True, key="diag_mult")
    
    def sidebar_simple(
        self,
        datasets: list = None,
    ) -> Dict[str, Any]:
        """
        Render sidebar controls for simple regression.
        
        Returns:
            Dictionary with all selected parameters
        """
        if datasets is None:
            datasets = ["electronics", "advertising", "temperature"]
        
        st.sidebar.markdown("## ⚙️ Parameter")
        
        dataset = st.sidebar.selectbox(
            "Datensatz",
            datasets,
            format_func=lambda x: {
                "electronics": "🏪 Elektronikmarkt",
                "advertising": "📢 Werbestudie",
                "temperature": "🍦 Eisverkauf",
            }.get(x, x)
        )
        
        n = st.sidebar.slider("Stichprobengrösse (n)", 10, 100, 50)
        noise = st.sidebar.slider("Rauschen (σ)", 0.1, 2.0, 0.4, 0.1)
        seed = st.sidebar.number_input("Random Seed", 1, 9999, 42)
        
        with st.sidebar.expander("Wahre Parameter"):
            true_intercept = st.slider("β₀ (Intercept)", -2.0, 3.0, 0.6, 0.1)
            true_slope = st.slider("β₁ (Steigung)", 0.1, 2.0, 0.52, 0.01)
            show_true = st.checkbox("Wahre Linie zeigen", True)
        
        show_formulas = st.sidebar.checkbox("Formeln anzeigen", True)
        
        return {
            "dataset": dataset,
            "n": n,
            "noise": noise,
            "seed": seed,
            "true_intercept": true_intercept,
            "true_slope": true_slope,
            "show_true_line": show_true,
            "show_formulas": show_formulas,
        }
    
    def sidebar_multiple(
        self,
        datasets: list = None,
    ) -> Dict[str, Any]:
        """
        Render sidebar controls for multiple regression.
        
        Returns:
            Dictionary with selected parameters
        """
        if datasets is None:
            datasets = ["cities", "houses"]
        
        st.sidebar.markdown("## ⚙️ Parameter")
        
        dataset = st.sidebar.selectbox(
            "Datensatz",
            datasets,
            format_func=lambda x: {
                "cities": "🏙️ Städte-Umsatzstudie",
                "houses": "🏠 Häuserpreise",
            }.get(x, x)
        )
        
        n = st.sidebar.slider("Stichprobengrösse (n)", 20, 200, 75)
        noise = st.sidebar.slider("Rauschen (σ)", 1.0, 10.0, 3.5, 0.5)
        seed = st.sidebar.number_input("Random Seed", 1, 9999, 42)
        show_formulas = st.sidebar.checkbox("Formeln anzeigen", True)
        
        return {
            "dataset": dataset,
            "n": n,
            "noise": noise,
            "seed": seed,
            "show_formulas": show_formulas,
        }
    
    # =========================================================
    # PRIVATE: Display Helpers
    # =========================================================
    
    def _display_key_metrics_simple(self, result: RegressionResult) -> None:
        """Display key metrics in columns."""
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("R²", f"{result.r_squared:.4f}")
        col2.metric("β₀ (Intercept)", f"{result.intercept:.4f}")
        col3.metric("β₁ (Steigung)", f"{result.slope:.4f}")
        col4.metric("n", f"{result.n}")
    
    def _display_key_metrics_multiple(self, result: MultipleRegressionResult) -> None:
        """Display key metrics for multiple regression."""
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("R²", f"{result.r_squared:.4f}")
        col2.metric("R² adj", f"{result.r_squared_adj:.4f}")
        col3.metric("F-Statistik", f"{result.f_statistic:.2f}")
        col4.metric("n", f"{result.n}")
    
    def _display_equation_simple(
        self, result: RegressionResult, data: DataResult, show_formulas: bool
    ) -> None:
        """Display regression equation."""
        st.markdown("### 📐 Regressionsgleichung")
        
        sign = "+" if result.slope >= 0 else ""
        st.success(f"**ŷ = {result.intercept:.4f} {sign} {result.slope:.4f} · x**")
        
        if show_formulas:
            st.markdown("**Interpretation:**")
            st.markdown(f"- Wenn {data.x_label} um 1 steigt, ändert sich {data.y_label} um **{result.slope:.4f}**")
            st.markdown(f"- Bei {data.x_label} = 0 ist der erwartete {data.y_label} = **{result.intercept:.4f}**")
    
    def _display_equation_multiple(
        self, result: MultipleRegressionResult, data: MultipleRegressionDataResult, show_formulas: bool
    ) -> None:
        """Display multiple regression equation."""
        st.markdown("### 📐 Regressionsgleichung")
        
        b0 = result.intercept
        b1, b2 = result.coefficients
        sign1 = "+" if b1 >= 0 else ""
        sign2 = "+" if b2 >= 0 else ""
        
        st.success(f"**ŷ = {b0:.3f} {sign1} {b1:.3f}·x₁ {sign2} {b2:.3f}·x₂**")
        
        if show_formulas:
            st.markdown("**Interpretation (ceteris paribus):**")
            st.markdown(f"- Pro Einheit {data.x1_label}: {data.y_label} ändert sich um **{b1:.3f}**")
            st.markdown(f"- Pro Einheit {data.x2_label}: {data.y_label} ändert sich um **{b2:.3f}**")
    
    def _display_coefficient_table_simple(
        self, result: RegressionResult, data: DataResult
    ) -> None:
        """Display coefficient table."""
        st.markdown("### 📋 Koeffizienten")
        
        df = pd.DataFrame({
            "Parameter": ["β₀ (Intercept)", f"β₁ ({data.x_label})"],
            "Schätzwert": [result.intercept, result.slope],
            "Std. Error": [result.se_intercept, result.se_slope],
            "t-Wert": [result.t_intercept, result.t_slope],
            "p-Wert": [result.p_intercept, result.p_slope],
            "Signif.": [get_signif_stars(result.p_intercept), get_signif_stars(result.p_slope)],
        })
        
        st.dataframe(
            df.style.format({
                "Schätzwert": "{:.4f}",
                "Std. Error": "{:.4f}",
                "t-Wert": "{:.3f}",
                "p-Wert": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        
        st.caption("Signif.: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")
    
    def _display_coefficient_table_multiple(
        self, result: MultipleRegressionResult, data: MultipleRegressionDataResult
    ) -> None:
        """Display coefficient table for multiple regression."""
        st.markdown("### 📋 Koeffizienten")
        
        labels = ["β₀ (Intercept)", f"β₁ ({data.x1_label})", f"β₂ ({data.x2_label})"]
        coefs = [result.intercept] + result.coefficients
        
        df = pd.DataFrame({
            "Parameter": labels,
            "Schätzwert": coefs,
            "Std. Error": result.se_coefficients,
            "t-Wert": result.t_values,
            "p-Wert": result.p_values,
            "Signif.": [get_signif_stars(p) for p in result.p_values],
        })
        
        st.dataframe(
            df.style.format({
                "Schätzwert": "{:.4f}",
                "Std. Error": "{:.4f}",
                "t-Wert": "{:.3f}",
                "p-Wert": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        
        st.caption("Signif.: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")
