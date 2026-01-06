"""
Data loading module for the Linear Regression Guide.

This module provides a simplified interface to the data and model services,
using proper statsmodels OLS for real regression calculations.
"""

from typing import Dict, Any, Optional
import numpy as np
import statsmodels.api as sm


def _map_dataset_name(display_name: str, regression_type: str) -> str:
    """
    Map UI display names to internal dataset names.
    
    Args:
        display_name: The display name with emojis from the UI
        regression_type: Either 'simple' or 'multiple'
    
    Returns:
        Internal dataset name used by generators
    """
    if regression_type == 'multiple':
        multiple_mappings = {
            "🏙️ Städte-Umsatzstudie (75 Städte)": "Cities",
            "🏠 Häuserpreise mit Pool (1000 Häuser)": "Houses",
            "🏪 Elektronikmarkt (simuliert)": "Electronics",
        }
        return multiple_mappings.get(display_name, "Cities")
    else:
        simple_mappings = {
            "🏙️ Advertising Study (75 Städte)": "advertising",
            "🏪 Elektronikmarkt (simuliert)": "electronics",
            "🏙️ Städte-Umsatzstudie (75 Städte)": "advertising",
        }
        return simple_mappings.get(display_name, "advertising")


def load_multiple_regression_data(
    dataset_choice: str,
    n: int,
    noise_level: float,
    seed: int
) -> Dict[str, Any]:
    """
    Load and prepare multiple regression data with REAL statsmodels OLS.

    Args:
        dataset_choice: Name of the dataset to load
        n: Number of observations
        noise_level: Noise level for data generation
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing all prepared data and model results
    """
    from .data_generators.multiple_regression_generator import generate_multiple_regression_data

    internal_name = _map_dataset_name(dataset_choice, 'multiple')
    
    try:
        raw_data = generate_multiple_regression_data(internal_name, n, noise_level, seed)
    except ValueError:
        raw_data = generate_multiple_regression_data("Cities", n, noise_level, seed)

    # Extract predictor and response arrays
    x1 = np.array(raw_data.get("x2_preis", raw_data.get("x1", np.random.randn(n))))
    x2 = np.array(raw_data.get("x3_werbung", raw_data.get("x2", np.random.randn(n))))
    y = np.array(raw_data.get("y_mult", raw_data.get("y", np.random.randn(n))))
    
    # Fit REAL statsmodels OLS model
    X = sm.add_constant(np.column_stack([x1, x2]))
    model = sm.OLS(y, X).fit()
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Build coefficients info from REAL model
    mult_coeffs = {
        "params": list(model.params),
        "bse": list(model.bse),
        "tvalues": list(model.tvalues),
        "pvalues": list(model.pvalues),
    }
    
    # Build summary from REAL model
    mult_summary = {
        "rsquared": float(model.rsquared),
        "rsquared_adj": float(model.rsquared_adj),
        "fvalue": float(model.fvalue) if hasattr(model, 'fvalue') and model.fvalue is not None else 0.0,
        "f_pvalue": float(model.f_pvalue) if hasattr(model, 'f_pvalue') and model.f_pvalue is not None else 0.0,
    }
    
    # Build diagnostics
    mult_diagnostics = {
        "resid": residuals,
        "sse": float(np.sum(residuals ** 2)),
    }
    
    # Get labels
    x1_name = raw_data.get("x1_name", "Variable 1")
    x2_name = raw_data.get("x2_name", "Variable 2")
    y_name = raw_data.get("y_name", "Zielvariable")
    
    return {
        "x2_preis": x1,
        "x3_werbung": x2,
        "y_mult": y,
        "y_pred_mult": y_pred,
        "model_mult": model,
        "mult_coeffs": mult_coeffs,
        "mult_summary": mult_summary,
        "mult_diagnostics": mult_diagnostics,
        "x1_name": x1_name,
        "x2_name": x2_name,
        "y_name": y_name,
    }


def load_simple_regression_data(
    dataset_choice: str,
    x_variable: Optional[str],
    n: int,
    true_intercept: float = 0,
    true_beta: float = 0,
    noise_level: float = 0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Load and prepare simple regression data with REAL statsmodels OLS.

    Args:
        dataset_choice: Name of the dataset to load
        x_variable: X variable to use (for multi-variable datasets)
        n: Number of observations
        true_intercept: True intercept (for simulated data)
        true_beta: True slope (for simulated data)
        noise_level: Noise level for data generation
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing all prepared data and model results
    """
    from .data_generators.simple_regression_generator import generate_simple_regression_data

    internal_name = _map_dataset_name(dataset_choice, 'simple')
    
    try:
        raw_data = generate_simple_regression_data(internal_name, n, noise_level, seed)
    except Exception:
        # Fallback: generate synthetic data
        np.random.seed(seed)
        x = np.random.uniform(2, 10, n)
        y = true_intercept + true_beta * x + np.random.normal(0, noise_level, n)
        raw_data = {
            "x": x,
            "y": y,
            "x_label": "X",
            "y_label": "Y",
        }

    # Extract arrays
    x = np.array(raw_data.get("x", np.random.randn(n)))
    y = np.array(raw_data.get("y", np.random.randn(n)))
    
    # Fit REAL statsmodels OLS model
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    y_pred = model.predict(X)
    residuals = model.resid
    
    # Extract coefficients from REAL model
    b0 = float(model.params[0])
    b1 = float(model.params[1])
    
    # Compute statistics
    n_obs = len(x)
    x_mean = float(np.mean(x))
    y_mean_val = float(np.mean(y))
    cov_xy = float(np.cov(x, y)[0, 1]) if n_obs > 1 else 0.0
    var_x = float(np.var(x, ddof=1)) if n_obs > 1 else 0.0
    var_y = float(np.var(y, ddof=1)) if n_obs > 1 else 0.0
    corr_xy = float(np.corrcoef(x, y)[0, 1]) if n_obs > 1 else 0.0
    
    # Compute sums of squares
    sse = float(np.sum(residuals ** 2))
    sst = float(np.sum((y - y_mean_val) ** 2))
    ssr = sst - sse
    mse = sse / (n_obs - 2) if n_obs > 2 else 0.0
    se_regression = float(np.sqrt(mse)) if mse > 0 else 0.0
    
    # Get labels and context
    x_label = raw_data.get("x_label", "X")
    y_label = raw_data.get("y_label", "Y")
    x_unit = raw_data.get("x_unit", "")
    y_unit = raw_data.get("y_unit", "")
    context_title = raw_data.get("context_title", dataset_choice)
    context_description = raw_data.get("context_description", "")
    
    return {
        "x": x,
        "y": y,
        "y_pred": y_pred,
        "residuals": residuals,
        "model": model,
        "b0": b0,
        "b1": b1,
        "x_label": x_label,
        "y_label": y_label,
        "x_unit": x_unit,
        "y_unit": y_unit,
        "context_title": context_title,
        "context_description": context_description,
        "x_mean": x_mean,
        "y_mean_val": y_mean_val,
        "cov_xy": cov_xy,
        "var_x": var_x,
        "var_y": var_y,
        "corr_xy": corr_xy,
        "sse": sse,
        "sst": sst,
        "ssr": ssr,
        "mse": mse,
        "se_regression": se_regression,
    }


def compute_simple_regression_model(
    x, y, x_label: str, y_label: str, n: int
) -> Dict[str, Any]:
    """
    Compute simple regression model using statsmodels OLS.

    Args:
        x: X variable data
        y: Y variable data
        x_label: Label for X variable
        y_label: Label for Y variable
        n: Number of observations

    Returns:
        Dictionary containing model and all computed statistics
    """
    x = np.array(x)
    y = np.array(y)
    
    # Fit REAL model
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    y_pred = model.predict(X)
    
    return {
        'model': model,
        'x': x,
        'y': y,
        'x_label': x_label,
        'y_label': y_label,
        'y_pred': y_pred,
        'b0': float(model.params[0]),
        'b1': float(model.params[1]),
        'residuals': model.resid,
    }
