"""
Data contracts for the Linear Regression Guide.

This module defines typed dataclasses that serve as contracts between
data generation, model fitting, and UI rendering layers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class SimpleRegressionResult:
    """
    Complete result of a simple linear regression analysis.
    
    This dataclass defines the contract for data passed between
    the data layer and the UI layer for simple regression.
    """
    # Core data
    x: np.ndarray
    y: np.ndarray
    y_pred: np.ndarray
    residuals: np.ndarray
    
    # Model object (statsmodels RegressionResultsWrapper)
    model: Any
    
    # Coefficients
    b0: float  # Intercept
    b1: float  # Slope
    
    # Labels and context
    x_label: str
    y_label: str
    x_unit: str = ""
    y_unit: str = ""
    context_title: str = ""
    context_description: str = ""
    
    # Statistics
    x_mean: float = 0.0
    y_mean_val: float = 0.0
    cov_xy: float = 0.0
    var_x: float = 0.0
    var_y: float = 0.0
    corr_xy: float = 0.0
    
    # Sum of squares
    sse: float = 0.0  # Sum of Squared Errors
    sst: float = 0.0  # Total Sum of Squares
    ssr: float = 0.0  # Regression Sum of Squares
    mse: float = 0.0  # Mean Squared Error
    se_regression: float = 0.0  # Standard Error of Regression
    
    def __post_init__(self):
        """Validate and compute derived values."""
        n = len(self.x)
        
        # Compute statistics if not provided
        if self.x_mean == 0.0:
            self.x_mean = float(np.mean(self.x))
        if self.y_mean_val == 0.0:
            self.y_mean_val = float(np.mean(self.y))
        if self.cov_xy == 0.0 and n > 1:
            self.cov_xy = float(np.cov(self.x, self.y)[0, 1])
        if self.var_x == 0.0 and n > 1:
            self.var_x = float(np.var(self.x, ddof=1))
        if self.var_y == 0.0 and n > 1:
            self.var_y = float(np.var(self.y, ddof=1))
        if self.corr_xy == 0.0 and n > 1:
            self.corr_xy = float(np.corrcoef(self.x, self.y)[0, 1])
            
        # Compute sums of squares if not provided
        if self.sse == 0.0:
            self.sse = float(np.sum((self.y - self.y_pred) ** 2))
        if self.sst == 0.0:
            self.sst = float(np.sum((self.y - self.y_mean_val) ** 2))
        if self.ssr == 0.0:
            self.ssr = self.sst - self.sse
        if self.mse == 0.0 and n > 2:
            self.mse = self.sse / (n - 2)
        if self.se_regression == 0.0 and self.mse > 0:
            self.se_regression = float(np.sqrt(self.mse))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            'x': self.x,
            'y': self.y,
            'y_pred': self.y_pred,
            'residuals': self.residuals,
            'model': self.model,
            'b0': self.b0,
            'b1': self.b1,
            'x_label': self.x_label,
            'y_label': self.y_label,
            'x_unit': self.x_unit,
            'y_unit': self.y_unit,
            'context_title': self.context_title,
            'context_description': self.context_description,
            'x_mean': self.x_mean,
            'y_mean_val': self.y_mean_val,
            'cov_xy': self.cov_xy,
            'var_x': self.var_x,
            'var_y': self.var_y,
            'corr_xy': self.corr_xy,
            'sse': self.sse,
            'sst': self.sst,
            'ssr': self.ssr,
            'mse': self.mse,
            'se_regression': self.se_regression,
        }


@dataclass
class MultipleRegressionResult:
    """
    Complete result of a multiple linear regression analysis.
    
    This dataclass defines the contract for data passed between
    the data layer and the UI layer for multiple regression.
    """
    # Predictor data
    x2_preis: np.ndarray
    x3_werbung: np.ndarray
    y_mult: np.ndarray
    y_pred_mult: np.ndarray
    
    # Model object (statsmodels RegressionResultsWrapper)
    model_mult: Any
    
    # Coefficients info
    mult_coeffs: Dict[str, Any]
    mult_summary: Dict[str, float]
    mult_diagnostics: Dict[str, Any]
    
    # Labels
    x1_name: str
    x2_name: str
    y_name: str
    
    def __post_init__(self):
        """Validate required fields."""
        if len(self.x2_preis) != len(self.x3_werbung):
            raise ValueError("x2_preis and x3_werbung must have same length")
        if len(self.x2_preis) != len(self.y_mult):
            raise ValueError("Predictors and response must have same length")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            'x2_preis': self.x2_preis,
            'x3_werbung': self.x3_werbung,
            'y_mult': self.y_mult,
            'y_pred_mult': self.y_pred_mult,
            'model_mult': self.model_mult,
            'mult_coeffs': self.mult_coeffs,
            'mult_summary': self.mult_summary,
            'mult_diagnostics': self.mult_diagnostics,
            'x1_name': self.x1_name,
            'x2_name': self.x2_name,
            'y_name': self.y_name,
        }


# Factory functions for creating regression results

def create_simple_regression_result(
    x: np.ndarray,
    y: np.ndarray,
    model: Any,
    x_label: str,
    y_label: str,
    **kwargs
) -> SimpleRegressionResult:
    """
    Factory function to create a SimpleRegressionResult from a fitted model.
    
    Args:
        x: X variable data
        y: Y variable data
        model: Fitted statsmodels OLS model
        x_label: Label for X variable
        y_label: Label for Y variable
        **kwargs: Additional optional fields
    
    Returns:
        Complete SimpleRegressionResult instance
    """
    import statsmodels.api as sm
    
    # Get predictions
    X = sm.add_constant(x)
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    return SimpleRegressionResult(
        x=x,
        y=y,
        y_pred=y_pred,
        residuals=residuals,
        model=model,
        b0=float(model.params[0]),
        b1=float(model.params[1]),
        x_label=x_label,
        y_label=y_label,
        **kwargs
    )


def create_multiple_regression_result(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    model: Any,
    x1_name: str,
    x2_name: str,
    y_name: str,
) -> MultipleRegressionResult:
    """
    Factory function to create a MultipleRegressionResult from a fitted model.
    
    Args:
        x1: First predictor data
        x2: Second predictor data
        y: Response variable data
        model: Fitted statsmodels OLS model
        x1_name: Name for first predictor
        x2_name: Name for second predictor
        y_name: Name for response variable
    
    Returns:
        Complete MultipleRegressionResult instance
    """
    import statsmodels.api as sm
    
    # Get predictions
    X = sm.add_constant(np.column_stack([x1, x2]))
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Build coefficients info
    mult_coeffs = {
        "params": list(model.params),
        "bse": list(model.bse),
        "tvalues": list(model.tvalues),
        "pvalues": list(model.pvalues),
    }
    
    # Build summary
    mult_summary = {
        "rsquared": float(model.rsquared),
        "rsquared_adj": float(model.rsquared_adj),
        "fvalue": float(model.fvalue) if hasattr(model, 'fvalue') else 0.0,
        "f_pvalue": float(model.f_pvalue) if hasattr(model, 'f_pvalue') else 0.0,
    }
    
    # Build diagnostics
    mult_diagnostics = {
        "resid": residuals,
        "sse": float(np.sum(residuals ** 2)),
    }
    
    return MultipleRegressionResult(
        x2_preis=x1,
        x3_werbung=x2,
        y_mult=y,
        y_pred_mult=y_pred,
        model_mult=model,
        mult_coeffs=mult_coeffs,
        mult_summary=mult_summary,
        mult_diagnostics=mult_diagnostics,
        x1_name=x1_name,
        x2_name=x2_name,
        y_name=y_name,
    )
