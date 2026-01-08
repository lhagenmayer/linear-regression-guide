"""
Infrastructure: Regression Service Implementation.
Implements IRegressionService using numpy/scipy for calculations.
"""
from typing import List
import numpy as np
from scipy import stats as scipy_stats

from ...core.domain.interfaces import IRegressionService
from ...core.domain.entities import RegressionModel
from ...core.domain.value_objects import RegressionParameters, RegressionMetrics


class RegressionServiceImpl(IRegressionService):
    """Concrete implementation of IRegressionService using numpy/scipy."""
    
    def train_simple(self, x: List[float], y: List[float]) -> RegressionModel:
        """Train simple linear regression: y = b0 + b1*x."""
        x_arr = np.array(x)
        y_arr = np.array(y)
        n = len(x_arr)
        
        # OLS Calculation
        x_mean = np.mean(x_arr)
        y_mean = np.mean(y_arr)
        
        ss_xy = np.sum((x_arr - x_mean) * (y_arr - y_mean))
        ss_xx = np.sum((x_arr - x_mean) ** 2)
        
        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        
        # Predictions and residuals
        y_pred = intercept + slope * x_arr
        residuals = y_arr - y_pred
        
        # Sum of squares
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_arr - y_mean) ** 2)
        ss_reg = ss_tot - ss_res
        
        # Metrics
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - 2) if n > 2 else r_squared
        mse = ss_res / (n - 2) if n > 2 else ss_res
        rmse = np.sqrt(mse)
        
        # F-statistic
        msr = ss_reg / 1
        f_stat = msr / mse if mse > 0 else 0
        p_value = 1 - scipy_stats.f.cdf(f_stat, 1, n - 2) if n > 2 else 1.0
        
        # Build model
        model = RegressionModel()
        model.parameters = RegressionParameters(
            intercept=float(intercept),
            coefficients={"x": float(slope)}
        )
        model.metrics = RegressionMetrics(
            r_squared=float(r_squared),
            r_squared_adj=float(r_squared_adj),
            mse=float(mse),
            rmse=float(rmse),
            f_statistic=float(f_stat),
            p_value=float(p_value)
        )
        model.predictions = y_pred.tolist()
        model.residuals = residuals.tolist()
        
        return model
    
    def train_multiple(self, x: List[List[float]], y: List[float], variable_names: List[str]) -> RegressionModel:
        """Train multiple regression: y = b0 + b1*x1 + b2*x2 + ..."""
        # x is a list of columns: [[x1_1, x1_2, ...], [x2_1, x2_2, ...]]
        X_cols = [np.array(col) for col in x]
        y_arr = np.array(y)
        n = len(y_arr)
        k = len(X_cols)
        
        # Build design matrix with intercept
        X = np.column_stack([np.ones(n)] + X_cols)
        
        # OLS: β = (X'X)^-1 X'y
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            beta = XtX_inv @ X.T @ y_arr
        except np.linalg.LinAlgError:
            # Fallback if singular
            beta = np.linalg.lstsq(X, y_arr, rcond=None)[0]
        
        intercept = beta[0]
        slopes = beta[1:]
        
        # Predictions and residuals
        y_pred = X @ beta
        residuals = y_arr - y_pred
        
        # Sum of squares
        y_mean = np.mean(y_arr)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_arr - y_mean) ** 2)
        ss_reg = ss_tot - ss_res
        
        # Metrics
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        df_res = n - k - 1
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / df_res if df_res > 0 else r_squared
        mse = ss_res / df_res if df_res > 0 else ss_res
        rmse = np.sqrt(mse)
        
        # F-statistic
        msr = ss_reg / k if k > 0 else 0
        f_stat = msr / mse if mse > 0 else 0
        p_value = 1 - scipy_stats.f.cdf(f_stat, k, df_res) if df_res > 0 else 1.0
        
        # Build coefficients dict
        coeffs_dict = {}
        for i, name in enumerate(variable_names):
            coeffs_dict[name] = float(slopes[i]) if i < len(slopes) else 0.0
        
        # Build model
        model = RegressionModel()
        model.parameters = RegressionParameters(
            intercept=float(intercept),
            coefficients=coeffs_dict
        )
        model.metrics = RegressionMetrics(
            r_squared=float(r_squared),
            r_squared_adj=float(r_squared_adj),
            mse=float(mse),
            rmse=float(rmse),
            f_statistic=float(f_stat),
            p_value=float(p_value)
        )
        model.predictions = y_pred.tolist()
        model.residuals = residuals.tolist()
        
        return model
