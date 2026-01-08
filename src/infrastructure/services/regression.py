"""
Infrastructure: Implementierung des Regression Services.
Nutzt NumPy und SciPy für die mathematischen Berechnungen.
"""
from typing import List
import numpy as np
from scipy import stats as scipy_stats

from ...core.domain.interfaces import IRegressionService
from ...core.domain.entities import RegressionModel
from ...core.domain.value_objects import RegressionParameters, RegressionMetrics, DomainError
from ...config.logger import get_logger, log_service_error


class RegressionServiceImpl(IRegressionService):
    """Konkrete Implementierung von IRegressionService mittels NumPy/SciPy."""
    
    def train_simple(self, x: List[float], y: List[float]) -> RegressionModel:
        """
        Trainiert eine einfache lineare Regression: y = b0 + b1*x.
        Berechnung erfolgt nach der Methode der kleinsten Quadrate (OLS).
        """
        x_arr = np.array(x)
        y_arr = np.array(y)
        n = len(x_arr)
        
        # OLS Berechnung: Arithmetische Mittel
        x_mean = np.mean(x_arr)
        y_mean = np.mean(y_arr)
        
        # Summen der Abweichungsquadrate (SS_xy und SS_xx)
        ss_xy = np.sum((x_arr - x_mean) * (y_arr - y_mean))
        ss_xx = np.sum((x_arr - x_mean) ** 2)
        
        # Schätzung von Steigung (Slope) und Achsenabschnitt (Intercept)
        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        
        # Vorhersagen und Residuen (Fehler)
        y_pred = intercept + slope * x_arr
        residuals = y_arr - y_pred
        
        # Quadratsummen zur Berechnung der Metriken
        ss_res = np.sum(residuals ** 2) # Sum of Squares Residuals
        ss_tot = np.sum((y_arr - y_mean) ** 2) # Total Sum of Squares
        ss_reg = ss_tot - ss_res # Explained Sum of Squares
        
        # Qualitätsmetriken
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        # Adjustiertes R² (berücksichtigt Freiheitsgrade)
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - 2) if n > 2 else r_squared
        mse = ss_res / (n - 2) if n > 2 else ss_res # Mean Squared Error
        rmse = np.sqrt(mse) # Root Mean Squared Error
        
        # F-Statistik und p-Wert für die Signifikanz des Gesamtmodells
        msr = ss_reg / 1
        f_stat = msr / mse if mse > 0 else 0
        p_value = 1 - scipy_stats.f.cdf(f_stat, 1, n - 2) if n > 2 else 1.0
        
        # Erzeugung der Domain Entity
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
        """
        Trainiert eine multiple Regression: y = b0 + b1*x1 + b2*x2 + ...
        Nutzt Matrix-Algebra für die Schätzung.
        """
        # x ist eine Liste von Spaltenvektoren
        X_cols = [np.array(col) for col in x]
        y_arr = np.array(y)
        n = len(y_arr)
        k = len(X_cols) # Anzahl der Prädiktoren (Features)
        
        # Konstruktion der Design-Matrix (X) mit einer Einserspalte für den Intercept
        X = np.column_stack([np.ones(n)] + X_cols)
        
        # Schätzung der Koeffizienten mittels Matrixformel: β = (X'X)⁻¹ X'y
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError as e:
            error = DomainError(
                "Die Datenmatrix ist singulär. Mögliche Ursachen: Perfekte Multikollinearität (Variablen hängen exakt voneinander ab) oder zu wenig Beobachtungen.",
                code="SINGULAR_MATRIX"
            )
            log_service_error(
                logger=self.logger,
                error=error,
                service_name="RegressionService",
                operation="train_multiple",
                input_params={
                    "n": n,
                    "k": len(variable_names),
                    "variable_names": variable_names
                },
                matrix_shape=XtX.shape,
                original_error=str(e)
            )
            raise error
            
        beta = XtX_inv @ X.T @ y_arr
        
        intercept = beta[0]
        slopes = beta[1:]
        
        # Vorhersagen und Residuen
        y_pred = X @ beta
        residuals = y_arr - y_pred
        
        # Quadratsummenberechnung
        y_mean = np.mean(y_arr)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_arr - y_mean) ** 2)
        ss_reg = ss_tot - ss_res
        
        # Metriken
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        df_res = n - k - 1 # Freiheitsgrade der Residuen
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / df_res if df_res > 0 else r_squared
        mse = ss_res / df_res if df_res > 0 else ss_res
        rmse = np.sqrt(mse)
        
        # F-Statistik für multiple Prädiktoren
        msr = ss_reg / k if k > 0 else 0
        f_stat = msr / mse if mse > 0 else 0
        p_value = 1 - scipy_stats.f.cdf(f_stat, k, df_res) if df_res > 0 else 1.0
        
        # Mapping der Koeffizienten auf Variablennamen
        coeffs_dict = {}
        for i, name in enumerate(variable_names):
            coeffs_dict[name] = float(slopes[i]) if i < len(slopes) else 0.0
        
        # Modell-Zustand speichern
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
