"""
Schritt 2: BERECHNEN (CALCULATE)

Dieses Modul verwaltet alle statistischen Berechnungen.
Es berechnet Regressionsmodelle und Statistiken aus den bereitgestellten Daten.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

from ...config import get_logger
from ..data.generators import DataResult, MultipleRegressionDataResult

logger = get_logger(__name__)


@dataclass
class RegressionResult:
    """Ergebnis einer einfachen Regressionsberechnung."""
    # Koeffizienten
    intercept: float
    slope: float
    
    # Vorhersagen
    y_pred: np.ndarray
    residuals: np.ndarray
    
    # Modell-Güte (Model Fit)
    r_squared: float
    r_squared_adj: float
    
    # Standardfehler
    se_intercept: float
    se_slope: float
    
    # Teststatistiken
    t_intercept: float
    t_slope: float
    p_intercept: float
    p_slope: float
    
    # Quadratsummen (Sum of Squares)
    sse: float  # Sum of Squared Errors (Residualsumme)
    sst: float  # Total Sum of Squares (Gesamtsumme)
    ssr: float  # Regression Sum of Squares (Erklärte Summe)
    mse: float  # Mean Squared Error (Mittlerer quadratischer Fehler)
    
    # Stichproben-Informationen
    n: int
    df: int  # Freiheitsgrade (Degrees of freedom)
    
    # Zusätzliche Statistiken
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultipleRegressionResult:
    """Ergebnis einer multiplen Regressionsberechnung."""
    # Koeffizienten
    intercept: float
    coefficients: List[float]
    
    # Vorhersagen
    y_pred: np.ndarray
    residuals: np.ndarray
    
    # Modell-Güte
    r_squared: float
    r_squared_adj: float
    f_statistic: float
    f_pvalue: float
    
    # Standardfehler und Tests für jeden Koeffizienten
    se_coefficients: List[float]
    t_values: List[float]
    p_values: List[float]
    
    # Quadratsummen
    sse: float
    sst: float
    ssr: float
    
    # Stichproben-Informationen
    n: int
    k: int  # Anzahl der Prädiktoren
    
    extra: Dict[str, Any] = field(default_factory=dict)


class StatisticsCalculator:
    """
    Klasse für statistische Berechnungen (Schritt 2: CALCULATE).
    
    Berechnet Regressionsmodelle und zugehörige Statistiken.
    Nutzt transparente, nachvollziehbare Formeln (keine Black-Box-Bibliotheken).
    
    Beispiel:
        calc = StatisticsCalculator()
        result = calc.simple_regression(data.x, data.y)
    """
    
    def __init__(self):
        logger.info("StatisticsCalculator initialisiert")
    
    def simple_regression(
        self, 
        x: np.ndarray, 
        y: np.ndarray
    ) -> RegressionResult:
        """
        Berechnet eine einfache lineare Regression: ŷ = b₀ + b₁x
        
        Alle Formeln sind für didaktische Transparenz explizit implementiert:
        - b₁ = Cov(x,y) / Var(x) = Σ(xᵢ-x̄)(yᵢ-ȳ) / Σ(xᵢ-x̄)²
        - b₀ = ȳ - b₁x̄
        - R² = 1 - SSE/SST
        
        Args:
            x: Unabhängige Variable (Prädiktor)
            y: Abhängige Variable (Response)
        
        Returns:
            RegressionResult mit allen statistischen Kennzahlen
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)
        
        logger.info(f"Berechne einfache Regression: n={n}")
        
        # =========================================================
        # SCHRITT 1: Basale Statistiken (Mittelwerte)
        # =========================================================
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Abweichungen vom Mittelwert
        x_dev = x - x_mean
        y_dev = y - y_mean
        
        # =========================================================
        # SCHRITT 2: OLS-Koeffizienten (Ordinary Least Squares)
        # b₁ = Σ(xᵢ-x̄)(yᵢ-ȳ) / Σ(xᵢ-x̄)²
        # b₀ = ȳ - b₁x̄
        # =========================================================
        ss_xx = np.sum(x_dev ** 2)  # Σ(xᵢ-x̄)²
        ss_xy = np.sum(x_dev * y_dev)  # Σ(xᵢ-x̄)(yᵢ-ȳ)
        
        b1 = ss_xy / ss_xx if ss_xx > 0 else 0  # Steigung (Slope)
        b0 = y_mean - b1 * x_mean  # Achsenabschnitt (Intercept)
        
        # =========================================================
        # SCHRITT 3: Vorhersagen & Residuen
        # =========================================================
        y_pred = b0 + b1 * x
        residuals = y - y_pred
        
        # =========================================================
        # SCHRITT 4: Quadratsummen (Sum of Squares)
        # =========================================================
        sse = np.sum(residuals ** 2)  # SSE = Σ(yᵢ - ŷᵢ)² (Fehlersumme)
        sst = np.sum(y_dev ** 2)  # SST = Σ(yᵢ - ȳ)² (Gesamtsumme)
        ssr = sst - sse  # SSR = SST - SSE (Erklärte Summe)
        
        # =========================================================
        # SCHRITT 5: R² und adjustiertes R²
        # =========================================================
        r_squared = 1 - (sse / sst) if sst > 0 else 0
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - 2) if n > 2 else 0
        
        # =========================================================
        # SCHRITT 6: Standardfehler
        # =========================================================
        df = n - 2  # Freiheitsgrade
        mse = sse / df if df > 0 else 0
        se_regression = np.sqrt(mse)
        
        # SE(b₁) = s / √(Σ(xᵢ-x̄)²)
        se_b1 = se_regression / np.sqrt(ss_xx) if ss_xx > 0 else 0
        
        # SE(b₀) = s * √(1/n + x̄²/Σ(xᵢ-x̄)²)
        se_b0 = se_regression * np.sqrt(1/n + x_mean**2 / ss_xx) if ss_xx > 0 else 0
        
        # =========================================================
        # SCHRITT 7: t-Statistiken und p-Werte
        # =========================================================
        t_b0 = b0 / se_b0 if se_b0 > 0 else 0
        t_b1 = b1 / se_b1 if se_b1 > 0 else 0
        
        # p-Werte unter Nutzung der t-Verteilung
        from scipy import stats
        p_b0 = 2 * (1 - stats.t.cdf(abs(t_b0), df)) if df > 0 else 1
        p_b1 = 2 * (1 - stats.t.cdf(abs(t_b1), df)) if df > 0 else 1
        
        logger.info(f"Regression abgeschlossen: R²={r_squared:.4f}, b₀={b0:.4f}, b₁={b1:.4f}")
        
        return RegressionResult(
            intercept=b0,
            slope=b1,
            y_pred=y_pred,
            residuals=residuals,
            r_squared=r_squared,
            r_squared_adj=r_squared_adj,
            se_intercept=se_b0,
            se_slope=se_b1,
            t_intercept=t_b0,
            t_slope=t_b1,
            p_intercept=p_b0,
            p_slope=p_b1,
            sse=sse,
            sst=sst,
            ssr=ssr,
            mse=mse,
            n=n,
            df=df,
            extra={
                "x_mean": x_mean,
                "y_mean": y_mean,
                "se_regression": se_regression,
                "correlation": np.sqrt(r_squared) * (1 if b1 >= 0 else -1),
            }
        )
    
    def multiple_regression(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        y: np.ndarray
    ) -> MultipleRegressionResult:
        """
        Berechnet eine multiple Regression: ŷ = b₀ + b₁x₁ + b₂x₂
        
        Nutzt die Matrix-Formel: b = (X'X)⁻¹X'y
        
        Args:
            x1: Erster Prädiktor
            x2: Zweiter Prädiktor
            y: Zielvariable (Response)
        
        Returns:
            MultipleRegressionResult mit allen Kennzahlen
        """
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        k = 2  # Anzahl der Prädiktoren
        
        logger.info(f"Berechne multiple Regression: n={n}, k={k}")
        
        # =========================================================
        # SCHRITT 1: Design-Matrix aufbauen
        # X = [1, x₁, x₂]
        # =========================================================
        X = np.column_stack([np.ones(n), x1, x2])
        
        # =========================================================
        # SCHRITT 2: OLS via Matrix-Algebra
        # b = (X'X)⁻¹X'y
        # =========================================================
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        Xty = X.T @ y
        b = XtX_inv @ Xty
        
        b0, b1, b2 = b[0], b[1], b[2]
        
        # =========================================================
        # SCHRITT 3: Vorhersagen & Residuen
        # =========================================================
        y_pred = X @ b
        residuals = y - y_pred
        y_mean = np.mean(y)
        
        # =========================================================
        # SCHRITT 4: Quadratsummen
        # =========================================================
        sse = np.sum(residuals ** 2)
        sst = np.sum((y - y_mean) ** 2)
        ssr = sst - sse
        
        # =========================================================
        # SCHRITT 5: R² und adjustiertes R²
        # =========================================================
        r_squared = 1 - (sse / sst) if sst > 0 else 0
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k - 1) if n > k + 1 else 0
        
        # =========================================================
        # SCHRITT 6: Standardfehler der Koeffizienten
        # Var(b) = σ²(X'X)⁻¹
        # =========================================================
        df = n - k - 1
        mse = sse / df if df > 0 else 0
        var_b = mse * XtX_inv
        se_b = np.sqrt(np.diag(var_b))
        
        # =========================================================
        # SCHRITT 7: t-Statistiken und p-Werte
        # =========================================================
        t_values = b / se_b
        
        from scipy import stats
        p_values = [2 * (1 - stats.t.cdf(abs(t), df)) if df > 0 else 1 for t in t_values]
        
        # =========================================================
        # SCHRITT 8: F-Statistik (Gesamtmodell-Signifikanz)
        # F = (SSR/k) / (SSE/(n-k-1))
        # =========================================================
        msr = ssr / k if k > 0 else 0
        f_stat = msr / mse if mse > 0 else 0
        f_pvalue = 1 - stats.f.cdf(f_stat, k, df) if df > 0 else 1
        
        logger.info(f"Multiple Regression abgeschlossen: R²={r_squared:.4f}")
        
        return MultipleRegressionResult(
            intercept=b0,
            coefficients=[b1, b2],
            y_pred=y_pred,
            residuals=residuals,
            r_squared=r_squared,
            r_squared_adj=r_squared_adj,
            f_statistic=f_stat,
            f_pvalue=f_pvalue,
            se_coefficients=list(se_b),
            t_values=list(t_values),
            p_values=p_values,
            sse=sse,
            sst=sst,
            ssr=ssr,
            n=n,
            k=k,
            extra={"y_mean": y_mean, "mse": mse}
        )
    
    def basic_stats(self, data: np.ndarray) -> Dict[str, float]:
        """
        Berechnet grundlegende deskriptive Statistiken.
        
        Args:
            data: Numerisches Array
        
        Returns:
            Dictionary mit Mittelwert, StdAbw, Min, Max etc.
        """
        data = np.asarray(data, dtype=float)
        return {
            "n": len(data),
            "mean": float(np.mean(data)),
            "std": float(np.std(data, ddof=1)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
            "q25": float(np.percentile(data, 25)),
            "q75": float(np.percentile(data, 75)),
        }
