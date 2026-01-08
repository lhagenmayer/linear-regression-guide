"""
Validierungstests: Vergleich unserer Regression-Implementierungen mit scikit-learn.

Diese Tests stellen sicher, dass unsere eigenen Berechnungen mit der
etablierten scikit-learn-Bibliothek übereinstimmen.
"""

import pytest
import numpy as np

# Optional import für scikit-learn
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Skip all tests in this module if sklearn is not available
    pytestmark = pytest.mark.skip(reason="scikit-learn nicht installiert. Installiere mit: pip install scikit-learn")

from src.infrastructure.services.regression import RegressionServiceImpl
from src.infrastructure.data.generators import DataFetcher


class TestSimpleRegressionValidation:
    """Validierungstests für einfache lineare Regression."""
    
    @pytest.fixture
    def service(self):
        """Regression Service für Tests."""
        return RegressionServiceImpl()
    
    @pytest.fixture
    def sklearn_model(self):
        """Scikit-learn Modell für Vergleich."""
        return LinearRegression()
    
    def test_simple_regression_synthetic_linear(self, service, sklearn_model):
        """Test mit perfekt linearer Beziehung (y = 2x + 1)."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 10, n)
        y = 2 * x + 1 + np.random.normal(0, 0.1, n)
        
        # Unsere Implementierung
        our_model = service.train_simple(x.tolist(), y.tolist())
        
        # Scikit-learn
        sklearn_model.fit(x.reshape(-1, 1), y)
        sklearn_r2 = r2_score(y, sklearn_model.predict(x.reshape(-1, 1)))
        sklearn_mse_biased = mean_squared_error(y, sklearn_model.predict(x.reshape(-1, 1)))
        # Unsere MSE ist unbiased (dividiert durch n-2), sklearn ist biased (dividiert durch n)
        # Konvertiere sklearn MSE zu unbiased: mse_unbiased = mse_biased * n / (n - 2)
        sklearn_mse_unbiased = sklearn_mse_biased * n / (n - 2) if n > 2 else sklearn_mse_biased
        
        # Vergleich der Koeffizienten
        assert abs(our_model.parameters.intercept - sklearn_model.intercept_) < 1e-6
        assert abs(our_model.parameters.coefficients["x"] - sklearn_model.coef_[0]) < 1e-6
        
        # Vergleich der Metriken
        assert abs(our_model.metrics.r_squared - sklearn_r2) < 1e-6
        # Vergleich mit unbiased MSE
        assert abs(our_model.metrics.mse - sklearn_mse_unbiased) < 1e-5
        assert abs(our_model.metrics.rmse - np.sqrt(sklearn_mse_unbiased)) < 1e-5
    
    def test_simple_regression_with_noise(self, service, sklearn_model):
        """Test mit starker Rauschen."""
        np.random.seed(42)
        n = 200
        x = np.random.uniform(0, 20, n)
        y = 3.5 * x - 2.0 + np.random.normal(0, 5, n)  # Stärkeres Rauschen
        
        # Unsere Implementierung
        our_model = service.train_simple(x.tolist(), y.tolist())
        
        # Scikit-learn
        sklearn_model.fit(x.reshape(-1, 1), y)
        sklearn_r2 = r2_score(y, sklearn_model.predict(x.reshape(-1, 1)))
        sklearn_mse_biased = mean_squared_error(y, sklearn_model.predict(x.reshape(-1, 1)))
        # Konvertiere zu unbiased MSE
        sklearn_mse_unbiased = sklearn_mse_biased * n / (n - 2) if n > 2 else sklearn_mse_biased
        
        # Vergleich (mit etwas größerer Toleranz wegen Rauschen)
        assert abs(our_model.parameters.intercept - sklearn_model.intercept_) < 1e-5
        assert abs(our_model.parameters.coefficients["x"] - sklearn_model.coef_[0]) < 1e-5
        assert abs(our_model.metrics.r_squared - sklearn_r2) < 1e-5
        assert abs(our_model.metrics.mse - sklearn_mse_unbiased) < 1e-2
    
    def test_simple_regression_electronics_dataset(self, service, sklearn_model):
        """Test mit Elektronikmarkt-Datensatz."""
        fetcher = DataFetcher()
        data = fetcher.get_simple("electronics", n=100, noise=0.3, seed=42)
        
        # Unsere Implementierung
        our_model = service.train_simple(data.x.tolist(), data.y.tolist())
        
        # Scikit-learn
        sklearn_model.fit(data.x.reshape(-1, 1), data.y)
        sklearn_r2 = r2_score(data.y, sklearn_model.predict(data.x.reshape(-1, 1)))
        sklearn_mse_biased = mean_squared_error(data.y, sklearn_model.predict(data.x.reshape(-1, 1)))
        n = len(data.y)
        sklearn_mse_unbiased = sklearn_mse_biased * n / (n - 2) if n > 2 else sklearn_mse_biased
        
        # Vergleich
        assert abs(our_model.parameters.intercept - sklearn_model.intercept_) < 1e-5
        assert abs(our_model.parameters.coefficients["x"] - sklearn_model.coef_[0]) < 1e-5
        assert abs(our_model.metrics.r_squared - sklearn_r2) < 1e-5
        assert abs(our_model.metrics.mse - sklearn_mse_unbiased) < 1e-2
    
    def test_simple_regression_advertising_dataset(self, service, sklearn_model):
        """Test mit Werbestudie-Datensatz."""
        fetcher = DataFetcher()
        data = fetcher.get_simple("advertising", n=150, noise=0.2, seed=42)
        
        # Unsere Implementierung
        our_model = service.train_simple(data.x.tolist(), data.y.tolist())
        
        # Scikit-learn
        sklearn_model.fit(data.x.reshape(-1, 1), data.y)
        sklearn_r2 = r2_score(data.y, sklearn_model.predict(data.x.reshape(-1, 1)))
        
        # Vergleich
        assert abs(our_model.parameters.intercept - sklearn_model.intercept_) < 1e-4
        assert abs(our_model.parameters.coefficients["x"] - sklearn_model.coef_[0]) < 1e-4
        assert abs(our_model.metrics.r_squared - sklearn_r2) < 1e-4
    
    def test_simple_regression_temperature_dataset(self, service, sklearn_model):
        """Test mit Temperatur-Datensatz."""
        fetcher = DataFetcher()
        data = fetcher.get_simple("temperature", n=80, noise=0.15, seed=42)
        
        # Unsere Implementierung
        our_model = service.train_simple(data.x.tolist(), data.y.tolist())
        
        # Scikit-learn
        sklearn_model.fit(data.x.reshape(-1, 1), data.y)
        sklearn_r2 = r2_score(data.y, sklearn_model.predict(data.x.reshape(-1, 1)))
        
        # Vergleich
        assert abs(our_model.parameters.intercept - sklearn_model.intercept_) < 1e-4
        assert abs(our_model.parameters.coefficients["x"] - sklearn_model.coef_[0]) < 1e-4
        assert abs(our_model.metrics.r_squared - sklearn_r2) < 1e-4
    
    def test_simple_regression_predictions_match(self, service, sklearn_model):
        """Test dass Vorhersagen übereinstimmen."""
        np.random.seed(42)
        n = 50
        x = np.random.uniform(0, 10, n)
        y = 1.5 * x + 0.8 + np.random.normal(0, 0.5, n)
        
        # Unsere Implementierung
        our_model = service.train_simple(x.tolist(), y.tolist())
        
        # Scikit-learn
        sklearn_model.fit(x.reshape(-1, 1), y)
        sklearn_pred = sklearn_model.predict(x.reshape(-1, 1))
        
        # Vergleich der Vorhersagen
        our_pred = np.array(our_model.predictions)
        np.testing.assert_allclose(our_pred, sklearn_pred, rtol=1e-6, atol=1e-6)
        
        # Vergleich der Residuen
        our_residuals = np.array(our_model.residuals)
        sklearn_residuals = y - sklearn_pred
        np.testing.assert_allclose(our_residuals, sklearn_residuals, rtol=1e-6, atol=1e-6)


class TestMultipleRegressionValidation:
    """Validierungstests für multiple lineare Regression."""
    
    @pytest.fixture
    def service(self):
        """Regression Service für Tests."""
        return RegressionServiceImpl()
    
    @pytest.fixture
    def sklearn_model(self):
        """Scikit-learn Modell für Vergleich."""
        return LinearRegression()
    
    def test_multiple_regression_synthetic(self, service, sklearn_model):
        """Test mit synthetischen Daten (y = 1 + 2*x1 - 1.5*x2)."""
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(0, 10, n)
        x2 = np.random.uniform(0, 5, n)
        y = 1 + 2 * x1 - 1.5 * x2 + np.random.normal(0, 0.5, n)
        
        # Unsere Implementierung
        our_model = service.train_multiple(
            [x1.tolist(), x2.tolist()], 
            y.tolist(), 
            ["X1", "X2"]
        )
        
        # Scikit-learn
        X = np.column_stack([x1, x2])
        sklearn_model.fit(X, y)
        sklearn_r2 = r2_score(y, sklearn_model.predict(X))
        sklearn_mse_biased = mean_squared_error(y, sklearn_model.predict(X))
        # Konvertiere zu unbiased MSE: mse_unbiased = mse_biased * n / (n - k - 1)
        k = 2  # Anzahl Prädiktoren
        sklearn_mse_unbiased = sklearn_mse_biased * n / (n - k - 1) if (n - k - 1) > 0 else sklearn_mse_biased
        
        # Vergleich der Koeffizienten
        assert abs(our_model.parameters.intercept - sklearn_model.intercept_) < 1e-5
        assert abs(our_model.parameters.coefficients["X1"] - sklearn_model.coef_[0]) < 1e-5
        assert abs(our_model.parameters.coefficients["X2"] - sklearn_model.coef_[1]) < 1e-5
        
        # Vergleich der Metriken
        assert abs(our_model.metrics.r_squared - sklearn_r2) < 1e-5
        assert abs(our_model.metrics.mse - sklearn_mse_unbiased) < 1e-2
    
    def test_multiple_regression_cities_dataset(self, service, sklearn_model):
        """Test mit Städte-Datensatz."""
        fetcher = DataFetcher()
        data = fetcher.get_multiple("cities", n=100, noise=1.0, seed=42)
        
        # Unsere Implementierung
        our_model = service.train_multiple(
            [data.x1.tolist(), data.x2.tolist()],
            data.y.tolist(),
            ["X1", "X2"]
        )
        
        # Scikit-learn
        X = np.column_stack([data.x1, data.x2])
        sklearn_model.fit(X, data.y)
        sklearn_r2 = r2_score(data.y, sklearn_model.predict(X))
        sklearn_mse_biased = mean_squared_error(data.y, sklearn_model.predict(X))
        n = len(data.y)
        k = 2
        sklearn_mse_unbiased = sklearn_mse_biased * n / (n - k - 1) if (n - k - 1) > 0 else sklearn_mse_biased
        
        # Vergleich
        assert abs(our_model.parameters.intercept - sklearn_model.intercept_) < 1e-4
        assert abs(our_model.parameters.coefficients["X1"] - sklearn_model.coef_[0]) < 1e-4
        assert abs(our_model.parameters.coefficients["X2"] - sklearn_model.coef_[1]) < 1e-4
        assert abs(our_model.metrics.r_squared - sklearn_r2) < 1e-4
        assert abs(our_model.metrics.mse - sklearn_mse_unbiased) < 1e-1
    
    def test_multiple_regression_houses_dataset(self, service, sklearn_model):
        """Test mit Immobilien-Datensatz."""
        fetcher = DataFetcher()
        data = fetcher.get_multiple("houses", n=120, noise=2.0, seed=42)
        
        # Unsere Implementierung
        our_model = service.train_multiple(
            [data.x1.tolist(), data.x2.tolist()],
            data.y.tolist(),
            ["X1", "X2"]
        )
        
        # Scikit-learn
        X = np.column_stack([data.x1, data.x2])
        sklearn_model.fit(X, data.y)
        sklearn_r2 = r2_score(data.y, sklearn_model.predict(X))
        
        # Vergleich
        assert abs(our_model.parameters.intercept - sklearn_model.intercept_) < 1e-3
        assert abs(our_model.parameters.coefficients["X1"] - sklearn_model.coef_[0]) < 1e-3
        assert abs(our_model.parameters.coefficients["X2"] - sklearn_model.coef_[1]) < 1e-3
        assert abs(our_model.metrics.r_squared - sklearn_r2) < 1e-3
    
    def test_multiple_regression_cantons_dataset(self, service, sklearn_model):
        """Test mit Schweizer Kantone-Datensatz."""
        fetcher = DataFetcher()
        data = fetcher.get_multiple("cantons", n=150, noise=1000, seed=42)
        
        # Unsere Implementierung
        our_model = service.train_multiple(
            [data.x1.tolist(), data.x2.tolist()],
            data.y.tolist(),
            ["X1", "X2"]
        )
        
        # Scikit-learn
        X = np.column_stack([data.x1, data.x2])
        sklearn_model.fit(X, data.y)
        sklearn_r2 = r2_score(data.y, sklearn_model.predict(X))
        
        # Vergleich
        assert abs(our_model.parameters.intercept - sklearn_model.intercept_) < 1e-2
        assert abs(our_model.parameters.coefficients["X1"] - sklearn_model.coef_[0]) < 1e-2
        assert abs(our_model.parameters.coefficients["X2"] - sklearn_model.coef_[1]) < 1e-2
        assert abs(our_model.metrics.r_squared - sklearn_r2) < 1e-3
    
    def test_multiple_regression_weather_dataset(self, service, sklearn_model):
        """Test mit Wetter-Datensatz."""
        fetcher = DataFetcher()
        data = fetcher.get_multiple("weather", n=80, noise=0.5, seed=42)
        
        # Unsere Implementierung
        our_model = service.train_multiple(
            [data.x1.tolist(), data.x2.tolist()],
            data.y.tolist(),
            ["X1", "X2"]
        )
        
        # Scikit-learn
        X = np.column_stack([data.x1, data.x2])
        sklearn_model.fit(X, data.y)
        sklearn_r2 = r2_score(data.y, sklearn_model.predict(X))
        
        # Vergleich
        assert abs(our_model.parameters.intercept - sklearn_model.intercept_) < 1e-4
        assert abs(our_model.parameters.coefficients["X1"] - sklearn_model.coef_[0]) < 1e-4
        assert abs(our_model.parameters.coefficients["X2"] - sklearn_model.coef_[1]) < 1e-4
        assert abs(our_model.metrics.r_squared - sklearn_r2) < 1e-4
    
    def test_multiple_regression_three_predictors(self, service, sklearn_model):
        """Test mit drei Prädiktoren."""
        np.random.seed(42)
        n = 150
        x1 = np.random.uniform(0, 10, n)
        x2 = np.random.uniform(0, 5, n)
        x3 = np.random.uniform(0, 8, n)
        y = 2 + 1.5 * x1 - 0.8 * x2 + 0.5 * x3 + np.random.normal(0, 0.3, n)
        
        # Unsere Implementierung
        our_model = service.train_multiple(
            [x1.tolist(), x2.tolist(), x3.tolist()],
            y.tolist(),
            ["X1", "X2", "X3"]
        )
        
        # Scikit-learn
        X = np.column_stack([x1, x2, x3])
        sklearn_model.fit(X, y)
        sklearn_r2 = r2_score(y, sklearn_model.predict(X))
        sklearn_mse_biased = mean_squared_error(y, sklearn_model.predict(X))
        k = 3  # Anzahl Prädiktoren
        sklearn_mse_unbiased = sklearn_mse_biased * n / (n - k - 1) if (n - k - 1) > 0 else sklearn_mse_biased
        
        # Vergleich
        assert abs(our_model.parameters.intercept - sklearn_model.intercept_) < 1e-5
        assert abs(our_model.parameters.coefficients["X1"] - sklearn_model.coef_[0]) < 1e-5
        assert abs(our_model.parameters.coefficients["X2"] - sklearn_model.coef_[1]) < 1e-5
        assert abs(our_model.parameters.coefficients["X3"] - sklearn_model.coef_[2]) < 1e-5
        assert abs(our_model.metrics.r_squared - sklearn_r2) < 1e-5
        assert abs(our_model.metrics.mse - sklearn_mse_unbiased) < 1e-2
    
    def test_multiple_regression_predictions_match(self, service, sklearn_model):
        """Test dass Vorhersagen übereinstimmen."""
        np.random.seed(42)
        n = 60
        x1 = np.random.uniform(0, 10, n)
        x2 = np.random.uniform(0, 5, n)
        y = 1 + 2 * x1 - 1.5 * x2 + np.random.normal(0, 0.2, n)
        
        # Unsere Implementierung
        our_model = service.train_multiple(
            [x1.tolist(), x2.tolist()],
            y.tolist(),
            ["X1", "X2"]
        )
        
        # Scikit-learn
        X = np.column_stack([x1, x2])
        sklearn_model.fit(X, y)
        sklearn_pred = sklearn_model.predict(X)
        
        # Vergleich der Vorhersagen
        our_pred = np.array(our_model.predictions)
        np.testing.assert_allclose(our_pred, sklearn_pred, rtol=1e-6, atol=1e-6)
        
        # Vergleich der Residuen
        our_residuals = np.array(our_model.residuals)
        sklearn_residuals = y - sklearn_pred
        np.testing.assert_allclose(our_residuals, sklearn_residuals, rtol=1e-6, atol=1e-6)
    
    def test_multiple_regression_adjusted_r2(self, service, sklearn_model):
        """Test für adjustiertes R² (manuelle Berechnung vs. unsere Implementierung)."""
        np.random.seed(42)
        n = 50
        k = 2  # Anzahl Prädiktoren
        x1 = np.random.uniform(0, 10, n)
        x2 = np.random.uniform(0, 5, n)
        y = 1 + 2 * x1 - 1.5 * x2 + np.random.normal(0, 0.5, n)
        
        # Unsere Implementierung
        our_model = service.train_multiple(
            [x1.tolist(), x2.tolist()],
            y.tolist(),
            ["X1", "X2"]
        )
        
        # Scikit-learn
        X = np.column_stack([x1, x2])
        sklearn_model.fit(X, y)
        sklearn_r2 = r2_score(y, sklearn_model.predict(X))
        
        # Manuelle Berechnung von R²_adj
        # R²_adj = 1 - (1 - R²) * (n - 1) / (n - k - 1)
        manual_r2_adj = 1 - (1 - sklearn_r2) * (n - 1) / (n - k - 1)
        
        # Vergleich
        assert abs(our_model.metrics.r_squared_adj - manual_r2_adj) < 1e-5


class TestStatisticalMetricsValidation:
    """Validierung statistischer Metriken."""
    
    @pytest.fixture
    def service(self):
        """Regression Service für Tests."""
        return RegressionServiceImpl()
    
    def test_f_statistic_calculation(self, service):
        """Test F-Statistik Berechnung."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 10, n)
        y = 2 * x + 1 + np.random.normal(0, 0.5, n)
        
        our_model = service.train_simple(x.tolist(), y.tolist())
        
        # Manuelle Berechnung der F-Statistik
        # F = MSR / MSE = (SSR / df_reg) / (SSE / df_res)
        y_pred = np.array(our_model.predictions)
        y_mean = np.mean(y)
        ssr = np.sum((y_pred - y_mean) ** 2)  # Regression Sum of Squares
        sse = np.sum((y - y_pred) ** 2)  # Error Sum of Squares
        df_reg = 1  # Für einfache Regression
        df_res = n - 2
        
        msr = ssr / df_reg
        mse = sse / df_res
        manual_f = msr / mse if mse > 0 else 0
        
        assert abs(our_model.metrics.f_statistic - manual_f) < 1e-5
    
    def test_r_squared_adj_formula(self, service):
        """Test dass adjustiertes R² korrekt berechnet wird."""
        np.random.seed(42)
        n = 50
        x1 = np.random.uniform(0, 10, n)
        x2 = np.random.uniform(0, 5, n)
        y = 1 + 2 * x1 - 1.5 * x2 + np.random.normal(0, 0.3, n)
        
        our_model = service.train_multiple(
            [x1.tolist(), x2.tolist()],
            y.tolist(),
            ["X1", "X2"]
        )
        
        # Manuelle Berechnung
        r2 = our_model.metrics.r_squared
        k = 2  # Anzahl Prädiktoren
        df_res = n - k - 1
        manual_r2_adj = 1 - (1 - r2) * (n - 1) / df_res
        
        assert abs(our_model.metrics.r_squared_adj - manual_r2_adj) < 1e-6
