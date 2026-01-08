"""
Validierungstests: Vergleich unserer R-Output-Formatierung mit echten R-Berechnungen.

Diese Tests stellen sicher, dass unsere R-Style-Output-Formatierung mit
tatsächlichen R-Berechnungen übereinstimmt.
"""

import pytest
import numpy as np
import re

# Optional import für rpy2 (R-Interface für Python)
# Wir skippen nur, wenn rpy2 selbst fehlt. Ob R korrekt konfiguriert ist,
# soll durch die Tests sichtbar werden (Fehler statt globalem Skip).
try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    R_AVAILABLE = True
except ImportError as e:
    R_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason=f"rpy2 nicht verfügbar: {e}")

from src.infrastructure.services.regression import RegressionServiceImpl
from src.infrastructure.services.calculate import StatisticsCalculator
from src.infrastructure.ai.formatters import ROutputFormatter
from src.infrastructure.data.generators import DataFetcher


@pytest.fixture(scope="module")
def r_base():
    """R base package."""
    return importr('base')


@pytest.fixture(scope="module")
def r_stats():
    """R stats package."""
    return importr('stats')


class TestSimpleRegressionROutput:
    """Validierung der R-Output-Formatierung für einfache Regression."""
    
    @pytest.fixture
    def service(self):
        """Regression Service."""
        return RegressionServiceImpl()
    
    @pytest.fixture
    def calculator(self):
        """Statistics Calculator."""
        return StatisticsCalculator()
    
    def _run_r_regression(self, x, y, r_stats):
        """Führt Regression in R durch und gibt Summary zurück."""
        # Daten nach R übertragen
        r_x = ro.FloatVector(x)
        r_y = ro.FloatVector(y)
        
        # Regression durchführen
        formula = ro.Formula('y ~ x')
        ro.globalenv['x'] = r_x
        ro.globalenv['y'] = r_y
        
        model = r_stats.lm(formula)
        # summary ist eine generische R-Funktion
        summary_model = ro.r('summary')(model)
        
        # Koeffizienten extrahieren (rpy2 verwendet 1-basierte Indizes mit rx)
        coef = summary_model.rx2('coefficients')
        intercept = float(coef.rx(1, 1)[0])  # Zeile 1, Spalte 1 (Intercept, Estimate)
        slope = float(coef.rx(2, 1)[0])      # Zeile 2, Spalte 1 (Slope, Estimate)
        se_intercept = float(coef.rx(1, 2)[0])  # Zeile 1, Spalte 2 (Intercept, Std. Error)
        se_slope = float(coef.rx(2, 2)[0])     # Zeile 2, Spalte 2 (Slope, Std. Error)
        t_intercept = float(coef.rx(1, 3)[0])   # Zeile 1, Spalte 3 (Intercept, t value)
        t_slope = float(coef.rx(2, 3)[0])       # Zeile 2, Spalte 3 (Slope, t value)
        p_intercept = float(coef.rx(1, 4)[0])   # Zeile 1, Spalte 4 (Intercept, Pr(>|t|))
        p_slope = float(coef.rx(2, 4)[0])       # Zeile 2, Spalte 4 (Slope, Pr(>|t|))
        
        # Weitere Statistiken
        r_squared = float(summary_model.rx2('r.squared')[0])
        r_squared_adj = float(summary_model.rx2('adj.r.squared')[0])
        f_statistic = float(summary_model.rx2('fstatistic')[0])
        df = int(summary_model.rx2('df')[1])  # Residual degrees of freedom
        
        # Residuals (mit Converter für numpy-Array)
        with localconverter(ro.default_converter + numpy2ri.converter):
            residuals = np.array(model.rx2('residuals'))
        
        # MSE berechnen (R verwendet n-2 für einfache Regression)
        mse = float(np.var(residuals) * (len(residuals) - 1) / (len(residuals) - 2))
        
        return {
            'intercept': float(intercept),
            'slope': float(slope),
            'se_intercept': float(se_intercept),
            'se_slope': float(se_slope),
            't_intercept': float(t_intercept),
            't_slope': float(t_slope),
            'p_intercept': float(p_intercept),
            'p_slope': float(p_slope),
            'r_squared': r_squared,
            'r_squared_adj': r_squared_adj,
            'f_statistic': f_statistic,
            'df': df,
            'mse': mse,
            'residuals': residuals
        }
    
    def test_simple_regression_r_output_synthetic(self, service, calculator, r_stats):
        """Vergleich unserer Formatierung mit R für synthetische Daten."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 10, n)
        y = 2 * x + 1 + np.random.normal(0, 0.5, n)
        
        # Unsere Berechnung
        our_model = service.train_simple(x.tolist(), y.tolist())
        our_result = calculator.simple_regression(x, y)
        
        # R-Berechnung
        r_result = self._run_r_regression(x.tolist(), y.tolist(), r_stats)
        
        # Vergleich der Koeffizienten
        assert abs(our_model.parameters.intercept - r_result['intercept']) < 1e-5
        assert abs(our_model.parameters.coefficients['x'] - r_result['slope']) < 1e-5
        
        # Vergleich der Metriken
        assert abs(our_model.metrics.r_squared - r_result['r_squared']) < 1e-5
        assert abs(our_model.metrics.r_squared_adj - r_result['r_squared_adj']) < 1e-5
        
        # Vergleich der Standardfehler (mit etwas größerer Toleranz)
        assert abs(our_result.se_intercept - r_result['se_intercept']) < 1e-3
        assert abs(our_result.se_slope - r_result['se_slope']) < 1e-3
        
        # Vergleich der t-Werte
        assert abs(our_result.t_intercept - r_result['t_intercept']) < 1e-2
        assert abs(our_result.t_slope - r_result['t_slope']) < 1e-2
        
        # Vergleich der p-Werte
        assert abs(our_result.p_intercept - r_result['p_intercept']) < 1e-3
        assert abs(our_result.p_slope - r_result['p_slope']) < 1e-3
    
    def test_simple_regression_r_output_format(self, service, calculator, r_stats):
        """Vergleich des formatierten R-Outputs."""
        np.random.seed(42)
        n = 50
        x = np.linspace(0, 10, n)
        y = 1.5 * x + 0.8 + np.random.normal(0, 0.3, n)
        
        # Unsere Berechnung
        our_result = calculator.simple_regression(x, y)
        
        # R-Berechnung
        r_result = self._run_r_regression(x.tolist(), y.tolist(), r_stats)
        
        # Stats-Dictionary für unsere Formatierung
        stats = {
            'x_label': 'X',
            'y_label': 'Y',
            'intercept': our_result.intercept,
            'slope': our_result.slope,
            'se_intercept': our_result.se_intercept,
            'se_slope': our_result.se_slope,
            't_intercept': our_result.t_intercept,
            't_slope': our_result.t_slope,
            'p_intercept': our_result.p_intercept,
            'p_slope': our_result.p_slope,
            'r_squared': our_result.r_squared,
            'r_squared_adj': our_result.r_squared_adj,
            'f_statistic': our_result.f_statistic if hasattr(our_result, 'f_statistic') else 0,
            'df': our_result.df,
            'mse': our_result.mse,
            'residuals': our_result.residuals
        }
        
        # Unsere Formatierung
        our_output = ROutputFormatter.format(stats)
        
        # Prüfe wichtige Elemente im Output
        assert "lm(formula = Y ~ X)" in our_output
        assert "(Intercept)" in our_output
        assert "Estimate Std. Error t value Pr(>|t|)" in our_output
        assert "Multiple R-squared:" in our_output
        assert "Adjusted R-squared:" in our_output
        assert "F-statistic:" in our_output
        
        # Extrahiere Werte aus unserem Output und vergleiche mit R
        # Intercept
        intercept_match = re.search(r'\(Intercept\)\s+([-\d.]+)', our_output)
        if intercept_match:
            our_intercept_str = intercept_match.group(1).strip()
            assert abs(float(our_intercept_str) - r_result['intercept']) < 1e-3
        
        # Slope
        slope_match = re.search(r'X\s+([-\d.]+)', our_output)
        if slope_match:
            our_slope_str = slope_match.group(1).strip()
            assert abs(float(our_slope_str) - r_result['slope']) < 1e-3
        
        # R-squared
        r2_match = re.search(r'Multiple R-squared:\s+([\d.]+)', our_output)
        if r2_match:
            our_r2 = float(r2_match.group(1))
            assert abs(our_r2 - r_result['r_squared']) < 1e-3
    
    def test_simple_regression_electronics_dataset(self, service, calculator, r_stats):
        """Vergleich mit Elektronikmarkt-Datensatz."""
        fetcher = DataFetcher()
        data = fetcher.get_simple("electronics", n=80, noise=0.3, seed=42)
        
        # Unsere Berechnung
        our_result = calculator.simple_regression(data.x, data.y)
        
        # R-Berechnung
        r_result = self._run_r_regression(data.x.tolist(), data.y.tolist(), r_stats)
        
        # Vergleich
        assert abs(our_result.intercept - r_result['intercept']) < 1e-3
        assert abs(our_result.slope - r_result['slope']) < 1e-3
        assert abs(our_result.r_squared - r_result['r_squared']) < 1e-3
        assert abs(our_result.r_squared_adj - r_result['r_squared_adj']) < 1e-3


class TestMultipleRegressionROutput:
    """Validierung der R-Output-Formatierung für multiple Regression."""
    
    @pytest.fixture
    def service(self):
        """Regression Service."""
        return RegressionServiceImpl()
    
    def _run_r_multiple_regression(self, x1, x2, y, r_stats):
        """Führt multiple Regression in R durch."""
        # Daten nach R übertragen
        r_x1 = ro.FloatVector(x1)
        r_x2 = ro.FloatVector(x2)
        r_y = ro.FloatVector(y)
        
        ro.globalenv['x1'] = r_x1
        ro.globalenv['x2'] = r_x2
        ro.globalenv['y'] = r_y
        
        # Regression durchführen
        formula = ro.Formula('y ~ x1 + x2')
        model = r_stats.lm(formula)
        # summary ist eine generische R-Funktion (ohne Converter-Context)
        summary_model = ro.r('summary')(model)
        
        # Koeffizienten extrahieren (rpy2 verwendet 1-basierte Indizes)
        coef = summary_model.rx2('coefficients')
        intercept = float(coef.rx(1, 1)[0])  # Zeile 1, Spalte 1
        b1 = float(coef.rx(2, 1)[0])         # Zeile 2, Spalte 1
        b2 = float(coef.rx(3, 1)[0])         # Zeile 3, Spalte 1
        se_intercept = float(coef.rx(1, 2)[0])
        se_b1 = float(coef.rx(2, 2)[0])
        se_b2 = float(coef.rx(3, 2)[0])
        t_intercept = float(coef.rx(1, 3)[0])
        t_b1 = float(coef.rx(2, 3)[0])
        t_b2 = float(coef.rx(3, 3)[0])
        p_intercept = float(coef.rx(1, 4)[0])
        p_b1 = float(coef.rx(2, 4)[0])
        p_b2 = float(coef.rx(3, 4)[0])
        
        # Weitere Statistiken
        r_squared = float(summary_model.rx2('r.squared')[0])
        r_squared_adj = float(summary_model.rx2('adj.r.squared')[0])
        f_statistic = float(summary_model.rx2('fstatistic')[0])
        df = int(summary_model.rx2('df')[1])
        
        # Residuals (mit Converter für numpy-Array)
        with localconverter(ro.default_converter + numpy2ri.converter):
            residuals = np.array(model.rx2('residuals'))
        
        # MSE (R verwendet n-k-1 für multiple Regression mit k Prädiktoren)
        k = 2
        mse = float(np.var(residuals) * (len(residuals) - 1) / (len(residuals) - k - 1))
        
        return {
            'intercept': float(intercept),
            'b1': float(b1),
            'b2': float(b2),
            'se_intercept': float(se_intercept),
            'se_b1': float(se_b1),
            'se_b2': float(se_b2),
            't_intercept': float(t_intercept),
            't_b1': float(t_b1),
            't_b2': float(t_b2),
            'p_intercept': float(p_intercept),
            'p_b1': float(p_b1),
            'p_b2': float(p_b2),
            'r_squared': r_squared,
            'r_squared_adj': r_squared_adj,
            'f_statistic': f_statistic,
            'df': df,
            'mse': mse,
            'residuals': residuals
        }
    
    def test_multiple_regression_r_output_synthetic(self, service, r_stats):
        """Vergleich unserer Formatierung mit R für multiple Regression."""
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(0, 10, n)
        x2 = np.random.uniform(0, 5, n)
        y = 1 + 2 * x1 - 1.5 * x2 + np.random.normal(0, 0.5, n)
        
        # Unsere Berechnung
        our_model = service.train_multiple(
            [x1.tolist(), x2.tolist()],
            y.tolist(),
            ["X1", "X2"]
        )
        
        # R-Berechnung
        r_result = self._run_r_multiple_regression(
            x1.tolist(), x2.tolist(), y.tolist(), r_stats
        )
        
        # Vergleich der Koeffizienten
        assert abs(our_model.parameters.intercept - r_result['intercept']) < 1e-4
        assert abs(our_model.parameters.coefficients['X1'] - r_result['b1']) < 1e-4
        assert abs(our_model.parameters.coefficients['X2'] - r_result['b2']) < 1e-4
        
        # Vergleich der Metriken
        assert abs(our_model.metrics.r_squared - r_result['r_squared']) < 1e-4
        assert abs(our_model.metrics.r_squared_adj - r_result['r_squared_adj']) < 1e-4
        assert abs(our_model.metrics.f_statistic - r_result['f_statistic']) < 1e-2
    
    def test_multiple_regression_r_output_format(self, service, r_stats):
        """Vergleich des formatierten R-Outputs für multiple Regression."""
        np.random.seed(42)
        n = 60
        x1 = np.random.uniform(0, 10, n)
        x2 = np.random.uniform(0, 5, n)
        y = 1 + 2 * x1 - 1.5 * x2 + np.random.normal(0, 0.3, n)
        
        # Unsere Berechnung
        our_model = service.train_multiple(
            [x1.tolist(), x2.tolist()],
            y.tolist(),
            ["X1", "X2"]
        )
        
        # R-Berechnung
        r_result = self._run_r_multiple_regression(
            x1.tolist(), x2.tolist(), y.tolist(), r_stats
        )
        
        # Stats-Dictionary für unsere Formatierung
        # Wir müssen die Standardfehler und t-Werte aus R verwenden, da unsere
        # RegressionServiceImpl diese nicht berechnet
        stats = {
            'y_label': 'Y',
            'intercept': our_model.parameters.intercept,
            'se_intercept': r_result['se_intercept'],
            't_intercept': r_result['t_intercept'],
            'p_intercept': r_result['p_intercept'],
            'coefficients': {
                'X1': our_model.parameters.coefficients['X1'],
                'X2': our_model.parameters.coefficients['X2']
            },
            'se_coefficients': [r_result['se_b1'], r_result['se_b2']],
            't_values': [r_result['t_b1'], r_result['t_b2']],
            'p_values': [r_result['p_b1'], r_result['p_b2']],
            'feature_names': ['X1', 'X2'],
            'r_squared': our_model.metrics.r_squared,
            'r_squared_adj': our_model.metrics.r_squared_adj,
            'f_statistic': our_model.metrics.f_statistic,
            'f_p_value': our_model.metrics.p_value,
            'df': n - 3,  # n - k - 1
            'mse': our_model.metrics.mse,
            'residuals': our_model.residuals
        }
        
        # Unsere Formatierung
        our_output = ROutputFormatter.format(stats)
        
        # Prüfe wichtige Elemente
        assert "lm(formula = Y ~ X1 + X2)" in our_output
        assert "(Intercept)" in our_output
        assert "X1" in our_output
        assert "X2" in our_output
        assert "Multiple R-squared:" in our_output
        
        # Extrahiere und vergleiche Werte
        # Intercept
        intercept_match = re.search(r'\(Intercept\)\s+([-\d.]+)', our_output)
        if intercept_match:
            our_intercept = float(intercept_match.group(1).strip())
            assert abs(our_intercept - r_result['intercept']) < 1e-3
        
        # R-squared
        r2_match = re.search(r'Multiple R-squared:\s+([\d.]+)', our_output)
        if r2_match:
            our_r2 = float(r2_match.group(1))
            assert abs(our_r2 - r_result['r_squared']) < 1e-3
    
    def test_multiple_regression_cities_dataset(self, service, r_stats):
        """Vergleich mit Städte-Datensatz."""
        fetcher = DataFetcher()
        data = fetcher.get_multiple("cities", n=100, noise=1.0, seed=42)
        
        # Unsere Berechnung
        our_model = service.train_multiple(
            [data.x1.tolist(), data.x2.tolist()],
            data.y.tolist(),
            ["X1", "X2"]
        )
        
        # R-Berechnung
        r_result = self._run_r_multiple_regression(
            data.x1.tolist(), data.x2.tolist(), data.y.tolist(), r_stats
        )
        
        # Vergleich
        assert abs(our_model.parameters.intercept - r_result['intercept']) < 1e-2
        assert abs(our_model.parameters.coefficients['X1'] - r_result['b1']) < 1e-2
        assert abs(our_model.parameters.coefficients['X2'] - r_result['b2']) < 1e-2
        assert abs(our_model.metrics.r_squared - r_result['r_squared']) < 1e-3


class TestRResidualsFormatting:
    """Validierung der Residuen-Formatierung im R-Output."""
    
    def test_residuals_quantiles_match_r(self, r_stats):
        """Test dass Residuen-Quantile mit R übereinstimmen."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 10, n)
        y = 2 * x + 1 + np.random.normal(0, 0.5, n)
        
        # R-Berechnung
        r_x = ro.FloatVector(x.tolist())
        r_y = ro.FloatVector(y.tolist())
        ro.globalenv['x'] = r_x
        ro.globalenv['y'] = r_y
        
        formula = ro.Formula('y ~ x')
        model = r_stats.lm(formula)
        # Residuals mit Converter für numpy-Array
        with localconverter(ro.default_converter + numpy2ri.converter):
            r_residuals = np.array(model.rx2('residuals'))
        
        # R-Quantile
        r_quantiles = np.percentile(r_residuals, [0, 25, 50, 75, 100])
        
        # Unsere Formatierung sollte ähnliche Quantile haben
        stats = {
            'x_label': 'X',
            'y_label': 'Y',
            'intercept': 1.0,
            'slope': 2.0,
            'residuals': r_residuals.tolist()
        }
        
        our_output = ROutputFormatter.format(stats)
        
        # Extrahiere Residuen-Quantile aus Output
        # Format: "     Min       1Q   Median       3Q      Max"
        # Dann nächste Zeile: " -0.1500  -0.1000   0.0500   0.1000   0.2000"
        lines = our_output.split('\n')
        our_min = our_q1 = our_med = our_q3 = our_max = None
        
        for i, line in enumerate(lines):
            if 'Min' in line and '1Q' in line and 'Median' in line:
                # Nächste Zeile sollte die Werte enthalten
                if i + 1 < len(lines):
                    values_line = lines[i + 1].strip()
                    # Parse numerische Werte (kann negative Zahlen enthalten)
                    import re
                    numeric_parts = re.findall(r'-?\d+\.?\d*', values_line)
                    if len(numeric_parts) >= 5:
                        our_min = float(numeric_parts[0])
                        our_q1 = float(numeric_parts[1])
                        our_med = float(numeric_parts[2])
                        our_q3 = float(numeric_parts[3])
                        our_max = float(numeric_parts[4])
                        break
        
        # Vergleich (mit etwas Toleranz wegen Formatierung)
        if our_min is not None:
            assert abs(our_min - r_quantiles[0]) < 0.1
            assert abs(our_q1 - r_quantiles[1]) < 0.1
            assert abs(our_med - r_quantiles[2]) < 0.1
            assert abs(our_q3 - r_quantiles[3]) < 0.1
            assert abs(our_max - r_quantiles[4]) < 0.1
        else:
            pytest.fail("Konnte Residuen-Quantile nicht aus Output extrahieren")
