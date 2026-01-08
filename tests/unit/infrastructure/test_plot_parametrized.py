"""
Parametrisierte Tests für PlotBuilder.

Testet verschiedene Datensätze und Edge Cases.
"""
import pytest
import numpy as np

from src.infrastructure.services.plot import PlotBuilder
from src.infrastructure.services.calculate import RegressionResult
from src.infrastructure.data.generators import DataResult


class TestPlotBuilderParametrized:
    """Parametrisierte Tests für PlotBuilder."""
    
    @pytest.fixture
    def plotter(self):
        return PlotBuilder()
    
    @pytest.mark.parametrize("n_points", [2, 10, 50, 100, 1000])
    def test_simple_regression_plots_various_sizes(self, plotter, n_points):
        """Test Plot-Erstellung mit verschiedenen Datenmengen."""
        np.random.seed(42)
        x = np.linspace(0, 10, n_points)
        y = 2 * x + 1 + np.random.normal(0, 0.5, n_points)
        
        data = DataResult(
            x=x,
            y=y,
            x_label="X",
            y_label="Y",
            n=n_points
        )
        
        # Einfaches Ergebnis berechnen
        from src.infrastructure.services.calculate import StatisticsCalculator
        calc = StatisticsCalculator()
        result = calc.simple_regression(x, y)
        
        plots = plotter.simple_regression_plots(data, result)
        
        assert plots is not None
        assert plots.scatter is not None
        assert plots.residuals is not None
    
    @pytest.mark.parametrize("r_squared", [0.0, 0.3, 0.5, 0.7, 0.9, 1.0])
    def test_plots_with_various_r_squared(self, plotter, r_squared):
        """Test Plots mit verschiedenen R²-Werten."""
        np.random.seed(42)
        n = 50
        x = np.linspace(0, 10, n)
        
        # Generiere y mit kontrolliertem R²
        if r_squared == 1.0:
            y = 2 * x + 1  # Perfekte Korrelation
        elif r_squared == 0.0:
            y = np.random.normal(0, 1, n)  # Keine Korrelation
        else:
            # Berechne Noise-Level für gewünschtes R²
            signal = 2 * x + 1
            noise_std = np.std(signal) * np.sqrt((1 - r_squared) / r_squared)
            y = signal + np.random.normal(0, noise_std, n)
        
        data = DataResult(x=x, y=y, x_label="X", y_label="Y", n=n)
        
        from src.infrastructure.services.calculate import StatisticsCalculator
        calc = StatisticsCalculator()
        result = calc.simple_regression(x, y)
        
        plots = plotter.simple_regression_plots(data, result)
        
        assert plots is not None
        # R² sollte nahe am erwarteten Wert sein (mit Toleranz)
        assert abs(result.r_squared - r_squared) < 0.2 or r_squared == 0.0
    
    @pytest.mark.parametrize("noise_level", [0.0, 0.1, 0.5, 1.0, 2.0, 5.0])
    def test_plots_with_various_noise(self, plotter, noise_level):
        """Test Plots mit verschiedenen Noise-Leveln."""
        np.random.seed(42)
        n = 50
        x = np.linspace(0, 10, n)
        y = 2 * x + 1 + np.random.normal(0, noise_level, n)
        
        data = DataResult(x=x, y=y, x_label="X", y_label="Y", n=n)
        
        from src.infrastructure.services.calculate import StatisticsCalculator
        calc = StatisticsCalculator()
        result = calc.simple_regression(x, y)
        
        plots = plotter.simple_regression_plots(data, result)
        
        assert plots is not None
        # Mit mehr Noise sollte R² niedriger sein
        if noise_level > 0:
            assert result.r_squared < 1.0
