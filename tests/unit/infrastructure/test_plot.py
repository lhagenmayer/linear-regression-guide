"""
Unit Tests für PlotBuilder.

Testet die Visualisierungs-Funktionalität.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.infrastructure.services.plot import PlotBuilder, PlotCollection
from src.infrastructure.services.calculate import RegressionResult, MultipleRegressionResult
from src.infrastructure.data.generators import DataResult, MultipleRegressionDataResult
from src.core.domain.value_objects import ClassificationResult, ClassificationMetrics


class TestPlotBuilder:
    """Tests für PlotBuilder."""
    
    @pytest.fixture
    def plotter(self):
        """Erstellt einen PlotBuilder."""
        return PlotBuilder()
    
    @pytest.fixture
    def simple_data(self):
        """Erstellt einfache Regressionsdaten."""
        return DataResult(
            x=np.array([1, 2, 3, 4, 5]),
            y=np.array([2, 4, 6, 8, 10]),
            x_label="X",
            y_label="Y"
        )
    
    @pytest.fixture
    def simple_result(self):
        """Erstellt einfaches Regressionsergebnis."""
        return RegressionResult(
            intercept=0.0,
            slope=2.0,
            y_pred=np.array([2, 4, 6, 8, 10]),
            residuals=np.array([0, 0, 0, 0, 0]),
            r_squared=1.0,
            r_squared_adj=1.0,
            se_intercept=0.0,
            se_slope=0.0,
            t_intercept=0.0,
            t_slope=0.0,
            p_intercept=1.0,
            p_slope=1.0,
            sse=0.0,
            sst=40.0,
            ssr=40.0,
            mse=0.0,
            n=5,
            df=3
        )
    
    @pytest.fixture
    def multiple_data(self):
        """Erstellt multiple Regressionsdaten."""
        return MultipleRegressionDataResult(
            x1=np.array([1, 2, 3]),
            x2=np.array([4, 5, 6]),
            y=np.array([10, 12, 14]),
            x1_label="X1",
            x2_label="X2",
            y_label="Y"
        )
    
    @pytest.fixture
    def multiple_result(self):
        """Erstellt multiples Regressionsergebnis."""
        return MultipleRegressionResult(
            intercept=0.0,
            coefficients=[2.0, 1.0],  # Liste, nicht Array
            y_pred=np.array([10, 12, 14]),
            residuals=np.array([0, 0, 0]),
            r_squared=1.0,
            r_squared_adj=1.0,
            f_statistic=100.0,
            f_pvalue=0.001,
            se_coefficients=[0.0, 0.0, 0.0],  # se_coefficients, nicht standard_errors
            t_values=[0.0, 0.0, 0.0],
            p_values=[1.0, 1.0, 1.0],
            sse=0.0,
            sst=8.0,
            ssr=8.0,
            n=3,
            k=2
        )
    
    def test_simple_regression_plots(self, plotter, simple_data, simple_result):
        """Test Erstellung einfacher Regressions-Plots."""
        plots = plotter.simple_regression_plots(simple_data, simple_result)
        
        assert isinstance(plots, PlotCollection)
        assert plots.scatter is not None
        assert plots.residuals is not None
        assert plots.diagnostics is not None
    
    def test_multiple_regression_plots(self, plotter, multiple_data, multiple_result):
        """Test Erstellung multipler Regressions-Plots."""
        plots = plotter.multiple_regression_plots(multiple_data, multiple_result)
        
        assert isinstance(plots, PlotCollection)
        assert plots.scatter is not None
        assert plots.residuals is not None
    
    def test_classification_plots(self, plotter):
        """Test Erstellung von Klassifikations-Plots."""
        from src.infrastructure.data.generators import ClassificationDataResult
        
        # Mock ClassificationResult
        metrics = ClassificationMetrics(
            accuracy=0.95,
            precision=0.94,
            recall=0.96,
            f1_score=0.95,
            confusion_matrix=np.array([[45, 5], [3, 47]]),
            auc=0.98
        )
        
        result = ClassificationResult(
            classes=[0, 1],
            predictions=np.array([0, 1, 0, 1]),
            probabilities=np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]]),
            metrics=metrics,
            model_params={"weights": np.array([1.0, 1.0]), "intercept": 0.0}
        )
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        
        data = ClassificationDataResult(
            X=X,
            y=y,
            feature_names=["Feature1", "Feature2"],
            target_names=["Class0", "Class1"]
        )
        
        plots = plotter.classification_plots(data, result)
        
        assert isinstance(plots, PlotCollection)
        assert plots.scatter is not None  # Main plot
        assert plots.residuals is not None  # Confusion matrix
        assert plots.diagnostics is not None  # ROC curve
    
    def test_plot_collection_initialization(self):
        """Test PlotCollection-Initialisierung."""
        scatter = Mock()
        residuals = Mock()
        
        collection = PlotCollection(scatter=scatter, residuals=residuals)
        
        assert collection.scatter is scatter
        assert collection.residuals is residuals
        assert collection.diagnostics is None
        assert collection.extra == {}


class TestPlotBuilderEdgeCases:
    """Tests für Edge Cases im PlotBuilder."""
    
    @pytest.fixture
    def plotter(self):
        return PlotBuilder()
    
    def test_empty_data(self, plotter):
        """Test mit leeren Daten - sollte graceful handhaben."""
        data = DataResult(
            x=np.array([]),
            y=np.array([]),
            x_label="X",
            y_label="Y"
        )
        
        result = RegressionResult(
            intercept=0.0,
            slope=0.0,
            y_pred=np.array([]),
            residuals=np.array([]),
            r_squared=0.0,
            r_squared_adj=0.0,
            se_intercept=0.0,
            se_slope=0.0,
            t_intercept=0.0,
            t_slope=0.0,
            p_intercept=1.0,
            p_slope=1.0,
            sse=0.0,
            sst=0.0,
            ssr=0.0,
            mse=0.0,
            n=0,
            df=0
        )
        
        # Leere Daten sollten graceful gehandhabt werden
        # Entweder eine Exception (was OK ist) oder leere Plots
        try:
            plots = plotter.simple_regression_plots(data, result)
            # Wenn es funktioniert, sollte es ein PlotCollection sein
            assert plots is not None
        except (ValueError, ZeroDivisionError, IndexError):
            # Es ist OK, wenn leere Daten eine Exception werfen
            pass
    
    def test_single_data_point(self, plotter):
        """Test mit nur einem Datenpunkt."""
        data = DataResult(
            x=np.array([1]),
            y=np.array([2]),
            x_label="X",
            y_label="Y"
        )
        
        result = RegressionResult(
            intercept=2.0,
            slope=0.0,
            y_pred=np.array([2]),
            residuals=np.array([0]),
            r_squared=0.0,
            r_squared_adj=0.0,
            se_intercept=0.0,
            se_slope=0.0,
            t_intercept=0.0,
            t_slope=0.0,
            p_intercept=1.0,
            p_slope=1.0,
            sse=0.0,
            sst=0.0,
            ssr=0.0,
            mse=0.0,
            n=1,
            df=0
        )
        
        plots = plotter.simple_regression_plots(data, result)
        assert plots is not None
