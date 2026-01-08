"""
Unit Tests für Serializers.

Testet die JSON-Serialisierung aller Datenstrukturen.
"""
import pytest
import numpy as np
from dataclasses import dataclass
from unittest.mock import Mock, patch

from src.api.serializers import (
    DataSerializer,
    StatsSerializer,
    PlotSerializer,
    ContentSerializer,
    PipelineSerializer,
    _to_list,
    _to_float
)


class TestHelperFunctions:
    """Tests für Helper-Funktionen."""
    
    def test_to_list_numpy_array(self):
        """Test Konvertierung von NumPy-Array zu Liste."""
        arr = np.array([1, 2, 3])
        result = _to_list(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)
    
    def test_to_list_python_list(self):
        """Test dass Python-Listen unverändert bleiben."""
        lst = [1, 2, 3]
        result = _to_list(lst)
        assert result == [1, 2, 3]
    
    def test_to_list_none(self):
        """Test None-Handling."""
        assert _to_list(None) == []
    
    def test_to_float_normal(self):
        """Test normale Float-Konvertierung."""
        assert _to_float(1.5) == 1.5
        assert _to_float(42) == 42.0
        assert _to_float("3.14") == 3.14
    
    def test_to_float_nan(self):
        """Test NaN-Handling."""
        assert _to_float(np.nan) is None
        assert _to_float(float('nan')) is None
    
    def test_to_float_inf(self):
        """Test Infinity-Handling."""
        assert _to_float(np.inf) is None
        assert _to_float(float('inf')) is None
    
    def test_to_float_none(self):
        """Test None-Handling."""
        assert _to_float(None) is None


class TestDataSerializer:
    """Tests für DataSerializer."""
    
    @pytest.fixture
    def simple_data(self):
        """Erstellt einfache Regressionsdaten."""
        from src.infrastructure.data.generators import DataResult
        
        return DataResult(
            x=np.array([1, 2, 3]),
            y=np.array([2, 4, 6]),
            x_label="X",
            y_label="Y",
            x_unit="units",
            y_unit="units",
            context_title="Test",
            context_description="Test description",
            extra={"key": "value"}
        )
    
    @pytest.fixture
    def multiple_data(self):
        """Erstellt multiple Regressionsdaten."""
        from src.infrastructure.data.generators import MultipleRegressionDataResult
        
        return MultipleRegressionDataResult(
            x1=np.array([1, 2]),
            x2=np.array([3, 4]),
            y=np.array([5, 6]),
            x1_label="X1",
            x2_label="X2",
            y_label="Y",
            extra={"key": "value"}
        )
    
    def test_serialize_simple(self, simple_data):
        """Test Serialisierung einfacher Daten."""
        result = DataSerializer.serialize_simple(simple_data)
        
        assert result["type"] == "simple_regression_data"
        assert result["x"] == [1, 2, 3]
        assert result["y"] == [2, 4, 6]
        assert result["n"] == 3
        assert result["x_label"] == "X"
        assert result["y_label"] == "Y"
        assert result["context"]["title"] == "Test"
        assert result["extra"]["key"] == "value"
    
    def test_serialize_multiple(self, multiple_data):
        """Test Serialisierung multipler Daten."""
        result = DataSerializer.serialize_multiple(multiple_data)
        
        assert result["type"] == "multiple_regression_data"
        assert result["x1"] == [1, 2]
        assert result["x2"] == [3, 4]
        assert result["y"] == [5, 6]
        assert result["n"] == 2


class TestStatsSerializer:
    """Tests für StatsSerializer."""
    
    @pytest.fixture
    def simple_result(self):
        """Erstellt einfaches Regressionsergebnis."""
        @dataclass
        class SimpleResult:
            intercept: float
            slope: float
            r_squared: float
            r_squared_adj: float
            se_intercept: float
            se_slope: float
            t_intercept: float
            t_slope: float
            p_intercept: float
            p_slope: float
            sse: float
            sst: float
            ssr: float
            mse: float
            n: int
            df: int
            y_pred: np.ndarray
            residuals: np.ndarray
            extra: dict = None
        
        return SimpleResult(
            intercept=1.0,
            slope=2.0,
            r_squared=0.95,
            r_squared_adj=0.94,
            se_intercept=0.1,
            se_slope=0.2,
            t_intercept=10.0,
            t_slope=10.0,
            p_intercept=0.001,
            p_slope=0.001,
            sse=10.0,
            sst=200.0,
            ssr=190.0,
            mse=0.1,
            n=100,
            df=98,
            y_pred=np.array([3, 5, 7]),
            residuals=np.array([0.1, -0.1, 0.0]),
            extra={"f_statistic": 100.0}
        )
    
    def test_serialize_simple(self, simple_result):
        """Test Serialisierung einfacher Statistiken."""
        result = StatsSerializer.serialize_simple(simple_result)
        
        assert result["type"] == "simple_regression_stats"
        assert result["coefficients"]["intercept"] == 1.0
        assert result["coefficients"]["slope"] == 2.0
        assert result["model_fit"]["r_squared"] == 0.95
        assert result["t_tests"]["slope"]["t_value"] == 10.0
        assert result["predictions"] == [3, 5, 7]
        assert result["residuals"] == [0.1, -0.1, 0.0]
        assert result["sample"]["n"] == 100
        assert result["extra"]["f_statistic"] == 100.0
    
    def test_serialize_with_nan(self, simple_result):
        """Test Serialisierung mit NaN-Werten."""
        simple_result.r_squared = np.nan
        result = StatsSerializer.serialize_simple(simple_result)
        
        assert result["model_fit"]["r_squared"] is None


class TestPlotSerializer:
    """Tests für PlotSerializer."""
    
    def test_serialize_figure(self):
        """Test Serialisierung von Plotly-Figuren."""
        import plotly.graph_objects as go
        
        fig = go.Figure(
            data=[go.Scatter(x=[1, 2, 3], y=[2, 4, 6])],
            layout=go.Layout(title="Test")
        )
        
        result = PlotSerializer.serialize_figure(fig)
        
        assert "data" in result
        assert "layout" in result
        assert result["layout"]["title"]["text"] == "Test"
    
    def test_serialize_collection(self):
        """Test Serialisierung von PlotCollection."""
        from src.infrastructure.services.plot import PlotCollection
        import plotly.graph_objects as go
        
        scatter = go.Figure(data=[go.Scatter(x=[1, 2], y=[2, 4])])
        residuals = go.Figure(data=[go.Scatter(x=[1, 2], y=[0, 0])])
        
        plots = PlotCollection(scatter=scatter, residuals=residuals)
        
        result = PlotSerializer.serialize_collection(plots)
        
        assert "scatter" in result
        assert "residuals" in result


class TestPipelineSerializer:
    """Tests für PipelineSerializer."""
    
    def test_serialize_with_predictions(self):
        """Test Serialisierung mit Vorhersagen."""
        from src.infrastructure.regression_pipeline import PipelineResult
        from src.infrastructure.data.generators import DataResult
        import numpy as np
        
        # Erstelle echte Objekte
        data = DataResult(
            x=np.array([1, 2, 3]),
            y=np.array([2, 4, 6]),
            x_label="X",
            y_label="Y"
        )
        
        from src.infrastructure.services.calculate import RegressionResult
        stats = RegressionResult(
            intercept=0.0,
            slope=2.0,
            y_pred=np.array([2, 4, 6]),
            residuals=np.array([0, 0, 0]),
            r_squared=1.0,
            r_squared_adj=1.0,
            se_intercept=0.0,
            se_slope=0.0,
            t_intercept=0.0,
            t_slope=0.0,
            p_intercept=1.0,
            p_slope=1.0,
            sse=0.0,
            sst=8.0,
            ssr=8.0,
            mse=0.0,
            n=3,
            df=1
        )
        
        from src.infrastructure.services.plot import PlotCollection
        import plotly.graph_objects as go
        plots = PlotCollection(
            scatter=go.Figure(),
            residuals=go.Figure()
        )
        
        pipeline_result = PipelineResult(
            data=data,
            stats=stats,
            plots=plots,
            pipeline_type="simple",
            params={}
        )
        
        result = PipelineSerializer.serialize(pipeline_result, include_predictions=True)
        
        assert "data" in result
        assert "stats" in result
        assert "plots" in result
    
    def test_serialize_without_predictions(self):
        """Test Serialisierung ohne Vorhersagen."""
        from src.infrastructure.regression_pipeline import PipelineResult
        from src.infrastructure.data.generators import DataResult
        import numpy as np
        
        data = DataResult(
            x=np.array([1, 2, 3]),
            y=np.array([2, 4, 6]),
            x_label="X",
            y_label="Y"
        )
        
        from src.infrastructure.services.calculate import RegressionResult
        stats = RegressionResult(
            intercept=0.0,
            slope=2.0,
            y_pred=np.array([2, 4, 6]),
            residuals=np.array([0, 0, 0]),
            r_squared=1.0,
            r_squared_adj=1.0,
            se_intercept=0.0,
            se_slope=0.0,
            t_intercept=0.0,
            t_slope=0.0,
            p_intercept=1.0,
            p_slope=1.0,
            sse=0.0,
            sst=8.0,
            ssr=8.0,
            mse=0.0,
            n=3,
            df=1
        )
        
        from src.infrastructure.services.plot import PlotCollection
        import plotly.graph_objects as go
        plots = PlotCollection(
            scatter=go.Figure(),
            residuals=go.Figure()
        )
        
        pipeline_result = PipelineResult(
            data=data,
            stats=stats,
            plots=plots,
            pipeline_type="simple",
            params={}
        )
        
        result = PipelineSerializer.serialize(pipeline_result, include_predictions=False)
        
        assert "data" in result
        assert "stats" in result
