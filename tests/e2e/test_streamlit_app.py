import pytest
from streamlit.testing.v1 import AppTest
import sys
import os

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

class TestStreamlitApp:
    def test_app_startup(self):
        """Test that the app starts up correctly."""
        at = AppTest.from_file("run.py")
        at.run()
        assert "Simple Regression" in at.title[0].value
        assert not at.exception
        
    def test_sidebar_interactions(self):
        """Test sidebar interactions and state changes."""
        at = AppTest.from_file("run.py")
        at.run()
        assert at.sidebar.radio[0].value == "Simple Regression"
        at.sidebar.radio[0].set_value("Multiple Regression").run()
        assert "Multiple Regression" in at.title[0].value
        assert not at.exception
        
    def test_dataset_selection(self):
        """Test dataset selection updates."""
        at = AppTest.from_file("run.py")
        at.run()
        if len(at.sidebar.selectbox) > 0:
            initial_dataset = at.sidebar.selectbox[0].value
            if len(at.sidebar.selectbox[0].options) > 1:
                new_option = at.sidebar.selectbox[0].options[1]
                at.sidebar.selectbox[0].set_value(new_option).run()
                assert not at.exception
                assert at.sidebar.selectbox[0].value == new_option

    def test_parameter_changes(self):
        """Test parameter sliders."""
        at = AppTest.from_file("run.py")
        at.run()
        if len(at.sidebar.slider) > 1:
            at.sidebar.slider[1].set_value(1.0).run() 
            assert not at.exception

    def test_ai_interpretation_section(self):
        """Test AI interpretation section presence."""
        at = AppTest.from_file("run.py")
        at.run()
        assert "🤖 AI-Interpretation des R-Outputs" in [h.value for h in at.subheader]

    def test_content_rendering(self):
        """Test that API content and plots are actually rendered."""
        at = AppTest.from_file("run.py")
        at.run(timeout=10)
        
        assert not at.exception
        
        # Verify Metrics
        metrics = at.metric
        assert len(metrics) > 0, "No metrics were rendered!"
        
        found_r2 = False
        for metric in metrics:
            # Check label for R²
            if "R²" in metric.label:
                found_r2 = True
                break
        assert found_r2, "R² metric not found!"
        
        # Verify Dataframes
        dataframes = at.dataframe
        assert len(dataframes) > 0, "No dataframes were rendered!"
        
        # Verify Content Sections
        headers = [h.value for h in at.header]
        assert any("Kapitel" in h for h in headers), "No content chapters found!"
        
        # Check if plotly_chart is available (might be missing in some test envs)
        if hasattr(at, 'plotly_chart'):
             assert len(at.plotly_chart) > 0, "No plots were rendered!"

    def test_multiple_regression_content(self):
        """Test content rendering for Multiple Regression."""
        at = AppTest.from_file("run.py")
        at.run()
        
        at.sidebar.radio[0].set_value("Multiple Regression").run()
        
        assert not at.exception
        assert "Multiple Regression" in at.title[0].value
        
        metrics = at.metric
        found_f = False
        for metric in metrics:
            # Label is "F", help is "F-Statistik"
            # We check if label is "F" or help contains "F-Statistik" (if exposed)
            if metric.label == "F":
                found_f = True
                break
            if "F-Statistik" in metric.label:
                found_f = True
                break
        
        assert found_f, f"F-Statistic metric not found! Labels: {[m.label for m in metrics]}"
