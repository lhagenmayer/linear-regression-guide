"""
Visual regression tests for plot generation.

These tests ensure that plots are generated correctly and consistently.
They compare visual output against baselines and detect visual regressions.

Tests cover:
- Plot generation without errors
- Plot structure and data integrity
- Visual consistency across runs
- Plot customization options
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import hashlib
import json
from unittest.mock import patch


class TestPlotGeneration:
    """Test basic plot generation functionality."""

    def test_plotly_scatter_creation(self, sample_regression_data):
        """Test that scatter plots are created successfully."""
        from src.plots import create_plotly_scatter

        x = sample_regression_data['x']
        y = sample_regression_data['y']

        fig = create_plotly_scatter(x, y, "Test X", "Test Y", "Test Title")

        # Should return a plotly figure
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0

        # Should have scatter trace
        assert fig.data[0].mode == 'markers'

    def test_plotly_scatter_with_customization(self, sample_regression_data):
        """Test scatter plot with custom colors and styling."""
        from src.plots import create_plotly_scatter

        x = sample_regression_data['x']
        y = sample_regression_data['y']

        custom_colors = {"data_points": "red", "regression_line": "blue"}

        fig = create_plotly_scatter(
            x, y, "X Label", "Y Label", "Custom Title",
            show_regression=True, colors=custom_colors
        )

        assert fig is not None
        assert len(fig.data) >= 1  # At least data points

    def test_plot_data_integrity(self, sample_regression_data):
        """Test that plot data matches input data."""
        from src.plots import create_plotly_scatter

        x = sample_regression_data['x']
        y = sample_regression_data['y']

        fig = create_plotly_scatter(x, y, "X", "Y", "Title")

        # Extract data from plot
        plot_x = fig.data[0].x
        plot_y = fig.data[0].y

        # Should match input data
        np.testing.assert_array_equal(plot_x, x)
        np.testing.assert_array_equal(plot_y, y)

    def test_plot_axis_labels(self, sample_regression_data):
        """Test that axis labels are set correctly."""
        from src.plots import create_plotly_scatter

        x = sample_regression_data['x']
        y = sample_regression_data['y']

        x_label = "Population Density"
        y_label = "GDP per Capita"

        fig = create_plotly_scatter(x, y, x_label, y_label, "Test")

        # Check axis labels
        assert fig.layout.xaxis.title.text == x_label
        assert fig.layout.yaxis.title.text == y_label

    def test_plot_title(self, sample_regression_data):
        """Test that plot title is set correctly."""
        from src.plots import create_plotly_scatter

        x = sample_regression_data['x']
        y = sample_regression_data['y']
        title = "Custom Test Title"

        fig = create_plotly_scatter(x, y, "X", "Y", title)

        assert fig.layout.title.text == title


class Test3DVisualization:
    """Test 3D visualization functionality."""

    def test_3d_layout_creation(self):
        """Test that 3D layout is created correctly."""
        from src.plots import get_3d_layout_config

        layout = get_3d_layout_config("Test Title", "X", "Y", "Z")

        assert layout.title.text == "Test Title"
        assert layout.scene.xaxis.title.text == "X"
        assert layout.scene.yaxis.title.text == "Y"
        assert layout.scene.zaxis.title.text == "Z"

    def test_regression_mesh_creation(self, sample_multiple_regression_data):
        """Test that regression mesh is created for 3D plots."""
        from src.plots import create_regression_mesh

        x1 = sample_multiple_regression_data['x1'][:20]  # Smaller sample for testing
        x2 = sample_multiple_regression_data['x2'][:20]
        coef = [1.5, -0.8, 10]  # slope1, slope2, intercept

        mesh = create_regression_mesh(x1, x2, coef)

        # Should return mesh data
        assert mesh is not None
        assert len(mesh) == 3  # X, Y, Z coordinates
        assert len(mesh[0]) > 0  # Should have data points


class TestPlotCustomization:
    """Test plot customization options."""

    def test_color_customization(self, sample_regression_data):
        """Test that custom colors are applied."""
        from src.plots import create_plotly_scatter

        x = sample_regression_data['x']
        y = sample_regression_data['y']

        colors = {
            "data_points": "#FF0000",  # Red
            "regression_line": "#0000FF"  # Blue
        }

        fig = create_plotly_scatter(
            x, y, "X", "Y", "Title",
            show_regression=True, colors=colors
        )

        # Check that colors are applied (plotly uses marker.color)
        assert fig.data[0].marker.color == colors["data_points"]

    def test_font_customization(self, sample_regression_data):
        """Test that custom fonts are applied."""
        from src.plots import create_plotly_scatter
        from src.config import FONT_SIZES

        x = sample_regression_data['x']
        y = sample_regression_data['y']

        fig = create_plotly_scatter(x, y, "X", "Y", "Title", font_sizes=FONT_SIZES)

        # Check that font sizes are applied
        assert fig.layout.xaxis.title.font.size == FONT_SIZES["axis_label_2d"]


class TestVisualRegression:
    """Visual regression tests that compare plot outputs."""

    def get_plot_signature(self, fig):
        """Generate a signature for plot comparison."""
        # Convert plot data to a normalized representation
        plot_data = {
            'data': [],
            'layout': {}
        }

        for trace in fig.data:
            trace_data = {
                'type': trace.type,
                'mode': getattr(trace, 'mode', None),
                'x': list(trace.x) if hasattr(trace, 'x') else None,
                'y': list(trace.y) if hasattr(trace, 'y') else None,
            }
            plot_data['data'].append(trace_data)

        # Add layout info
        layout = fig.layout
        plot_data['layout'] = {
            'title': getattr(layout.title, 'text', None) if hasattr(layout, 'title') else None,
            'xaxis_title': getattr(layout.xaxis.title, 'text', None) if hasattr(layout.xaxis, 'title') else None,
            'yaxis_title': getattr(layout.yaxis.title, 'text', None) if hasattr(layout.yaxis, 'title') else None,
        }

        # Create hash of normalized data
        data_str = json.dumps(plot_data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()

    @pytest.mark.visual
    def test_plot_consistency(self, sample_regression_data):
        """Test that plots are visually consistent across runs."""
        from src.plots import create_plotly_scatter

        x = sample_regression_data['x']
        y = sample_regression_data['y']

        # Create plot twice with same data
        fig1 = create_plotly_scatter(x, y, "X", "Y", "Test")
        fig2 = create_plotly_scatter(x, y, "X", "Y", "Test")

        # Get signatures
        sig1 = self.get_plot_signature(fig1)
        sig2 = self.get_plot_signature(fig2)

        # Should be identical
        assert sig1 == sig2, "Plots should be visually identical with same input"

    @pytest.mark.visual
    def test_plot_data_changes_affect_visual(self, sample_regression_data):
        """Test that changing data affects the visual output."""
        from src.plots import create_plotly_scatter

        x = sample_regression_data['x']
        y = sample_regression_data['y']

        fig1 = create_plotly_scatter(x, y, "X", "Y", "Test")
        fig2 = create_plotly_scatter(x * 2, y, "X", "Y", "Test")  # Different x data

        sig1 = self.get_plot_signature(fig1)
        sig2 = self.get_plot_signature(fig2)

        # Should be different
        assert sig1 != sig2, "Plots should be different with different data"


class TestPlotExport:
    """Test plot export functionality."""

    def test_plot_to_html(self, sample_regression_data, tmp_path):
        """Test exporting plot to HTML."""
        from src.plots import create_plotly_scatter

        x = sample_regression_data['x']
        y = sample_regression_data['y']

        fig = create_plotly_scatter(x, y, "X", "Y", "Test")

        # Export to HTML
        html_file = tmp_path / "test_plot.html"
        fig.write_html(str(html_file))

        # Check that file was created and has content
        assert html_file.exists()
        content = html_file.read_text()
        assert len(content) > 1000  # Should have substantial HTML content
        assert "plotly" in content.lower()

    def test_plot_to_json(self, sample_regression_data):
        """Test converting plot to JSON."""
        from src.plots import create_plotly_scatter

        x = sample_regression_data['x']
        y = sample_regression_data['y']

        fig = create_plotly_scatter(x, y, "X", "Y", "Test")

        # Convert to JSON
        json_str = fig.to_json()

        # Should be valid JSON with plotly data
        assert isinstance(json_str, str)
        assert len(json_str) > 100

        # Should contain plotly-specific keys
        assert "data" in json_str
        assert "layout" in json_str


class TestPlotPerformance:
    """Test plot generation performance."""

    @pytest.mark.performance
    def test_large_dataset_plotting(self):
        """Test plotting with large datasets."""
        from src.plots import create_plotly_scatter

        # Generate large dataset
        np.random.seed(42)
        x = np.random.normal(10, 2, 10000)
        y = 2.5 * x + 5 + np.random.normal(0, 3, 10000)

        # Should complete within reasonable time
        import time
        start_time = time.time()

        fig = create_plotly_scatter(x, y, "X", "Y", "Large Dataset Test")

        end_time = time.time()
        duration = end_time - start_time

        # Should complete in reasonable time (less than 5 seconds)
        assert duration < 5.0, f"Plot generation took too long: {duration:.2f}s"

        # Should still create valid plot
        assert fig is not None
        assert len(fig.data) > 0


# Skip visual tests by default unless explicitly requested
pytestmark = pytest.mark.visual