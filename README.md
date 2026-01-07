# 📊 Linear Regression Guide

Ein interaktives, didaktisches Tool für lineare Regressionsanalyse.

**Frontend-Agnostisch:** Läuft sowohl mit **Streamlit** als auch mit **Flask** - automatische Framework-Erkennung!

## 🎯 Architektur

```
┌─────────────────────────────────────────────────────────────────┐
│                 CORE PIPELINE (Framework-Agnostic)              │
│  ┌─────────┐   ┌───────────┐   ┌──────────┐   ┌─────────────┐  │
│  │   GET   │ → │ CALCULATE │ → │   PLOT   │ → │   DISPLAY   │  │
│  │  Data   │   │   Stats   │   │  Plotly  │   │   Prepare   │  │
│  └─────────┘   └───────────┘   └──────────┘   └─────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    FRAMEWORK ADAPTERS                           │
│  ┌────────────────────────┐    ┌────────────────────────┐      │
│  │       STREAMLIT        │    │         FLASK          │      │
│  │  ┌──────────────────┐  │    │  ┌──────────────────┐  │      │
│  │  │ Educational Tabs │  │    │  │  HTML Templates  │  │      │
│  │  │   (st.* calls)   │  │    │  │   (Jinja2)       │  │      │
│  │  └──────────────────┘  │    │  └──────────────────┘  │      │
│  └────────────────────────┘    └────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Schnellstart

### Option 1: Streamlit (Interaktiv)
```bash
pip install -r requirements.txt
streamlit run run.py
```

### Option 2: Flask (Traditionell)
```bash
pip install -r requirements.txt
python run.py
# oder: flask --app src.adapters.flask_app:create_flask_app run
```

### Option 3: WSGI Server (Production)
```bash
gunicorn "run:create_app()"
```

## 📁 Projektstruktur

```
src/
├── pipeline/                    # CORE (Framework-Agnostic)
│   ├── get_data.py             # Step 1: Data fetching
│   ├── calculate.py            # Step 2: Statistics
│   ├── plot.py                 # Step 3: Plotly figures
│   ├── display.py              # Step 4: Data preparation
│   └── regression_pipeline.py  # Orchestrator
│
├── adapters/                    # FRAMEWORK ADAPTERS
│   ├── detector.py             # Auto-detection
│   ├── base.py                 # Abstract interface
│   │
│   ├── streamlit/              # Streamlit-specific
│   │   ├── app.py              # StreamlitRenderer
│   │   ├── simple_regression_educational.py   # st.* UI
│   │   └── multiple_regression_educational.py # st.* UI
│   │
│   ├── flask_app.py            # Flask renderer
│   └── templates/              # HTML templates
│       ├── base.html
│       ├── index.html
│       ├── simple_regression.html
│       └── multiple_regression.html
│
├── data/content.py             # Dynamic content
└── config/                     # Configuration

run.py                          # Unified entry point
```

## 🔄 Auto-Detection

| Aufruf | Erkanntes Framework |
|--------|---------------------|
| `streamlit run run.py` | Streamlit |
| `python run.py` | Flask |
| `REGRESSION_FRAMEWORK=streamlit` | Streamlit (explizit) |
| `gunicorn "run:create_app()"` | Flask (WSGI) |

## 💻 API Usage

```python
from src.pipeline import RegressionPipeline

# Pipeline ist komplett framework-agnostisch
pipeline = RegressionPipeline()

# Einfache Regression
result = pipeline.run_simple(dataset="electronics", n=100, seed=42)
print(f"R² = {result.stats.r_squared:.4f}")

# Multiple Regression
result = pipeline.run_multiple(dataset="cities", n=100, seed=42)
print(f"F = {result.stats.f_statistic:.2f}")
```

## 🏗️ Custom Adapter erstellen

```python
from src.adapters.base import BaseRenderer, RenderContext

class MyRenderer(BaseRenderer):
    def render(self, context: RenderContext):
        # Use context.to_dict() for template data
        data = context.to_dict()
        # Render with your framework...
    
    def render_simple_regression(self, context):
        pass
    
    def render_multiple_regression(self, context):
        pass
    
    def run(self, host, port, debug):
        # Start your server
        pass
```

## 🧪 Tests

```bash
pytest tests/ -v
# 26 tests covering pipeline + adapters
```

## 📦 Dependencies

```
# Core (required)
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
plotly>=5.18.0

# Frameworks (at least one)
streamlit>=1.28.0   # For interactive app
flask>=3.0.0        # For traditional web app
```

## 📄 Lizenz

MIT License
