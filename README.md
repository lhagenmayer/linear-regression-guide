# 📊 Regression Analysis Application

**Truly Frontend-Agnostic Statistical Learning Platform**

Eine interaktive Lernplattform für Regressionsanalyse, die sowohl mit **Streamlit** als auch mit **Flask** läuft - mit **identischem** Educational Content.

## 🏗️ Architektur: Option B - Content als Datenstruktur

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              run.py                                      │
│                         (Auto-Detection)                                 │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                ↓                                 ↓
    ┌───────────────────────┐         ┌───────────────────────┐
    │   Streamlit Frontend  │         │    Flask Frontend     │
    │   adapters/streamlit/ │         │  adapters/flask_app   │
    └───────────────────────┘         └───────────────────────┘
                │                                 │
                └────────────────┬────────────────┘
                                 ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    ContentBuilder (content/)                         │
    │        SimpleRegressionContent / MultipleRegressionContent           │
    │                                                                      │
    │   → Definiert Educational Content als DATENSTRUKTUREN               │
    │   → KEINE UI-Imports, KEINE Framework-Abhängigkeiten                │
    └────────────────────────────────┬────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ↓                                 ↓
    ┌─────────────────────────────┐   ┌─────────────────────────────┐
    │  StreamlitContentRenderer   │   │    HTMLContentRenderer      │
    │   (interprets → st.*)       │   │   (interprets → HTML)       │
    │   adapters/renderers/       │   │   adapters/renderers/       │
    └─────────────────────────────┘   └─────────────────────────────┘
```

### Warum Option B?

**Option A (vorher):** UI-Code in separaten Dateien für jedes Framework
- ❌ Code-Duplizierung
- ❌ Änderungen müssen zweimal gemacht werden
- ❌ Inhalt und Darstellung vermischt

**Option B (jetzt):** Content als Datenstruktur
- ✅ **KEINE Code-Duplizierung** - Content wird einmal definiert
- ✅ **Single Source of Truth** - Ein ContentBuilder für alle Frontends
- ✅ **Saubere Trennung** - Content ≠ Rendering
- ✅ **Einfache Erweiterung** - Neuer Renderer = Neues Frontend

## 📁 Projektstruktur

```
src/
├── content/                     # 📖 EDUCATIONAL CONTENT (Framework-Agnostic)
│   ├── __init__.py
│   ├── structure.py             # Content-Datenstrukturen (Chapter, Section, etc.)
│   ├── builder.py               # Base ContentBuilder
│   ├── simple_regression.py     # Simple Regression Content (11 Kapitel)
│   └── multiple_regression.py   # Multiple Regression Content (9 Kapitel)
│
├── pipeline/                    # 🔧 DATA PROCESSING (4-Step Pipeline)
│   ├── get_data.py              # Step 1: GET
│   ├── calculate.py             # Step 2: CALCULATE
│   ├── plot.py                  # Step 3: PLOT
│   ├── display.py               # Step 4: DISPLAY (prepares data)
│   └── regression_pipeline.py   # Unified Pipeline
│
├── adapters/                    # 🎨 FRONTEND ADAPTERS
│   ├── detector.py              # Framework Auto-Detection
│   ├── base.py                  # BaseRenderer, RenderContext
│   ├── renderers/
│   │   ├── streamlit_renderer.py  # Interprets Content → st.*
│   │   └── html_renderer.py       # Interprets Content → HTML
│   ├── streamlit/
│   │   └── app.py               # Streamlit Application
│   ├── flask_app.py             # Flask Application
│   └── templates/               # Jinja2 Templates for Flask
│
├── config/                      # ⚙️ Configuration
│   └── config.py, logger.py
│
└── data/                        # 📊 Data definitions
    └── content.py               # Static content definitions
```

## 🚀 Quick Start

### Streamlit (Empfohlen für Interaktivität)
```bash
streamlit run run.py
```

### Flask (Web-Server)
```bash
python run.py --flask
# oder
FLASK_APP=run.py flask run
```

### Auto-Detection
```bash
python run.py  # Erkennt automatisch
```

## 📖 Educational Content

### Simple Regression (11 Kapitel)
1. Einleitung - Die Analyse von Zusammenhängen
2. Mehrdimensionale Verteilungen
3. Das Fundament - Das einfache lineare Regressionsmodell
4. Kovarianz & Korrelation
5. Die Methode - OLS-Schätzung
6. Das Regressionsmodell im Detail
7. Die Güteprüfung
8. Die Signifikanz
9. ANOVA für Gruppenvergleiche
10. Heteroskedastizität
11. Fazit und Ausblick

### Multiple Regression (9 Kapitel)
1. Einleitung - Multiple Regression
2. Das Multiple Regressionsmodell
3. OLS in Matrixform
4. Interpretation der Koeffizienten
5. Modellgüte - R² und F-Test
6. Multikollinearität
7. Dummy-Variablen
8. Residuendiagnostik
9. Prognose

## 🧪 Tests

```bash
pytest tests/ -v
```

## 💡 Wie es funktioniert

### 1. Content wird als Daten definiert
```python
from src.content import SimpleRegressionContent

# Content Builder nimmt nur Statistiken
builder = SimpleRegressionContent(stats_dict, plots_dict)
content = builder.build()

# content ist eine EducationalContent-Datenstruktur:
# - content.title
# - content.chapters[0].sections[0] → Markdown, Formula, Plot, Table, etc.
```

### 2. Renderer interpretiert die Daten

**Streamlit:**
```python
from src.adapters.renderers import StreamlitContentRenderer

renderer = StreamlitContentRenderer(plots=plots, data=data, stats=stats)
renderer.render(content)  # → st.markdown(), st.plotly_chart(), etc.
```

**Flask/HTML:**
```python
from src.adapters.renderers import HTMLContentRenderer

renderer = HTMLContentRenderer(plots=plots, data=data, stats=stats)
html = renderer.render(content)  # → HTML string
```

### 3. Beide Frontends zeigen identischen Content

Die Content-Struktur ist **exakt** dieselbe - nur die Darstellung ist unterschiedlich.

## 📊 Content-Elemente

| Element | Beschreibung | Streamlit | Flask |
|---------|--------------|-----------|-------|
| `Markdown` | Text | `st.markdown()` | `<div class="markdown">` |
| `Formula` | LaTeX | `st.latex()` | MathJax |
| `Plot` | Visualisierung | `st.plotly_chart()` | Plotly.js |
| `Table` | Tabelle | `st.dataframe()` | `<table>` |
| `Metric` | KPI | `st.metric()` | Custom Card |
| `Expander` | Aufklappbar | `st.expander()` | Bootstrap Accordion |
| `InfoBox` | Info | `st.info()` | Bootstrap Alert |
| `Columns` | Spalten | `st.columns()` | Bootstrap Grid |

## 🔧 Erweiterung

### Neues Frontend hinzufügen

1. Neuen Renderer erstellen:
```python
class TerminalContentRenderer:
    def render(self, content: EducationalContent) -> str:
        # Interpretiere Content als Terminal-Output
        pass
```

2. In Adapter integrieren - fertig!

### Neuen Content hinzufügen

1. Neuen ContentBuilder erstellen:
```python
class TimeSeriesContent(ContentBuilder):
    def build(self) -> EducationalContent:
        return EducationalContent(
            title="📈 Zeitreihenanalyse",
            chapters=[...]
        )
```

2. Beide Frontends zeigen es automatisch an!

## 📄 Lizenz

MIT License
