# 📊 Regression Analysis

**Interactive Statistical Learning Platform**

Eine moderne, interaktive Lernplattform für Regressionsanalyse mit **Frontend-Agnostischer Architektur** - läuft identisch in Streamlit und Flask.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Tests](https://img.shields.io/badge/Tests-26%20passed-success.svg)

---

## ✨ Features

### 📈 Einfache Regression (11 Kapitel)
- Mehrdimensionale Verteilungen & bivariate Normalverteilung
- Kovarianz, Korrelation (Pearson & Spearman)
- OLS-Schätzung mit Residuenanalyse
- Gauss-Markov Annahmen & Diagnostik
- t-Tests, F-Tests, ANOVA
- Heteroskedastizität & robuste Standardfehler
- Interaktive 3D-Visualisierungen

### 📊 Multiple Regression (9 Kapitel)
- OLS in Matrixnotation
- Partielle vs. totale Effekte
- Multikollinearität & VIF
- Dummy-Variablen
- Residuendiagnostik
- 3D-Regressionsebene
- Interaktive Prognose

### 🎨 State-of-the-Art UI (Flask)
- 🌙 Dark/Light Mode mit Tastenkürzel (D)
- ⚡ HTMX für dynamische Updates ohne Reload
- 📱 Responsive Design mit Mobile-Sidebar
- 🎯 Scroll-Spy Navigation
- 📋 Copy-to-Clipboard für Code
- 🖨️ Print-optimierte Styles

---

## 🚀 Quick Start

### Installation

```bash
# Repository klonen
git clone <repository-url>
cd regression-analysis

# Dependencies installieren
pip install -r requirements.txt
```

### Ausführung

```bash
# Streamlit (interaktiv, empfohlen für Lernen)
streamlit run run.py

# Flask (Web-Server, state-of-the-art UI)
python run.py --flask

# Auto-Detection
python run.py
```

### URLs

| Framework | URL |
|-----------|-----|
| Streamlit | http://localhost:8501 |
| Flask | http://localhost:5000 |

---

## 🏗️ Architektur

### Option B: Content als Datenstruktur

```
┌─────────────────────────────────────────────────────────────────┐
│                         run.py                                   │
│                    (Auto-Detection)                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ↓                                 ↓
┌─────────────────────┐             ┌─────────────────────┐
│   Streamlit App     │             │     Flask App       │
└─────────────────────┘             └─────────────────────┘
            │                                 │
            └────────────────┬────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                  ContentBuilder (content/)                       │
│                                                                  │
│   SimpleRegressionContent    MultipleRegressionContent          │
│   → 11 Kapitel               → 9 Kapitel                        │
│   → Dynamischer Content      → Dynamischer Content              │
│   → KEINE UI-Abhängigkeiten  → KEINE UI-Abhängigkeiten         │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                ↓                                 ↓
┌───────────────────────────┐     ┌───────────────────────────┐
│  StreamlitContentRenderer │     │    HTMLContentRenderer    │
│     → st.markdown()       │     │     → HTML/Jinja2         │
│     → st.plotly_chart()   │     │     → Bootstrap           │
│     → st.expander()       │     │     → Plotly.js           │
└───────────────────────────┘     └───────────────────────────┘
```

### Warum Option B?

| Aspekt | Option A (vorher) | Option B (jetzt) |
|--------|-------------------|------------------|
| Content | Dupliziert pro Framework | **Einmal definiert** |
| Änderungen | 2x durchführen | **1x durchführen** |
| Konsistenz | Risiko für Divergenz | **Garantiert identisch** |
| Erweiterung | Neues Framework = Copy-Paste | **Neuer Renderer = fertig** |

---

## 📁 Projektstruktur

```
regression-analysis/
├── run.py                      # 🚀 Unified Entry Point
│
├── src/
│   ├── content/                # 📖 EDUCATIONAL CONTENT (Framework-Agnostic)
│   │   ├── structure.py        #    Content-Datenstrukturen
│   │   ├── builder.py          #    Base ContentBuilder
│   │   ├── simple_regression.py #   11 Kapitel Simple Regression
│   │   └── multiple_regression.py # 9 Kapitel Multiple Regression
│   │
│   ├── pipeline/               # 🔧 4-STEP DATA PIPELINE
│   │   ├── get_data.py         #    Step 1: GET
│   │   ├── calculate.py        #    Step 2: CALCULATE
│   │   ├── plot.py             #    Step 3: PLOT
│   │   ├── display.py          #    Step 4: DISPLAY
│   │   └── regression_pipeline.py # Unified Pipeline
│   │
│   ├── adapters/               # 🎨 FRONTEND ADAPTERS
│   │   ├── detector.py         #    Framework Auto-Detection
│   │   ├── base.py             #    BaseRenderer, RenderContext
│   │   ├── renderers/
│   │   │   ├── streamlit_renderer.py
│   │   │   └── html_renderer.py
│   │   ├── streamlit/
│   │   │   └── app.py          #    Streamlit Application
│   │   ├── flask_app.py        #    Flask Application
│   │   └── templates/          #    Jinja2 Templates
│   │       ├── base.html
│   │       ├── index.html
│   │       └── educational_content.html
│   │
│   ├── config/                 # ⚙️ Configuration
│   │   ├── config.py
│   │   └── logger.py
│   │
│   └── data/                   # 📊 Static Content
│       └── content.py
│
├── tests/                      # 🧪 Tests
│   └── unit/
│       └── test_pipeline.py    #    26 Unit Tests
│
├── requirements.txt
└── README.md
```

---

## 📖 Content-Elemente

Der ContentBuilder verwendet diese Datenstrukturen:

| Element | Beschreibung | Streamlit | Flask |
|---------|--------------|-----------|-------|
| `Markdown` | Text mit Formatierung | `st.markdown()` | HTML |
| `Formula` | LaTeX Formeln | `st.latex()` | MathJax |
| `Plot` | Visualisierungen | `st.plotly_chart()` | Plotly.js |
| `Table` | Datentabellen | `st.dataframe()` | `<table>` |
| `Metric` | KPI-Anzeige | `st.metric()` | Card |
| `MetricRow` | Mehrere KPIs | `st.columns()` | Grid |
| `Expander` | Aufklappbar | `st.expander()` | Accordion |
| `Columns` | Spalten-Layout | `st.columns()` | Bootstrap Row |
| `InfoBox` | Info-Hinweis | `st.info()` | Alert Info |
| `WarningBox` | Warnung | `st.warning()` | Alert Warning |
| `SuccessBox` | Erfolg | `st.success()` | Alert Success |
| `CodeBlock` | Code | `st.code()` | `<pre><code>` |

---

## 🔧 Dynamischer Content

Der Content passt sich automatisch dem Datensatz an:

```python
# Datensatz wählen
stats = {
    'context_title': 'Bildung und Einkommen',
    'x_label': 'Bildungsjahre',
    'y_label': 'Jahreseinkommen (CHF)',
    'slope': 5000.0,
    'intercept': 20000.0,
    # ... weitere Statistiken
}

# Content generieren
builder = SimpleRegressionContent(stats, plots)
content = builder.build()

# Rendern (Streamlit ODER Flask)
renderer = StreamlitContentRenderer(stats=stats)
renderer.render(content)
```

**Ergebnis:**
- Alle Labels, Interpretationen, Formeln sind datensatz-spezifisch
- R-Style Output zeigt korrekte Variablennamen
- Beispielrechnungen verwenden echte Werte

---

## 🧪 Tests

```bash
# Alle Tests ausführen
pytest tests/ -v

# Mit Coverage
pytest tests/ --cov=src --cov-report=html
```

**Aktueller Status:** 26 Tests ✅

---

## 🎨 Flask UI Features

### Dark Mode
- Toggle: Button unten rechts oder Taste **D**
- Speicherung in localStorage
- Plotly-Plots passen sich an

### HTMX
- Dataset-Wechsel ohne Reload
- Slider-Updates in Echtzeit
- Loading-Indicator

### Navigation
- Sticky Sidebar mit Kapitel-Links
- Scroll-Spy für aktives Kapitel
- Mobile-optimiert mit Toggle

---

## 🔄 Erweiterung

### Neues Frontend hinzufügen

```python
# 1. Neuen Renderer erstellen
class TerminalContentRenderer:
    def render(self, content: EducationalContent) -> str:
        for chapter in content.chapters:
            print(f"\n=== {chapter.title} ===")
            for section in chapter.sections:
                self._render_element(section)

# 2. Fertig! Derselbe Content wird angezeigt.
```

### Neuen Content hinzufügen

```python
# 1. Neuen ContentBuilder erstellen
class TimeSeriesContent(ContentBuilder):
    def build(self) -> EducationalContent:
        return EducationalContent(
            title="📈 Zeitreihenanalyse",
            subtitle="ARIMA, Saisonalität und mehr",
            chapters=[
                self._chapter_1_introduction(),
                # ...
            ]
        )

# 2. Alle Renderer zeigen es automatisch an!
```

---

## 📋 Requirements

```
flask>=3.0.0
streamlit>=1.28.0
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0
plotly>=5.18.0
```

---

## 📄 Lizenz

MIT License - siehe [LICENSE](LICENSE)

---

## 🙏 Credits

- **Bootstrap 5.3** - UI Framework mit Dark Mode
- **Plotly** - Interaktive Visualisierungen
- **MathJax** - LaTeX Rendering
- **HTMX** - Dynamic HTML
- **Alpine.js** - Reaktivität
