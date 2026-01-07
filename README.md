# 📊 Linear Regression Guide

Ein interaktives, didaktisches Tool für lineare Regressionsanalyse.

## 🎯 Architektur

Klare **4-Stufen-Pipeline**:

```
GET → CALCULATE → PLOT → DISPLAY
```

| Stufe | Modul | Funktion |
|-------|-------|----------|
| **GET** | `pipeline/get_data.py` | Daten generieren |
| **CALCULATE** | `pipeline/calculate.py` | Statistiken berechnen |
| **PLOT** | `pipeline/plot.py` | Visualisierungen erstellen |
| **DISPLAY** | `ui/tabs/*.py` | Edukativen Content rendern |

## 🚀 Schnellstart

```bash
# Dependencies installieren
pip install -r requirements.txt

# App starten
streamlit run src/app.py
```

## 📁 Projektstruktur

```
src/
├── app.py                    # Entry Point (Streamlit)
├── pipeline/                 # 4-Step Pipeline
│   ├── get_data.py          # Step 1: GET
│   ├── calculate.py         # Step 2: CALCULATE
│   ├── plot.py              # Step 3: PLOT
│   ├── display.py           # Step 4: DISPLAY (Adapter)
│   └── regression_pipeline.py  # Pipeline Orchestrator
├── ui/tabs/                  # Educational Content
│   ├── simple_regression_educational.py
│   └── multiple_regression_educational.py
├── data/
│   └── content.py           # Dynamic Content
└── config/                  # Configuration & Logging
```

## 🎓 Features

### Einfache Regression
- 11 Kapitel mit vollständigem edukativen Content
- Interaktive 3D-Visualisierungen
- LaTeX-Formeln
- R-Style Output
- Gauss-Markov Annahmen
- Heteroskedastizität & robuste SE

### Multiple Regression
- 9 Kapitel mit vollständigem Content
- 3D Regressionsebene
- VIF & Multikollinearität
- Dummy-Variablen Demo
- Interaktive Prognose

## 💻 Verwendung

```python
from src.pipeline import RegressionPipeline

# Pipeline initialisieren
pipeline = RegressionPipeline()

# Einfache Regression ausführen
result = pipeline.run_simple(
    dataset="electronics",
    n=100,
    seed=42
)

# Ergebnis enthält: data, stats, plots
print(f"R² = {result.stats.r_squared:.4f}")
```

## 🧪 Tests

```bash
pytest tests/ -v
```

## 📦 Dependencies

- `streamlit` - Web UI
- `plotly` - Interaktive Plots
- `numpy` - Numerische Berechnungen
- `pandas` - Datenstrukturen
- `scipy` - Statistische Funktionen

## 📄 Lizenz

MIT License
