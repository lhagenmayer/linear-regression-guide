# Linear Regression Guide

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)

Eine interaktive Web-App zum Erlernen linearer Regression. Gebaut mit Streamlit, plotly und statsmodels - für alle, die Regression verstehen wollen, ohne sich durch Formeln zu kämpfen.

**Warum diese App?**
Regression ist ein wichtiges statistisches Werkzeug, aber die Theorie kann überwältigend sein. Diese App macht Regression greifbar: Spiele mit Daten herum, sieh live, wie Modelle funktionieren, und verstehe die Konzepte visuell. Perfekt für Studierende, Datenanalysten und alle, die Regression anwenden wollen.

## Was kann die App?

**Interaktive Visualisierungen:**
- Scatterplots mit Regressionslinien
- 3D-Oberflächen für multiple Regression
- Residuenplots und Diagnostik
- Live-Updates bei Parameteränderungen

**Verschiedene Datensätze:**
- Simulierte Daten (Elektronikmarkt, Häuser, Städte)
- Echte Schweizer Daten (Kantone, Wetterstationen)
- Vollständig offline - keine API-Abhängigkeiten

**Lernpfad:**
- Grundlagen der linearen Regression
- Multiple Regression mit mehreren Prädiktoren
- Modellinterpretation und Diagnostik
- Statistische Tests und Hypothesen

**Einfach zu bedienen:**
- Navigation mit Tabs
- Anpassbare Parameter
- Klare Erklärungen
- Reagiert schnell

## Los geht's

**Voraussetzungen:**
- Python 3.9 oder neuer
- Ein virtuelles Environment (empfohlen)

**Installation:**
```bash
# Repository klonen
git clone <repository-url>
cd linear-regression-guide

# Abhängigkeiten installieren
pip install -r requirements.txt

# App starten
streamlit run app.py
```

Die App öffnet sich automatisch im Browser. Wenn nicht, gehe zu `http://localhost:8501`.

**Erste Schritte:**
1. Wähle ein Kapitel in der Sidebar
2. Spiele mit den Parametern herum
3. Beobachte, wie sich die Plots ändern
4. Lies die Erklärungen zu den statistischen Konzepten

## Entwicklung

Falls du den Code ändern möchtest:

```bash
# Zusätzliche Tools installieren
pip install -r requirements-dev.txt

# Automatische Code-Prüfung einrichten
pre-commit install

# Code formatieren
black *.py tests/*.py

# Tests laufen lassen
pytest tests/
```

## Tests

Es gibt Tests, um sicherzustellen, dass alles funktioniert.

```bash
# Tests laufen lassen
pytest tests/
```

Mehr Details in [TESTING.md](TESTING.md).

## Dateien

- `app.py` - Haupt-App
- `data.py` - Datenfunktionen
- `plots.py` - Diagramme
- `content.py` - Texte und Formeln
- `config.py` - Einstellungen
- `tests/` - Tests
- `requirements.txt` - Abhängigkeiten

## Wie benutzt man die App?

1. **Kapitel wählen:** In der Sidebar ein Thema auswählen
2. **Parameter anpassen:** Spiele mit Stichprobengröße, Rauschen und Seeds
3. **Visualisierungen beobachten:** Siehe live, wie sich Modelle ändern
4. **Erklärungen lesen:** Verstehe die statistischen Konzepte

**Tipp:** Verwende verschiedene Seeds, um zu sehen, wie zufällige Variationen die Ergebnisse beeinflussen.

## Technisches

Die App nutzt:
- Streamlit für die Web-Oberfläche
- Plotly für Diagramme
- Statsmodels für statistische Berechnungen
- Caching für bessere Performance

## Änderungen

Falls du etwas ändern möchtest, schau dir [DEVELOPMENT.md](DEVELOPMENT.md) an.

## Lizenz

MIT License - siehe [LICENSE](LICENSE) für Details. Frei verwendbar für Bildung und Forschung.
