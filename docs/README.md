# Linear Regression Guide - Documentation

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)

## 📚 Documentation Overview

Welcome to the Linear Regression Guide documentation! This project provides an interactive web application for learning linear regression concepts through visualization and experimentation.

### 🚀 Quick Access

| Getting Started | Development | Specialized |
|-----------------|-------------|-------------|
| **[INDEX.md](INDEX.md)** - Documentation overview | **[DEVELOPMENT.md](DEVELOPMENT.md)** - Development guide, testing, contributions | **[ACCESSIBILITY.md](ACCESSIBILITY.md)** - Accessibility features |
| **[README.md](../README.md)** - Project overview | | **[LOGGING.md](LOGGING.md)** - Logging system |

Eine interaktive Web-App zum Erlernen linearer Regression. Gebaut mit Streamlit, plotly und statsmodels - für alle, die Regression verstehen wollen, ohne sich durch Formeln zu kämpfen.

## 🚀 Quick Start

```bash
# Repository klonen
git clone <repository-url>
cd linear-regression-guide

# Abhängigkeiten installieren
pip install -r requirements.txt

# App starten
streamlit run run.py
```

## 📚 Features

### Interaktive Visualisierungen
- Scatterplots mit Regressionslinien
- 3D-Oberflächen für multiple Regression
- Residuenplots und Diagnostik
- Live-Updates bei Parameteränderungen

### Datensätze
- Simulierte Daten (Elektronikmarkt, Häuser, Städte)
- Echte Schweizer Daten (Kantone, Wetterstationen)
- Vollständig offline - keine API-Abhängigkeiten

### Lernpfad
- Grundlagen der linearen Regression
- Multiple Regression mit mehreren Prädiktoren
- Modellinterpretation und Diagnostik
- Statistische Tests und Hypothesen

## 🛠️ Development

### Setup
```bash
# Development environment setup
./scripts/setup_dev.sh

# Or manually:
pip install -r requirements-dev.txt
pre-commit install
```

### Code Quality
```bash
# Check code quality
./scripts/verify_code_quality.sh

# Auto-fix formatting
./scripts/verify_code_quality.sh --fix
```

### Testing
```bash
# Run all tests
./scripts/run_tests.sh

# Run specific test suites
./scripts/run_tests.sh --unit
./scripts/run_tests.sh --integration
```

### Deployment
```bash
# Prepare for deployment
./scripts/prepare_deployment.sh --deploy

# Deploy to Streamlit Cloud
# 1. Go to https://share.streamlit.io
# 2. Connect your GitHub repository
# 3. Set main file path to: run.py
# 4. Deploy!
```

## 📖 Documentation

- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Development guide, code quality, and contribution guidelines
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Development guide, testing, and contribution guidelines
- **[ACCESSIBILITY.md](ACCESSIBILITY.md)** - Accessibility features and implementation
- **[LOGGING.md](LOGGING.md)** - Logging system configuration and usage

## 🏗️ Project Structure

```
linear-regression-guide/
├── run.py                    # Main entry point
├── src/                      # Source code
│   ├── app.py               # Streamlit application
│   ├── config.py            # Configuration constants
│   ├── data.py              # Data generation functions
│   ├── plots.py             # Plotting functions
│   ├── content.py           # Content and text
│   ├── logger.py            # Logging utilities
│   └── accessibility.py     # Accessibility helpers
├── tests/                   # Test suite
├── scripts/                 # Development scripts
├── docs/                    # Documentation
└── requirements.txt         # Dependencies
```

## 🤝 Contributing

We welcome contributions! Please see [DEVELOPMENT.md](DEVELOPMENT.md) for detailed contribution guidelines.

## 📄 License

MIT License - see [LICENSE](../LICENSE) for details. Free to use for education and research.
