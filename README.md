# 📊 Linear Regression Guide & ML Bridge

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Architecture](https://img.shields.io/badge/Clean-Architecture-2ecc71?style=flat-square&logo=architecture&logoColor=white)](docs/ARCHITECTURE.md)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

An interactive, educational platform bridging the gap between **Statistical Analysis** and **Machine Learning**. Built with **Clean Architecture** principles to demonstrate enterprise-grade software design in Python.

## ✨ Key Features

### 📉 Statistical Analysis
- **Simple Linear Regression**: Interactive exploration of slope, intercept, and R².
- **Multiple Regression**: 3D visualization of regression planes (e.g., House Prices).
- **Diagnostics**: Residual plots, Q-Q plots, and heteroskedasticity checks.
- **Hypothesis Testing**: T-tests, F-statistics, and p-values explained.

### 🤖 Machine Learning Bridge (NEW)
- **Classification Algorithms**: Logistic Regression & K-Nearest Neighbors (KNN) built from scratch (NumPy-only).
- **3D Visualizations**:
    - **Loss Landscapes**: Visualize Gradient Descent on 3D error surfaces.
    - **Classification Surfaces**: 3D probability boundaries with colored error analysis.
    - **3D Confusion Matrices**: "Lego-style" visualization of model performance.
    - **3D ROC Curves**: Dynamic trade-off visualization (FPR vs TPR vs Threshold).
- **Educational Content**: Interactive chapters explaining the transition from OLS to Gradient Descent.
- **Data Explorer**: Inspect and download raw datasets (CSV) directly within the app.

### 🏗️ State-of-the-Art Architecture
- **Clean Architecture**: Strict separation of Domain, Application, and Infrastructure layers.
- **Pure Domain**: Business logic has ZERO external dependencies (no numpy/pandas in core).
- **Dependency Injection**: Decoupled components wired via a DI container.
- **Type Safety**: Strictly typed with Python 3.9+ type hints and Pydantic validation.
- **Structured Error Logging**: Comprehensive error tracking with unique error IDs, context, and stack traces.
- **Validation Testing**: All calculations validated against scikit-learn and R (rpy2) for accuracy.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Pip / Venv

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lhagenmayer/linear-regression-guide.git
   cd linear-regression-guide
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

**Option 1: Flask Web App (The Full Experience)**
```bash
python run.py --flask
# Opens http://localhost:5000
```

**Option 2: Streamlit Dashboard (Interactive Exploration)**
```bash
python run.py --streamlit
# Opens http://localhost:8501
```

**Option 3: REST API Server**
```bash
python run.py --api
# API available at http://localhost:8000
```

## 🧪 Testing

The project includes comprehensive test coverage:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run validation tests (requires scikit-learn and rpy2)
pytest tests/unit/infrastructure/test_regression_validation_sklearn.py -v
pytest tests/unit/ai/test_r_output_validation.py -v
```

**Test Coverage:**
- ✅ Unit tests for all services and components
- ✅ Integration tests for full pipeline flows
- ✅ Boundary tests (n=2, constant features, extreme values)
- ✅ Validation tests against scikit-learn (regression calculations)
- ✅ Validation tests against R (rpy2) for R-style output formatting
- ✅ Mocking for external API calls (Perplexity)

## 📚 Documentation

Detailed documentation is available in the `docs/` folder:

- **[Architecture Guide](docs/ARCHITECTURE.md)**: Deep dive into the Clean Architecture design.
- **[API Reference](docs/API.md)**: REST API endpoints and data schemas.
- **[Integration Guide](docs/INTEGRATION_GUIDE.md)**: How to integrate the frontend with the API.
- **[Deployment Guide](docs/DEPLOYMENT.md)**: Deployment instructions for various environments.

## 🛠️ Technology Stack

- **Core**: Python 3.11+, NumPy, SciPy
- **Architecture**: Domain-Driven Design (DDD), CQRS Patterns
- **API**: Flask, Pydantic (Validation)
- **Visualization**: Plotly (Interactive 3D Plots)
- **Frontend**: Flask (Jinja2 + Tailwind) & Streamlit
- **Testing**: pytest, scikit-learn (validation), rpy2 (R validation)
- **Logging**: Structured logging with error tracking and monitoring

## 🔍 Error Logging & Monitoring

The application includes comprehensive error logging:

- **Structured Logging**: All errors are logged with unique error IDs, context, and stack traces
- **Error Tracking**: Automatic error ID generation for tracking and debugging
- **Contextual Information**: Errors include request parameters, operation context, and service details
- **Log Files**: Separate log files for general logs, errors, and performance metrics
- **Log Rotation**: Automatic log rotation to prevent disk space issues

See `src/config/logger.py` for logging configuration and `docs/ARCHITECTURE.md` for details.

## 🤝 Contributing

Contributions are welcome! Please ensure you follow the **Clean Architecture** rules defined in `docs/ARCHITECTURE.md`.

1. **Domain Layer**: No external imports.
2. **Infrastructure Layer**: Implement interfaces defined in Domain.
3. **Tests**: Add unit tests for new logic.
4. **Error Handling**: Use structured error logging with `log_error_with_context()` or specialized functions.
5. **Validation**: Add validation tests against scikit-learn or R where applicable.

---

*Created for educational purposes by [Luca Hagenmayer].*
