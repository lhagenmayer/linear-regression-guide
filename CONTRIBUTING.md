# Contributing to Regression Analysis Platform

Thank you for your interest in contributing! This project is a **100% platform-agnostic** regression analysis tool designed for educational purposes.

## 🛠️ Development Setup

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone & Install
```bash
git clone https://github.com/lhagenmayer/linear-regression-guide.git
cd linear-regression-guide

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Locally

You can run the application in 3 modes:

**Mode A: Streamlit (Interactive UI)** -> *Best for development*
```bash
streamlit run run.py
```

**Mode B: API Server (Backend Logic)** -> *Best for testing API integrations*
```bash
python3 run.py --api
# Access Docs at http://localhost:8000/api/docs
```

**Mode C: Flask App (Validation)**
```bash
python3 run.py --flask
```

## 🏗️ Architecture Guidelines

This project follows strict architectural rules to maintain platform agnosticism:

1.  **Core Isolation**: Code in `src/pipeline`, `src/content`, and `src/ai` MUST NOT import any framework (Streamlit, Flask, etc.).
2.  **Pure Data**: All outputs from the core layer must be JSON-serializable (Dictionaries, Lists, Primitives).
3.  **Adapters**: Framework-specific code belongs ONLY in `src/adapters/`.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## 🧪 Testing

We use `pytest` for testing. Ensure all tests pass before submitting a PR.

```bash
# Run all tests
pytest tests/
```

## 📝 Pull Request Process

1.  Create a new branch: `git checkout -b feature/amazing-feature`.
2.  Commit your changes using conventional commits (e.g., `feat: add new dataset`, `fix: calculation error`).
3.  Push to the branch: `git push origin feature/amazing-feature`.
4.  Open a Pull Request and describe your changes.

## 🤝 Code Style

- Use **type hints** for all function arguments and return values.
- Follow **PEP 8** guidelines.
- Keep functions small and focused (Single Responsibility Principle).

Happy Coding! 🚀
