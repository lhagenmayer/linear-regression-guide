# 🚀 Deployment Guide

This guide describes how to deploy the Regression Analysis Platform to various environments.

## ☁️ Streamlit Cloud

The easiest way to deploy the interactive UI.

1.  Push your code to GitHub.
2.  Log in to [share.streamlit.io](https://share.streamlit.io).
3.  Click **"New app"**.
4.  Select your repository, branch (`master`), and main file (`run.py`).
5.  Click **"Deploy"**.

**Configuration:**
Streamlit Cloud automatically installs dependencies from `requirements.txt`.

## 🐳 Docker Deployment

You can containerize the application to run anywhere (AWS, GCP, Azure).

### Dockerfile
Create a `Dockerfile` in the root directory:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default to API mode
EXPOSE 8000
CMD ["python", "run.py", "--api", "--host", "0.0.0.0", "--port", "8000"]
```

### Build & Run

```bash
# Build image
docker build -t regression-app .

# Run API
docker run -p 8000:8000 regression-app

# Run Streamlit (Override command)
docker run -p 8501:8501 regression-app streamlit run run.py
```

## 🖥️ Traditional Server (Ubuntu/Debian)

### 1. System Setup
```bash
sudo apt update
sudo apt install python3-pip python3-venv git
```

### 2. Application Setup
```bash
git clone https://github.com/lhagenmayer/linear-regression-guide.git
cd linear-regression-guide
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Production Server (Gunicorn)
For production, use `gunicorn` instead of the development server.

**For REST API:**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 "run:create_api_app()"
```

**For Flask App:**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 "run:create_app()"
```

## 🌐 Environment Variables

| Variable | Description |
|----------|-------------|
| `PORT` | Port to bind (default: 8000/5000/8501) |
| `PERPLEXITY_API_KEY` | Optional. API Key for AI features. |
| `RUN_API` | Set to `true` to force API mode. |
| `FLASK_APP` | Set to `true` to force Flask mode. |

