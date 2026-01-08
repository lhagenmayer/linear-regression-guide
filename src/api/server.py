"""
API Server - Pure REST API Server.

Creates a minimal REST API server that can be used by ANY frontend.
Supports both Flask and FastAPI (if available).

This is a PURE API server - no HTML templates, no UI rendering.
All responses are JSON.
"""

import json
import logging
from typing import Dict, Any, Optional

from .endpoints import UnifiedAPI, RegressionAPI, ContentAPI, AIInterpretationAPI

logger = logging.getLogger(__name__)


def create_api_server(framework: str = "auto", cors_origins: list = None):
    """
    Create a REST API server.
    
    Args:
        framework: "flask", "fastapi", or "auto" (detect)
        cors_origins: List of allowed CORS origins (e.g., ["http://localhost:3000"])
        
    Returns:
        Flask app or FastAPI app
    """
    if cors_origins is None:
        cors_origins = ["*"]  # Allow all by default
    
    if framework == "auto":
        try:
            import fastapi
            framework = "fastapi"
        except ImportError:
            framework = "flask"
    
    logger.info(f"Creating API server with framework: {framework}")
    
    if framework == "fastapi":
        return _create_fastapi_server(cors_origins)
    else:
        return _create_flask_server(cors_origins)


def _create_flask_server(cors_origins: list):
    """Create Flask-based API server."""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    # CORS support
    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get('Origin', '*')
        if '*' in cors_origins or origin in cors_origins:
            response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        return response
    
    @app.route('/api/health', methods=['GET'])
    def health():
        return jsonify({"status": "ok", "framework": "flask"})
    
    # Initialize APIs
    api = UnifiedAPI()
    
    # =========================================================================
    # Regression Endpoints
    # =========================================================================
    
    @app.route('/api/regression/simple', methods=['POST', 'OPTIONS'])
    def regression_simple():
        if request.method == 'OPTIONS':
            return '', 204
        data = request.get_json() or {}
        return jsonify(api.regression.run_simple(**data))
    
    @app.route('/api/regression/multiple', methods=['POST', 'OPTIONS'])
    def regression_multiple():
        if request.method == 'OPTIONS':
            return '', 204
        data = request.get_json() or {}
        return jsonify(api.regression.run_multiple(**data))
    
    @app.route('/api/datasets', methods=['GET'])
    def datasets():
        return jsonify(api.regression.get_datasets())
    
    # =========================================================================
    # Content Endpoints
    # =========================================================================
    
    @app.route('/api/content/simple', methods=['POST', 'OPTIONS'])
    def content_simple():
        if request.method == 'OPTIONS':
            return '', 204
        data = request.get_json() or {}
        return jsonify(api.content.get_simple_content(**data))
    
    @app.route('/api/content/multiple', methods=['POST', 'OPTIONS'])
    def content_multiple():
        if request.method == 'OPTIONS':
            return '', 204
        data = request.get_json() or {}
        return jsonify(api.content.get_multiple_content(**data))
    
    @app.route('/api/content/schema', methods=['GET'])
    def content_schema():
        return jsonify(api.content.get_content_schema())
    
    # =========================================================================
    # AI Endpoints
    # =========================================================================
    
    @app.route('/api/ai/interpret', methods=['POST', 'OPTIONS'])
    def ai_interpret():
        if request.method == 'OPTIONS':
            return '', 204
        data = request.get_json() or {}
        return jsonify(api.ai.interpret(**data))
    
    @app.route('/api/ai/r-output', methods=['POST', 'OPTIONS'])
    def ai_r_output():
        if request.method == 'OPTIONS':
            return '', 204
        data = request.get_json() or {}
        return jsonify(api.ai.get_r_output(data.get('stats', {})))
    
    @app.route('/api/ai/status', methods=['GET'])
    def ai_status():
        return jsonify(api.ai.get_status())
    
    @app.route('/api/ai/cache/clear', methods=['POST'])
    def ai_cache_clear():
        return jsonify(api.ai.clear_cache())
    
    # =========================================================================
    # OpenAPI
    # =========================================================================
    
    @app.route('/api/openapi.json', methods=['GET'])
    def openapi():
        return jsonify(api.get_openapi_spec())
    
    return app


def _create_fastapi_server(cors_origins: list):
    """Create FastAPI-based API server."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
    except ImportError:
        raise ImportError("FastAPI not installed. Install with: pip install fastapi uvicorn")
    
    app = FastAPI(
        title="Regression Analysis API",
        description="Platform-agnostic API for regression analysis",
        version="1.0.0",
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize APIs
    api = UnifiedAPI()
    
    @app.get("/api/health")
    async def health():
        return {"status": "ok", "framework": "fastapi"}
    
    # =========================================================================
    # Regression Endpoints
    # =========================================================================
    
    @app.post("/api/regression/simple")
    async def regression_simple(request: Request):
        data = await request.json() if await request.body() else {}
        return api.regression.run_simple(**data)
    
    @app.post("/api/regression/multiple")
    async def regression_multiple(request: Request):
        data = await request.json() if await request.body() else {}
        return api.regression.run_multiple(**data)
    
    @app.get("/api/datasets")
    async def datasets():
        return api.regression.get_datasets()
    
    # =========================================================================
    # Content Endpoints
    # =========================================================================
    
    @app.post("/api/content/simple")
    async def content_simple(request: Request):
        data = await request.json() if await request.body() else {}
        return api.content.get_simple_content(**data)
    
    @app.post("/api/content/multiple")
    async def content_multiple(request: Request):
        data = await request.json() if await request.body() else {}
        return api.content.get_multiple_content(**data)
    
    @app.get("/api/content/schema")
    async def content_schema():
        return api.content.get_content_schema()
    
    # =========================================================================
    # AI Endpoints
    # =========================================================================
    
    @app.post("/api/ai/interpret")
    async def ai_interpret(request: Request):
        data = await request.json() if await request.body() else {}
        return api.ai.interpret(**data)
    
    @app.post("/api/ai/r-output")
    async def ai_r_output(request: Request):
        data = await request.json() if await request.body() else {}
        return api.ai.get_r_output(data.get('stats', {}))
    
    @app.get("/api/ai/status")
    async def ai_status():
        return api.ai.get_status()
    
    @app.post("/api/ai/cache/clear")
    async def ai_cache_clear():
        return api.ai.clear_cache()
    
    # =========================================================================
    # OpenAPI
    # =========================================================================
    
    @app.get("/api/openapi.json")
    async def openapi():
        return api.get_openapi_spec()
    
    return app


# =========================================================================
# Standalone Runner
# =========================================================================

def run_api_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    framework: str = "auto",
    cors_origins: list = None,
    debug: bool = False,
):
    """
    Run the API server standalone.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        framework: "flask", "fastapi", or "auto"
        cors_origins: CORS origins
        debug: Debug mode
    """
    if cors_origins is None:
        cors_origins = ["*"]
    
    app = create_api_server(framework, cors_origins)
    
    if hasattr(app, 'run'):
        # Flask
        print(f"🚀 Starting Flask API server on http://{host}:{port}")
        print(f"📚 OpenAPI spec: http://{host}:{port}/api/openapi.json")
        app.run(host=host, port=port, debug=debug)
    else:
        # FastAPI
        import uvicorn
        print(f"🚀 Starting FastAPI server on http://{host}:{port}")
        print(f"📚 OpenAPI spec: http://{host}:{port}/api/openapi.json")
        print(f"📖 Swagger UI: http://{host}:{port}/docs")
        uvicorn.run(app, host=host, port=port, log_level="info" if not debug else "debug")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Regression Analysis API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--framework", choices=["auto", "flask", "fastapi"], default="auto")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    run_api_server(args.host, args.port, args.framework, debug=args.debug)
