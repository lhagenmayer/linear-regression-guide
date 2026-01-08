"""
API Server - Reiner REST-API-Server.

Erstellt einen minimalen REST-API-Server, der von JEDEM Frontend genutzt werden kann.
Unterstützt sowohl Flask als auch FastAPI (falls installiert).

Dies ist ein REINER API-Server - keine HTML-Templates, kein UI-Rendering.
Alle Antworten sind im JSON-Format.

API-Dokumentation:
- /api/docs - Interaktive Swagger UI
- /api/openapi.json - OpenAPI 3.0 Spezifikation (JSON)
- /api/openapi.yaml - OpenAPI 3.0 Spezifikation (YAML)
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from .endpoints import UnifiedAPI, RegressionAPI, ContentAPI, AIInterpretationAPI

logger = logging.getLogger(__name__)

# Pfad zum docs-Verzeichnis für die OpenAPI-Spezifikation
DOCS_DIR = Path(__file__).parent.parent.parent / "docs"


def _get_full_openapi_spec() -> Dict[str, Any]:
    """Lädt die vollständige OpenAPI-Spezifikation aus der YAML-Datei."""
    yaml_path = DOCS_DIR / "openapi.yaml"
    
    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except ImportError:
            # PyYAML nicht installiert, Fallback auf generierte Spec
            pass
        except Exception as e:
            logger.warning(f"Konnte OpenAPI YAML nicht laden: {e}")
    
    # Fallback auf programmatisch generierte Spezifikation
    return UnifiedAPI().get_openapi_spec()


def _get_swagger_ui_html(openapi_url: str = "/api/openapi.json") -> str:
    """Generiert das HTML für die interaktive Swagger-Dokumentation."""
    return f'''<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Regression Analysis API - Dokumentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
    <style>
        html {{ box-sizing: border-box; overflow-y: scroll; }}
        *, *:before, *:after {{ box-sizing: inherit; }}
        body {{ margin: 0; background: #fafafa; }}
        .swagger-ui .topbar {{ display: none; }}
        .swagger-ui .info hgroup.main h2 {{ font-size: 1.5em; }}
        .custom-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .custom-header h1 {{ margin: 0; font-size: 2em; }}
        .custom-header p {{ margin: 10px 0 0; opacity: 0.9; }}
        .custom-header a {{ color: white; text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="custom-header">
        <h1>📊 Regression Analysis API</h1>
        <p>100% Plattform-Agnostische REST API für Statistische Analysen</p>
        <p style="font-size: 0.9em; margin-top: 15px;">
            <a href="/api/openapi.json">OpenAPI JSON</a> |
            <a href="/api/openapi.yaml">OpenAPI YAML</a> |
            <a href="/">← Zurück zur App</a>
        </p>
    </div>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {{
            window.ui = SwaggerUIBundle({{
                url: "{openapi_url}",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                defaultModelsExpandDepth: 2,
                defaultModelExpandDepth: 2,
                docExpansion: "list",
                filter: true,
                showExtensions: true,
                showCommonExtensions: true,
                tryItOutEnabled: true,
            }});
        }};
    </script>
</body>
</html>'''


def _get_redoc_html(openapi_url: str = "/api/openapi.json") -> str:
    """Generiert ReDoc HTML für alternative API-Dokumentation."""
    return f'''<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Regression Analysis API - ReDoc</title>
    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
    <style>
        body {{ margin: 0; padding: 0; }}
    </style>
</head>
<body>
    <redoc spec-url="{openapi_url}"></redoc>
    <script src="https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js"></script>
</body>
</html>'''


def create_api_server(framework: str = "auto", cors_origins: list = None):
    """
    Erstellt einen REST-API-Server.
    
    Wählt automatisch zwischen Flask und FastAPI, basierend auf der Verfügbarkeit.
    """
    if cors_origins is None:
        cors_origins = ["*"] # Standardmäßig alles erlauben
    
    if framework == "auto":
        try:
            import fastapi
            framework = "fastapi"
        except ImportError:
            framework = "flask"
    
    logger.info(f"Erstelle API-Server mit Framework: {framework}")
    
    if framework == "fastapi":
        return _create_fastapi_server(cors_origins)
    else:
        return _create_flask_server(cors_origins)


def _create_flask_server(cors_origins: list):
    """Erstellt den API-Server basierend auf Flask."""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    # Manuelle CORS-Unterstützung (Cross-Origin Resource Sharing)
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
        """Health-Check-Endpunkt."""
        return jsonify({"status": "ok", "framework": "flask"})
    
    # Initialisierung der framework-agnostischen APIs
    api = UnifiedAPI()
    
    # =========================================================================
    # Regressions-Endpunkte
    # =========================================================================
    
    @app.route('/api/regression/simple', methods=['POST', 'OPTIONS'])
    def regression_simple():
        """Führt eine einfache Regression über POST aus."""
        if request.method == 'OPTIONS':
            return '', 204
        data = request.get_json() or {}
        return jsonify(api.regression.run_simple(**data))
    
    @app.route('/api/regression/multiple', methods=['POST', 'OPTIONS'])
    def regression_multiple():
        """Führt eine multiple Regression über POST aus."""
        if request.method == 'OPTIONS':
            return '', 204
        data = request.get_json() or {}
        return jsonify(api.regression.run_multiple(**data))
    
    @app.route('/api/datasets', methods=['GET'])
    def datasets():
        """Gibt die Liste der verfügbaren Datensätze zurück."""
        return jsonify(api.regression.get_datasets())
    
    # =========================================================================
    # Inhalts-Endpunkte (Educational Content)
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
    # KI-Endpunkte (AI Interpretation)
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
    # Dokumentations-Endpunkte
    # =========================================================================
    
    @app.route('/api/openapi.json', methods=['GET'])
    def openapi_json():
        """Gibt die OpenAPI-Spezifikation im JSON-Format zurück."""
        return jsonify(_get_full_openapi_spec())
    
    @app.route('/api/openapi.yaml', methods=['GET'])
    def openapi_yaml():
        """Gibt die OpenAPI-Spezifikation im YAML-Format zurück."""
        from flask import Response
        yaml_path = DOCS_DIR / "openapi.yaml"
        
        if yaml_path.exists():
            with open(yaml_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return Response(content, mimetype='text/yaml')
        
        # Fallback: Generiere YAML aus dem JSON-Objekt
        try:
            import yaml
            spec = _get_full_openapi_spec()
            return Response(yaml.dump(spec, allow_unicode=True), mimetype='text/yaml')
        except ImportError:
            return Response("# PyYAML nicht installiert\n# Nutze /api/openapi.json", mimetype='text/yaml')
    
    @app.route('/api/docs', methods=['GET'])
    def swagger_ui():
        """Hostet die interaktive Swagger UI Dokumentation."""
        from flask import Response
        return Response(_get_swagger_ui_html(), mimetype='text/html')
    
    @app.route('/api/redoc', methods=['GET'])
    def redoc():
        """Hostet die alternative ReDoc Dokumentation."""
        from flask import Response
        return Response(_get_redoc_html(), mimetype='text/html')
    
    return app


def _create_fastapi_server(cors_origins: list):
    """Erstellt den API-Server basierend auf FastAPI."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
    except ImportError:
        raise ImportError("FastAPI nicht installiert. Installiere mit: pip install fastapi uvicorn")
    
    app = FastAPI(
        title="Regression Analysis API",
        description="Plattform-agnostische REST API für Regressionsanalysen",
        version="1.0.0",
    )
    
    # Konfiguration der CORS-Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Framework-agnostische API-Logik initialisieren
    api = UnifiedAPI()
    
    @app.get("/api/health")
    async def health():
        return {"status": "ok", "framework": "fastapi"}
    
    # =========================================================================
    # Regressions-Endpunkte
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
    # Inhalts-Endpunkte
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
    # KI-Endpunkte
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
    # Dokumentations-Endpunkte
    # =========================================================================
    
    @app.get("/api/openapi.json")
    async def openapi_json():
        """Gibt die Spec im JSON-Format zurück."""
        return _get_full_openapi_spec()
    
    @app.get("/api/openapi.yaml")
    async def openapi_yaml():
        """Gibt die Spec im YAML-Format zurück."""
        from fastapi.responses import PlainTextResponse
        yaml_path = DOCS_DIR / "openapi.yaml"
        
        if yaml_path.exists():
            with open(yaml_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return PlainTextResponse(content, media_type='text/yaml')
        
        # Fallback auf On-the-fly Generierung
        try:
            import yaml
            spec = _get_full_openapi_spec()
            return PlainTextResponse(yaml.dump(spec, allow_unicode=True), media_type='text/yaml')
        except ImportError:
            return PlainTextResponse("# PyYAML nicht installiert", media_type='text/yaml')
    
    @app.get("/api/docs")
    async def swagger_ui():
        """Interaktive Swagger-Doku."""
        from fastapi.responses import HTMLResponse
        return HTMLResponse(_get_swagger_ui_html())
    
    @app.get("/api/redoc")
    async def redoc():
        """Alternative ReDoc-Doku."""
        from fastapi.responses import HTMLResponse
        return HTMLResponse(_get_redoc_html())
    
    return app


# =========================================================================
# Standalone-Runner (CLI)
# =========================================================================

def run_api_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    framework: str = "auto",
    cors_origins: list = None,
    debug: bool = False,
):
    """
    Startet den API-Server als eigenständigen Prozess.
    """
    if cors_origins is None:
        cors_origins = ["*"]
    
    app = create_api_server(framework, cors_origins)
    
    print()
    print("=" * 60)
    print("📊 REGRESSION ANALYSIS API")
    print("=" * 60)
    
    if hasattr(app, 'run'):
        # Flask-spezifischer Start
        print(f"🚀 Server:      http://{host}:{port}")
        print(f"📖 Swagger UI:  http://{host}:{port}/api/docs")
        print(f"📄 OpenAPI:     http://{host}:{port}/api/openapi.json")
        print("=" * 60)
        print()
        app.run(host=host, port=port, debug=debug)
    else:
        # FastAPI-spezifischer Start via Uvicorn
        import uvicorn
        print(f"🚀 Server:      http://{host}:{port}")
        print(f"📖 Swagger UI:  http://{host}:{port}/api/docs")
        print(f"📄 OpenAPI:     http://{host}:{port}/api/openapi.json")
        print("=" * 60)
        print()
        uvicorn.run(app, host=host, port=port, log_level="info" if not debug else "debug")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Starte die Regression Analysis API")
    parser.add_argument("--host", default="0.0.0.0", help="Host-Adresse")
    parser.add_argument("--port", type=int, default=8000, help="Port-Nummer")
    parser.add_argument("--framework", choices=["auto", "flask", "fastapi"], default="auto")
    parser.add_argument("--debug", action="store_true", help="Debug-Modus aktivieren")
    
    args = parser.parse_args()
    run_api_server(args.host, args.port, args.framework, debug=args.debug)
