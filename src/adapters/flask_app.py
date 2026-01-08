"""
Flask Application - 100% Plattform-Agnostisch.

Nutzt dieselbe API-Schicht wie externe Frontends (Next.js, Vite, etc.).
Dies stellt die Konsistenz über alle Benutzeroberflächen hinweg sicher.

Architektur-Fluss:
    Flask App → API Layer → Core Pipeline
    
    Identisch zu:
    Next.js App → HTTP → API Layer → Core Pipeline
"""

import json
from flask import Flask, render_template, request, jsonify
from typing import Dict, Any

from ..config import get_logger
from ..api import RegressionAPI, ContentAPI, AIInterpretationAPI
from .renderers import HTMLContentRenderer

logger = get_logger(__name__)


def create_flask_app() -> Flask:
    """Erstellt und konfiguriert die Flask-Anwendung für das HTML-Frontend."""
    app = Flask(__name__, template_folder='templates')
    
    # Initalisierung der APIs (dieselben, die auch ein externes Frontend nutzen würde)
    regression_api = RegressionAPI()
    content_api = ContentAPI()
    ai_api = AIInterpretationAPI()
    
    @app.route('/')
    def index():
        """Hauptseite (Landing Page)."""
        # Abruf der verfügbaren Datensätze über die API
        datasets = regression_api.get_datasets()
        return render_template(
            'index.html',
            datasets=datasets['data'],
            api_status=ai_api.get_status()['status']
        )
    
    @app.route('/simple')
    def simple_regression():
        """Seite für die einfache lineare Regression."""
        # Parameter aus dem Query-String extrahieren
        dataset = request.args.get('dataset', 'electronics')
        n_points = int(request.args.get('n', 50))
        noise = float(request.args.get('noise', 0.4))
        seed = int(request.args.get('seed', 42))
        
        # Aufruf der Content-API (wie ein modernes JS-Frontend)
        response = content_api.get_simple_content(
            dataset=dataset,
            n=n_points,
            noise=noise,
            seed=seed
        )
        
        if not response['success']:
            return render_template('error.html', error=response.get('error', 'Unbekannter Fehler'))
        
        # Daten aus dem API-Response extrahieren
        content = response['content']
        plots = response['plots']
        stats = response['stats']
        data = response['data']
        
        # Statistiken für das Template flach klopfen
        stats_dict = _flatten_stats_for_template(stats, data)
        
        # Inhalte mittels HTMLContentRenderer in HTML umwandeln
        from ..content import SimpleRegressionContent
        content_builder = SimpleRegressionContent(stats_dict, {})
        content_obj = content_builder.build()
        
        renderer = HTMLContentRenderer(
            plots=plots,
            data=data,
            stats=stats_dict
        )
        content_dict = renderer.render_to_dict(content_obj)
        
        # Verfügbare Datensätze für das Dropdown-Menü
        datasets = regression_api.get_datasets()
        
        return render_template(
            'educational_content.html',
            title=content['title'],
            subtitle=content['subtitle'],
            content_html=content_dict['full_html'],
            chapters=content_dict['chapters'],
            analysis_type='simple',
            dataset=dataset,
            datasets=datasets['data']['simple'],
            n_points=n_points,
            noise=noise,
            seed=seed,
            stats=stats_dict,
            plots_json=json.dumps(plots),
            ai_configured=ai_api.get_status()['status']['configured']
        )
    
    @app.route('/multiple')
    def multiple_regression():
        """Seite für die multiple Regressionsanalyse."""
        # Parameter abrufen
        dataset = request.args.get('dataset', 'cities')
        n_points = int(request.args.get('n', 75))
        noise = float(request.args.get('noise', 3.5))
        seed = int(request.args.get('seed', 42))
        
        # Content-API aufrufen (identische Logik wie Simple Regression)
        response = content_api.get_multiple_content(
            dataset=dataset,
            n=n_points,
            noise=noise,
            seed=seed
        )
        
        if not response['success']:
            return render_template('error.html', error=response.get('error', 'Unbekannter Fehler'))
        
        # Daten extrahieren
        content = response['content']
        plots = response['plots']
        stats = response['stats']
        data = response['data']
        
        # Flache Struktur für die Template-Engine (Jinja2)
        stats_dict = _flatten_multiple_stats_for_template(stats, data)
        
        # Content generieren
        from ..content import MultipleRegressionContent
        content_builder = MultipleRegressionContent(stats_dict, {})
        content_obj = content_builder.build()
        
        renderer = HTMLContentRenderer(
            plots=plots,
            data=data,
            stats=stats_dict
        )
        content_dict = renderer.render_to_dict(content_obj)
        
        datasets = regression_api.get_datasets()
        
        return render_template(
            'educational_content.html',
            title=content['title'],
            subtitle=content['subtitle'],
            content_html=content_dict['full_html'],
            chapters=content_dict['chapters'],
            analysis_type='multiple',
            dataset=dataset,
            datasets=datasets['data']['multiple'],
            n_points=n_points,
            noise=noise,
            seed=seed,
            stats=stats_dict,
            plots_json=json.dumps(plots),
            ai_configured=ai_api.get_status()['status']['configured']
        )
    
    @app.route('/classification')
    def classification():
        """Seite für Klassifikationsanalysen (Logistische Regression & KNN)."""
        # Parameter (Klassifikation benötigt oft train_size und k)
        dataset = request.args.get('dataset', 'fruits')
        method = request.args.get('method', 'logistic')
        n_points = int(request.args.get('n', 100))
        noise = float(request.args.get('noise', 0.2))
        seed = int(request.args.get('seed', 42))
        k = int(request.args.get('k', 3))
        train_size = float(request.args.get('train_size', 0.8))
        stratify = request.args.get('stratify', 'true').lower() == 'true'
        
        # Content-API für Klassifikation abfragen
        response = content_api.get_classification_content(
            dataset=dataset,
            method=method,
            n=n_points,
            noise=noise,
            seed=seed,
            k=k,
            train_size=train_size,
            stratify=stratify
        )
        
        if not response['success']:
             return render_template('error.html', error=response.get('error', 'Unbekannter Fehler'))
             
        # Daten extrahieren
        content = response['content']
        plots = response['plots']
        stats = response['stats']
        data = response['data']
        results = response.get('results', {})
        
        # Statistiken für das Frontend aufbereiten
        stats_dict = _flatten_classification_stats_for_template(stats, data, results)
        
        # Konvertierung der API-Inhaltsstruktur in ein EducationalContent Objekt
        try:
             from ..infrastructure.content.structure import EducationalContent
             content_obj = EducationalContent.from_dict(content)
        except:
             # Rückfall, falls Konvertierung fehlschlägt
             from ..content import ClassificationContent
             content_builder = ClassificationContent(stats_dict, {})
             content_obj = content_builder.build()

        renderer = HTMLContentRenderer(
            plots=plots,
            data=data,
            stats=stats_dict
        )
        content_dict = renderer.render_to_dict(content_obj)
        
        datasets = regression_api.get_datasets()
        
        return render_template(
            'educational_content.html',
            title=content.get('title', 'Klassifikation'),
            subtitle=content.get('subtitle', 'Logistische Regression & KNN'),
            content_html=content_dict['full_html'],
            chapters=content_dict['chapters'],
            analysis_type='classification',
            dataset=dataset,
            datasets=datasets['data']['classification'],
            n_points=n_points,
            noise=noise,
            seed=seed,
            stats=stats_dict,
            plots_json=json.dumps(plots),
            ai_configured=ai_api.get_status()['status']['configured'],
            # Zusätzliche Parameter für UI-Controls
            method=method,
            k=k,
            train_size=train_size,
            stratify=str(stratify).lower()
        )
    
    # =========================================================================
    # API ENDPOINTS - Proxy to API Layer
    # =========================================================================
    
    # =========================================================================
    # API ENDPOINTS - Proxy zur API-Schicht
    # =========================================================================
    
    @app.route('/api/datasets', methods=['GET'])
    def api_datasets():
        """Gibt verfügbare Datensätze über die API zurück."""
        return jsonify(regression_api.get_datasets())
    
    @app.route('/api/regression/simple', methods=['POST'])
    def api_simple_regression():
        """Führt eine einfache Regression via API-Proxy aus."""
        data = request.get_json() or {}
        return jsonify(regression_api.run_simple(**data))
    
    @app.route('/api/regression/multiple', methods=['POST'])
    def api_multiple_regression():
        """Führt eine multiple Regression via API-Proxy aus."""
        data = request.get_json() or {}
        return jsonify(regression_api.run_multiple(**data))
    
    @app.route('/api/content/simple', methods=['POST'])
    def api_content_simple():
        """Liefert Inhalte für einfache Regression via API."""
        data = request.get_json() or {}
        return jsonify(content_api.get_simple_content(**data))
    
    @app.route('/api/content/multiple', methods=['POST'])
    def api_content_multiple():
        """Liefert Inhalte für multiple Regression via API."""
        data = request.get_json() or {}
        return jsonify(content_api.get_multiple_content(**data))
    
    @app.route('/api/content/schema', methods=['GET'])
    def api_content_schema():
        """Liefert das Inhalts-Schema (Metadaten)."""
        return jsonify(content_api.get_content_schema())
    
    @app.route('/api/ai/interpret', methods=['POST'])
    def api_ai_interpret():
        """
        KI-Interpretation via API-Proxy.
        Akzeptiert JSON mit 'stats' und liefert die Analyse zurück.
        """
        data = request.get_json() or {}
        stats = data.get('stats', {})
        use_cache = data.get('use_cache', True)
        
        result = ai_api.interpret(stats=stats, use_cache=use_cache)
        return jsonify(result)
    
    @app.route('/api/ai/interpret-html', methods=['POST'])
    def api_ai_interpret_html():
        """
        KI-Interpretation via API - liefert fertiges HTML zurück.
        Optimiert für HTMX-Integration (Partial Page Updates).
        """
        from ..ai.ui_components import AIInterpretationHTML
        from ..ai import PerplexityClient
        
        data = request.get_json() or {}
        stats = data.get('stats', {})
        
        # Interpretation über die API holen
        result = ai_api.interpret(stats=stats, use_cache=True)
        
        # Rendering als HTML Fragment
        client = PerplexityClient()
        ui = AIInterpretationHTML(stats, client)
        
        # Hilfsobjekt für das UI-Komponenten-Rendering
        class ResponseObj:
            def __init__(self, result):
                interp = result.get('interpretation', {})
                self.content = interp.get('content', '')
                self.model = interp.get('model', 'unknown')
                self.cached = interp.get('cached', False)
                self.latency_ms = interp.get('latency_ms', 0)
                self.error = not result.get('success', False)
                self.usage = result.get('usage', {})
                self.citations = result.get('citations', [])
        
        response_obj = ResponseObj(result)
        html = ui.render_response(response_obj)
        
        return html
    
    @app.route('/api/ai/r-output', methods=['POST'])
    def api_ai_r_output():
        """Generiert R-Style Summary Output via API."""
        data = request.get_json() or {}
        stats = data.get('stats', {})
        return jsonify(ai_api.get_r_output(stats))
    
    @app.route('/api/ai/status', methods=['GET'])
    def api_ai_status():
        """Liefert den Status des KI-Dienstes."""
        return jsonify(ai_api.get_status())
    
    @app.route('/api/openapi.json', methods=['GET'])
    def api_openapi():
        """Liefert die OpenAPI-Spezifikation."""
        from ..api.endpoints import UnifiedAPI
        unified = UnifiedAPI()
        return jsonify(unified.get_openapi_spec())
    
    @app.route('/api/health', methods=['GET'])
    def api_health():
        """Health-Check Endpunkt."""
        return jsonify({
            'status': 'ok',
            'framework': 'flask',
            'api_powered': True
        })
    
    # =========================================================================
    # KI-INTERPRETATIONSSEITEN
    # =========================================================================
    
    @app.route('/interpret/<analysis_type>')
    def interpret_page(analysis_type: str):
        """
        Dedizierte Seite für die KI-gestützte Interpretation der Ergebnisse.
        """
        dataset = request.args.get('dataset', 'electronics' if analysis_type == 'simple' else 'cities')
        n_points = int(request.args.get('n', 50 if analysis_type == 'simple' else 75))
        
        # Daten und Statistiken über die API beziehen
        if analysis_type == 'simple':
            response = content_api.get_simple_content(dataset=dataset, n=n_points)
            stats_dict = _flatten_stats_for_template(response['stats'], response['data'])
        else:
            response = content_api.get_multiple_content(dataset=dataset, n=n_points)
            stats_dict = _flatten_multiple_stats_for_template(response['stats'], response['data'])
        
        # Interpretation und R-Output über die API anfordern
        interp_result = ai_api.interpret(stats=stats_dict)
        r_output_result = ai_api.get_r_output(stats_dict)
        
        return render_template(
            'interpret.html',
            analysis_type=analysis_type,
            dataset=dataset,
            n_points=n_points,
            stats=stats_dict,
            interpretation=interp_result.get('interpretation', {}),
            r_output=r_output_result.get('r_output', ''),
            ai_configured=ai_api.get_status()['status']['configured'],
            usage=interp_result.get('usage', {}),
            citations=interp_result.get('citations', [])
        )
    
    return app


# =========================================================================
# HILFSFUNKTIONEN FÜR TEMPLATE-RENDERING
# Diese Funktionen transformieren API-Antworten in Jinja2-freundliche Formate.
# =========================================================================

def _flatten_stats_for_template(stats: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Konvertiert API-Stats in ein flaches Dict für die einfache Regressions-UI."""
    coefficients = stats.get('coefficients', {})
    model_fit = stats.get('model_fit', {})
    t_tests = stats.get('t_tests', {})
    sum_of_squares = stats.get('sum_of_squares', {})
    sample = stats.get('sample', {})
    standard_errors = stats.get('standard_errors', {})
    extra = stats.get('extra', {})
    
    return {
        # Context
        'context_title': data.get('context', {}).get('title', 'Regressionsanalyse'),
        'context_description': data.get('context', {}).get('description', ''),
        'x_label': data.get('x_label', 'X'),
        'y_label': data.get('y_label', 'Y'),
        'y_unit': data.get('y_unit', ''),
        
        # Sample
        'n': sample.get('n', 0),
        'df': sample.get('df', 0),
        
        # Coefficients
        'intercept': coefficients.get('intercept', 0),
        'slope': coefficients.get('slope', 0),
        
        # Standard errors
        'se_intercept': standard_errors.get('intercept', 0),
        'se_slope': standard_errors.get('slope', 0),
        
        # t-tests
        't_intercept': t_tests.get('intercept', {}).get('t_value', 0),
        't_slope': t_tests.get('slope', {}).get('t_value', 0),
        'p_intercept': t_tests.get('intercept', {}).get('p_value', 1),
        'p_slope': t_tests.get('slope', {}).get('p_value', 1),
        
        # Model fit
        'r_squared': model_fit.get('r_squared', 0),
        'r_squared_adj': model_fit.get('r_squared_adj', 0),
        
        # Sum of squares
        'sse': sum_of_squares.get('sse', 0),
        'sst': sum_of_squares.get('sst', 0),
        'ssr': sum_of_squares.get('ssr', 0),
        'mse': sum_of_squares.get('mse', 0),
        
        # Extra
        'correlation': extra.get('correlation', 0),
        'x_mean': extra.get('x_mean', 0),
        'y_mean': extra.get('y_mean', 0),
        
        # Computed
        'f_statistic': (sum_of_squares.get('ssr', 0) / 1) / sum_of_squares.get('mse', 1) if sum_of_squares.get('mse', 0) > 0 else 0,
        'p_f': t_tests.get('slope', {}).get('p_value', 1),
    }


def _flatten_multiple_stats_for_template(stats: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten API stats response for multiple regression templates."""
    coefficients = stats.get('coefficients', {})
    model_fit = stats.get('model_fit', {})
    t_tests = stats.get('t_tests', {})
    sample = stats.get('sample', {})
    standard_errors = stats.get('standard_errors', [0, 0, 0])
    
    slopes = coefficients.get('slopes', [0, 0])
    t_values = t_tests.get('t_values', [0, 0, 0])
    p_values = t_tests.get('p_values', [1, 1, 1])
    
    return {
        # Context
        'context_title': 'Multiple Regression',
        'context_description': f"Analyse von {data.get('y_label', 'Y')}",
        'x1_label': data.get('x1_label', 'X₁'),
        'x2_label': data.get('x2_label', 'X₂'),
        'y_label': data.get('y_label', 'Y'),
        
        # Sample
        'n': sample.get('n', 0),
        'k': sample.get('k', 2),
        'df': sample.get('n', 0) - 3,
        
        # Coefficients
        'intercept': coefficients.get('intercept', 0),
        'beta1': slopes[0] if len(slopes) > 0 else 0,
        'beta2': slopes[1] if len(slopes) > 1 else 0,
        
        # Standard errors
        'se_intercept': standard_errors[0] if len(standard_errors) > 0 else 0,
        'se_beta1': standard_errors[1] if len(standard_errors) > 1 else 0,
        'se_beta2': standard_errors[2] if len(standard_errors) > 2 else 0,
        
        # t-tests
        't_intercept': t_values[0] if len(t_values) > 0 else 0,
        't_beta1': t_values[1] if len(t_values) > 1 else 0,
        't_beta2': t_values[2] if len(t_values) > 2 else 0,
        'p_intercept': p_values[0] if len(p_values) > 0 else 1,
        'p_beta1': p_values[1] if len(p_values) > 1 else 1,
        'p_beta2': p_values[2] if len(p_values) > 2 else 1,
        
        # Model fit
        'r_squared': model_fit.get('r_squared', 0),
        'r_squared_adj': model_fit.get('r_squared_adj', 0),
        'f_statistic': model_fit.get('f_statistic', 0),
        'p_f': model_fit.get('f_p_value', 1),
    }


def _flatten_classification_stats_for_template(stats: Dict[str, Any], data: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    """Bereitet Klassifikations-Daten für das HTML-Template auf."""
    metrics = stats # Bei der Klassifikation sind Stats im Wesentlichen die Metriken
    test_metrics = results.get('test_metrics', {})
    
    return {
        'context_title': 'Klassifikationsanalyse',
        'context_description': f"Vorhersage von {data.get('target_names', ['Klasse'])[0] if isinstance(data.get('target_names'), list) and len(data.get('target_names'))>0 else 'Zielvariable'}",
        
        # Methoden-Informationen
        'method': stats.get('method', 'unbekannt'),
        'k': stats.get('k'),
        
        # Trainings-Metriken
        'accuracy': metrics.get('accuracy', 0),
        'precision': metrics.get('precision', 0),
        'recall': metrics.get('recall', 0),
        'f1': metrics.get('f1', 0),
        
        # Test-Metriken (falls Split vorhanden)
        'test_accuracy': test_metrics.get('accuracy'),
        'test_precision': test_metrics.get('precision'),
        'test_recall': test_metrics.get('recall'),
        'test_f1': test_metrics.get('f1'),
        
        # Konfusionsmatrix
        'confusion_matrix': metrics.get('confusion_matrix'),
        
        # Logistische Regressions-Parameter
        'coefficients': metrics.get('coefficients'),
        'intercept': metrics.get('intercept'),
        
        # Platzhalter für Regressions-Keys zur Vermeidung von Template-Fehlern
        'r_squared': 0, 
        'p_slope': 1,
        'slope': 0,
        'n': len(data.get('y', []))
    }

def run_flask(host: str = '0.0.0.0', port: int = 5000, debug: bool = True):
    """Startet die Flask-Anwendung."""
    app = create_flask_app()
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                    🌐 Flask Web-Anwendung                                ║
    ║                    100% API-Gestützte Architektur                        ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  Nutzt dieselbe API-Schicht wie moderne Frontends (React, Vue, etc.)      ║
    ║                                                                           ║
    ║  Seiten:                                                                  ║
    ║    /              - Startseite (Landing Page)                             ║
    ║    /simple        - Einfache Regressionsanalyse                           ║
    ║    /multiple      - Multiple Regressionsanalyse                           ║
    ║    /classification - Klassifikationsanalyse                                ║
    ║    /interpret/*   - KI-Interpretation                                     ║
    ║                                                                           ║
    ║  API-Endpunkte (identisch zum API-Server):                                ║
    ║    /api/datasets                  - Datensätze auflisten                  ║
    ║    /api/regression/simple         - Regression ausführen                  ║
    ║    /api/content/simple            - Edukativen Content laden              ║
    ║    /api/ai/interpret              - KI-Interpretation anfordern           ║
    ║    /api/openapi.json              - OpenAPI Spezifikation                 ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"🌐 Server: http://{host}:{port}")
    print(f"📚 API: http://{host}:{port}/api/openapi.json")
    print()
    
    app.run(debug=debug, host=host, port=port)


if __name__ == "__main__":
    run_flask()
