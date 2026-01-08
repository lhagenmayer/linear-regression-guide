"""
Framework-Detektor - Automatische Erkennung des aktiven UI-Frameworks.
"""

import sys
import os
from enum import Enum
from typing import Optional


class Framework(Enum):
    """Unterstützte Web-Frameworks."""
    STREAMLIT = "streamlit"
    FLASK = "flask"
    UNKNOWN = "unbekannt"


class FrameworkDetector:
    """
    Erkennt das aktuelle Framework zur Laufzeit.
    
    Reihenfolge der Erkennung:
    1. Umgebungsvariable REGRESSION_FRAMEWORK
    2. Streamlit Runtime-Kontext
    3. Flask App-Kontext
    4. Kommandozeilenargumente
    5. Standardmäßig 'unbekannt'
    """
    
    _cached_framework: Optional[Framework] = None
    
    @classmethod
    def detect(cls) -> Framework:
        """Führt die Erkennung aus und cached das Ergebnis."""
        if cls._cached_framework is not None:
            return cls._cached_framework
        
        # 1. Expliziter Override via Umgebungsvariable
        env_framework = os.environ.get("REGRESSION_FRAMEWORK", "").lower()
        if env_framework == "streamlit":
            cls._cached_framework = Framework.STREAMLIT
            return cls._cached_framework
        elif env_framework == "flask":
            cls._cached_framework = Framework.FLASK
            return cls._cached_framework
        
        # 2. Prüfung auf Streamlit-Kontext
        if cls._is_streamlit():
            cls._cached_framework = Framework.STREAMLIT
            return cls._cached_framework
        
        # 3. Prüfung auf Flask-Kontext (aktive App)
        if cls._is_flask():
            cls._cached_framework = Framework.FLASK
            return cls._cached_framework
        
        # 4. Analyse der CLI-Argumente
        if cls._check_cli_args():
            return cls._cached_framework
        
        # 5. Fallback
        cls._cached_framework = Framework.UNKNOWN
        return cls._cached_framework
    
    @classmethod
    def _is_streamlit(cls) -> bool:
        """Prüft, ob der Code innerhalb eines Streamlit-Kontextes läuft."""
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            ctx = get_script_run_ctx()
            return ctx is not None
        except ImportError:
            pass
        
        # Fallback: Prüfung ob streamlit in den sys.modules geladen ist
        if "streamlit" in sys.modules:
            for arg in sys.argv:
                if "streamlit" in arg.lower():
                    return True
        
        return False
    
    @classmethod
    def _is_flask(cls) -> bool:
        """Prüft, ob eine Flask-Instanz aktiv ist."""
        try:
            from flask import current_app
            return current_app is not None
        except (ImportError, RuntimeError):
            pass
        
        return False
    
    @classmethod
    def _check_cli_args(cls) -> bool:
        """Sucht in den Start-Argumenten nach Framework-Hinweisen."""
        args = " ".join(sys.argv).lower()
        
        if "streamlit" in args:
            cls._cached_framework = Framework.STREAMLIT
            return True
        elif "flask" in args or "gunicorn" in args or "waitress" in args:
            cls._cached_framework = Framework.FLASK
            return True
        
        return False
    
    @classmethod
    def reset(cls) -> None:
        """Setzt den Cache zurück (primär für Tests relevant)."""
        cls._cached_framework = None
    
    @classmethod
    def is_streamlit(cls) -> bool:
        """Hilfsmethode: Läuft Streamlit?"""
        return cls.detect() == Framework.STREAMLIT
    
    @classmethod
    def is_flask(cls) -> bool:
        """Hilfsmethode: Läuft Flask?"""
        return cls.detect() == Framework.FLASK
