"""
Domain Entities.
Objekte mit Identität und Lebenszyklus.
Pure Python - KEINE externen Abhängigkeiten (außer uuid für die ID-Generierung).
"""
from dataclasses import dataclass, field
from typing import List, Optional
import uuid

# Import der Value Objects aus dem Domain-Layer
from .value_objects import (
    RegressionParameters, 
    RegressionMetrics, 
    DatasetMetadata,
    RegressionType,
    ModelQuality,
)


@dataclass
class RegressionModel:
    """
    Core Domain Entity, die ein trainiertes Regressionsmodell repräsentiert.
    Besitzt eine eindeutige Identität (id) und hält den Zustand der Analyse.
    
    Architekturentscheidung: Wir nutzen Strings für Datumsangaben (ISO), 
    um Abhängigkeiten zum 'datetime'-Modul im Core-Layer zu vermeiden.
    Der Infrastructure-Layer kümmert sich um die Konvertierung.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at_iso: str = ""  # ISO-Format String, wird vom Infrastructure-Layer gesetzt
    regression_type: RegressionType = RegressionType.SIMPLE
    dataset_metadata: Optional[DatasetMetadata] = None
    
    # Zustand des Modells (Initial leer bis zum Training)
    parameters: Optional[RegressionParameters] = None
    metrics: Optional[RegressionMetrics] = None
    
    # Ergebnisse der Analyse
    residuals: List[float] = field(default_factory=list)
    predictions: List[float] = field(default_factory=list)
    
    def is_trained(self) -> bool:
        """Prüft, ob das Modell erfolgreich trainiert wurde (Parameter und Metriken vorhanden)."""
        return self.parameters is not None and self.metrics is not None

    def get_equation_string(self) -> str:
        """Domain-Logik zur Erzeugung einer mathematischen Gleichung als Text."""
        if not self.is_trained():
            return "Nicht trainiert"
            
        # Start mit dem Achsenabschnitt (Intercept)
        parts = [f"{self.parameters.intercept:.4f}"]
        # Hinzufügen der Koeffizienten für jede Variable
        for name, coef in self.parameters.coefficients.items():
            sign = "+" if coef >= 0 else "-"
            parts.append(f"{sign} {abs(coef):.4f}·{name}")
            
        return "ŷ = " + " ".join(parts)
    
    def get_quality(self) -> Optional[ModelQuality]:
        """Gibt die Qualitätsstufe des Modells zurück (basierend auf R²)."""
        if not self.is_trained():
            return None
        return self.metrics.quality
    
    def get_r_squared(self) -> Optional[float]:
        """Gibt den Determinationskoeffizienten (R²) zurück."""
        if not self.is_trained():
            return None
        return self.metrics.r_squared
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Überprüft die statistische Signifikanz zum gegebenen Alpha-Level."""
        if not self.is_trained():
            return False
        return self.metrics.is_significant(alpha)
    
    def validate(self) -> List[str]:
        """Validiert den Zustand der Entity und gibt eine Liste von Fehlern zurück."""
        errors = []
        if self.is_trained():
            if not self.predictions:
                errors.append("Ein trainiertes Modell sollte Vorhersagen enthalten")
            if not self.residuals:
                errors.append("Ein trainiertes Modell sollte Residuen enthalten")
            if len(self.predictions) != len(self.residuals):
                errors.append("Vorhersagen und Residuen müssen die gleiche Länge haben")
        return errors
