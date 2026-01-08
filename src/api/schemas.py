from enum import Enum
from pydantic import BaseModel, Field, field_validator

"""
API Schemas - Validierung der Eingabeparameter.
Nutzt Pydantic für Typsicherheit und automatische Fehlermeldungen.
"""


class DatasetType(str, Enum):
    """Verfügbare Datensatz-Typen für die Analyse."""
    ELECTRONICS = "electronics"
    ADVERTISING = "advertising"
    TEMPERATURE = "temperature"
    CITIES = "cities"
    HOUSES = "houses"
    CANTONS = "cantons"
    WEATHER = "weather"
    WORLD_BANK = "world_bank"
    FRED_ECONOMIC = "fred_economic"
    WHO_HEALTH = "who_health"
    EUROSTAT = "eurostat"
    NASA_WEATHER = "nasa_weather"


class BaseRegressionRequest(BaseModel):
    """Basis-Anwendungsparameter für alle Regressions-Requests."""
    dataset: DatasetType = Field(default=DatasetType.ELECTRONICS, description="Eindeutige ID des Datensatzes")
    n: int = Field(default=50, ge=10, le=1000, description="Stichprobengröße (10-1000)")
    noise: float = Field(default=0.4, ge=0.0, le=100.0, description="Rauschfaktor")
    seed: int = Field(default=42, description="Zufalls-Seed für Reproduzierbarkeit")
    include_predictions: bool = Field(default=True, description="Vorhersagen im Response einfügen")

    @field_validator("n")
    def validate_n(cls, v):
        """Stellt sicher, dass n eine Ganzzahl ist."""
        if v % 1 != 0:
            raise ValueError("n muss eine Ganzzahl sein")
        return v


class SimpleRegressionRequest(BaseRegressionRequest):
    """Parameter speziell für die einfache lineare Regression."""
    true_intercept: float = Field(default=0.6, description="Wahrer y-Achsenabschnitt (β₀) für die Generierung")
    true_slope: float = Field(default=0.52, description="Wahre Steigung (β₁) für die Generierung")


class MultipleRegressionRequest(BaseRegressionRequest):
    """Parameter für die multiple Regression (erbt von der Basis)."""
    pass


class AIInterpretationRequest(BaseModel):
    """Request für die KI-gestützte Interpretation von Ergebnissen."""
    stats: dict = Field(..., description="Dictionary mit statistischen Kennzahlen")
    use_cache: bool = Field(default=True, description="Gecachte Interpretation nutzen, falls verfügbar")
