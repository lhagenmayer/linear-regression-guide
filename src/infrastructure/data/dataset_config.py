"""
Konfiguration für Datensatz-Metadaten.

Zentralisiert alle hardcodierten Strings und Labels für Datensätze,
um Wartbarkeit und Konsistenz zu verbessern.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class DatasetMetadata:
    """Metadaten für einen einzelnen Datensatz."""
    y_label: str
    x_label: str = ""
    x_unit: str = ""
    y_unit: str = ""
    context_title: str = ""
    context_description: str = ""
    x1_label: str = ""
    x2_label: str = ""


# Konfiguration für einfache Regression
SIMPLE_DATASET_CONFIG: Dict[str, DatasetMetadata] = {
    "electronics": DatasetMetadata(
        x_label="Verkaufsfläche (100 qm)",
        y_label="Umsatz (Mio. €)",
        x_unit="100 qm",
        y_unit="Mio. €",
        context_title="🏪 Elektronikmarkt",
        context_description="Analyse des Zusammenhangs zwischen Verkaufsfläche und Umsatz"
    ),
    "advertising": DatasetMetadata(
        x_label="Werbeausgaben ($)",
        y_label="Umsatz ($)",
        x_unit="$",
        y_unit="$",
        context_title="📢 Werbestudie",
        context_description="Zusammenhang zwischen Werbeausgaben und Umsatz"
    ),
    "temperature": DatasetMetadata(
        x_label="Temperatur (°C)",
        y_label="Eisverkauf (Einheiten)",
        x_unit="°C",
        y_unit="Einheiten",
        context_title="🍦 Eisverkauf",
        context_description="Zusammenhang zwischen Temperatur und Eisverkauf"
    ),
    "synthetic": DatasetMetadata(
        x_label="X",
        y_label="Y",
        context_title="Synthetische Daten",
        context_description="Generierte Daten für Demonstrationszwecke"
    ),
}


# Konfiguration für multiple Regression
MULTIPLE_DATASET_CONFIG: Dict[str, DatasetMetadata] = {
    "cities": DatasetMetadata(
        x1_label="Preis (CHF)",
        x2_label="Werbung (1000 CHF)",
        y_label="Umsatz (1000 CHF)",
        context_title="Städtestudie",
        context_description="Preis & Werbung → Umsatz"
    ),
    "houses": DatasetMetadata(
        x1_label="Wohnfläche (sqft/10)",
        x2_label="Pool (0/1)",
        y_label="Preis ($1000)",
        context_title="Immobilienpreise",
        context_description="Wohnfläche & Pool → Preis"
    ),
    "cantons": DatasetMetadata(
        x1_label="Bevölkerungsdichte (Einwohner/km²)",
        x2_label="Ausländeranteil (%)",
        y_label="BIP pro Kopf (CHF)",
        context_title="🇨🇭 Schweizer Kantone",
        context_description="Schweizer Kantone (Sozioökonomisch)"
    ),
    "weather": DatasetMetadata(
        x1_label="Höhe über Meer (m)",
        x2_label="Sonnenstunden (h/Jahr)",
        y_label="Jahresmitteltemperatur (°C)",
        context_title="🌤️ Schweizer Wetter",
        context_description="Schweizer Wetterstationen"
    ),
    "world_bank": DatasetMetadata(
        x1_label="GDP per Capita (USD)",
        x2_label="Education Years",
        y_label="Life Expectancy (years)",
        context_title="🏦 World Bank",
        context_description="World Bank Development Indicators"
    ),
    "fred_economic": DatasetMetadata(
        x1_label="Unemployment Rate (%)",
        x2_label="Interest Rate (%)",
        y_label="GDP (Billions USD)",
        context_title="💰 FRED",
        context_description="FRED US Economic Data"
    ),
    "who_health": DatasetMetadata(
        x1_label="Health Expenditure ($)",
        x2_label="Sanitation Access (%)",
        y_label="Life Expectancy (years)",
        context_title="🏥 WHO",
        context_description="WHO Global Health Indicators"
    ),
    "eurostat": DatasetMetadata(
        x1_label="Employment Rate (%)",
        x2_label="Tertiary Education (%)",
        y_label="GDP (Million €)",
        context_title="🇪🇺 Eurostat",
        context_description="Eurostat Socioeconomic Data"
    ),
    "nasa_weather": DatasetMetadata(
        x1_label="Temperature (°C)",
        x2_label="Solar Radiation (kWh/m²/day)",
        y_label="Crop Yield Index",
        context_title="🛰️ NASA POWER",
        context_description="NASA POWER Agro-Climatology"
    ),
}


def get_simple_config(dataset_id: str) -> DatasetMetadata:
    """Holt die Konfiguration für einen einfachen Regressions-Datensatz."""
    return SIMPLE_DATASET_CONFIG.get(dataset_id, DatasetMetadata(
        x_label="X",
        y_label="Y",
        context_title="Unbekannter Datensatz",
        context_description="Keine Beschreibung verfügbar"
    ))


def get_multiple_config(dataset_id: str) -> DatasetMetadata:
    """Holt die Konfiguration für einen multiplen Regressions-Datensatz."""
    return MULTIPLE_DATASET_CONFIG.get(dataset_id, DatasetMetadata(
        x1_label="X1",
        x2_label="X2",
        y_label="Y",
        context_title="Unbekannter Datensatz",
        context_description="Keine Beschreibung verfügbar"
    ))
