"""
Infrastructure: Daten-Generierung und -Beschaffung.
Dieses Modul stellt alle Funktionen zur Verfügung, um Datensätze für Simulationen 
und Fallbeispiele (Regression & Klassifikation) zu erzeugen oder zu laden.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import requests
from typing import Dict, Any, Optional, List, Union

from ...config import get_logger
from ...config.logger import log_error_with_context
from .dataset_config import get_simple_config, get_multiple_config

logger = get_logger(__name__)


@dataclass
class DataResult:
    """Ergebnis eines Datenabrufs für die einfache Regression."""
    x: np.ndarray # Unabhängige Variable (Prädiktor)
    y: np.ndarray # Abhängige Variable (Target)
    x_label: str
    y_label: str
    x_unit: str = ""
    y_unit: str = ""
    context_title: str = ""
    context_description: str = ""
    extra: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra is None:
            self.extra = {}
    
    @property
    def n(self) -> int:
        """Anzahl der Beobachtungen."""
        return len(self.x)


@dataclass 
class MultipleRegressionDataResult:
    """Ergebnis eines Datenabrufs für die multiple Regression (2 Prädiktoren)."""
    x1: np.ndarray # Erster Prädiktor
    x2: np.ndarray # Zweiter Prädiktor
    y: np.ndarray  # Zielvariable
    x1_label: str
    x2_label: str
    y_label: str
    extra: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra is None:
            self.extra = {}
    
    @property
    def n(self) -> int:
        """Anzahl der Beobachtungen."""
        return len(self.y)


@dataclass
class ClassificationDataResult:
    """
    Ergebnis eines Datenabrufs für Klassifikationsaufgaben (z.B. KNN, Logistische Regression).
    Unterstützt mehrdimensionale Features und mehrere Klassen.
    """
    X: np.ndarray  # Feature-Matrix (n_samples, n_features)
    y: np.ndarray  # Label-Array (n_samples,)
    feature_names: List[str]
    target_names: List[str]
    context_title: str = ""
    context_description: str = ""
    extra: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra is None:
            self.extra = {}
    
    @property
    def n(self) -> int:
        """Anzahl der Stichproben."""
        return len(self.y)
    
    @property
    def n_features(self) -> int:
        """Anzahl der Merkmale (Features)."""
        return self.X.shape[1] if len(self.X.shape) > 1 else 1
    
    @property
    def n_classes(self) -> int:
        """Anzahl der Zielklassen."""
        return len(self.target_names)


class DataFetcher:
    """
    Zentrale Komponente zur Datenbeschaffung.
    Bietet ein einheitliches Interface für verschiedene Datenquellen und Simulationen.
    """
    
    def __init__(self):
        self._generators = {}
        logger.info("DataFetcher initialisiert")
    
    def get_simple(
        self,
        dataset: str,
        n: int = 50,
        noise: float = 0.4,
        seed: int = 42,
        true_intercept: float = 0.6,
        true_slope: float = 0.52,
    ) -> DataResult:
        """
        Beschafft Daten für die einfache lineare Regression.
        """
        logger.info(f"Abruf von Daten für einfache Regression: {dataset}, n={n}")
        np.random.seed(seed)
        
        result = None
        if dataset == "electronics":
            result = self._generate_electronics(n, noise, true_intercept, true_slope)
        elif dataset == "advertising":
            result = self._generate_advertising(n, noise)
        elif dataset == "temperature":
            result = self._generate_temperature(n, noise)
        elif dataset == "cantons":
             res = self._generate_cantons(n, noise)
             result = DataResult(x=res.x1, y=res.y, x_label=res.x1_label, y_label=res.y_label, context_title="Schweizer Kantone", context_description=res.extra.get("context", ""))
        elif dataset == "weather":
             res = self._generate_weather(n, noise)
             result = DataResult(x=res.x1, y=res.y, x_label=res.x1_label, y_label=res.y_label, context_title="Schweizer Wetter", context_description="Höhe -> Temperatur")
        elif dataset == "world_bank":
             res = self._generate_world_bank(n, noise)
             result = DataResult(x=res.x1, y=res.y, x_label=res.x1_label, y_label=res.y_label, context_title="World Bank", context_description="BIP -> Lebenserwartung")
        elif dataset == "fred_economic":
             res = self._generate_fred(n, noise)
             result = DataResult(x=res.x1, y=res.y, x_label=res.x1_label, y_label=res.y_label, context_title="FRED", context_description="Arbeitslosigkeit -> BIP")
        elif dataset == "who_health":
             res = self._generate_who(n, noise)
             result = DataResult(x=res.x1, y=res.y, x_label=res.x1_label, y_label=res.y_label, context_title="WHO", context_description="Gesundheitsausgaben -> Lebenserwartung")
        elif dataset == "eurostat":
             res = self._generate_eurostat(n, noise)
             result = DataResult(x=res.x1, y=res.y, x_label=res.x1_label, y_label=res.y_label, context_title="Eurostat", context_description="Beschäftigung -> BIP")
        elif dataset == "nasa_weather":
             res = self._generate_nasa(n, noise)
             result = DataResult(x=res.x1, y=res.y, x_label=res.x1_label, y_label=res.y_label, context_title="NASA", context_description="Temperatur -> Ernteertrag")
        else:
            raise ValueError(f"Unbekannter Datensatz für einfache Regression: {dataset}")
            
        if result:
            result.extra["dataset"] = dataset
            
        return result
    
    def get_multiple(
        self,
        dataset: str,
        n: int = 75,
        noise: float = 3.5,
        seed: int = 42,
    ) -> MultipleRegressionDataResult:
        """
        Beschafft Daten für die multiple lineare Regression.
        """
        logger.info(f"Abruf von Daten für multiple Regression: {dataset}, n={n}")
        np.random.seed(seed)
        
        result = None
        if dataset == "cities":
            result = self._generate_cities(n, noise)
        elif dataset == "houses":
            result = self._generate_houses(n, noise)
        elif dataset == "cantons":
             result = self._generate_cantons(n, noise)
        elif dataset == "weather":
             result = self._generate_weather(n, noise)
        elif dataset == "world_bank":
            result = self._generate_world_bank(n, noise)
        elif dataset == "fred_economic":
            result = self._generate_fred(n, noise)
        elif dataset == "who_health":
            result = self._generate_who(n, noise)
        elif dataset == "eurostat":
             result = self._generate_eurostat(n, noise)
        elif dataset == "nasa_weather":
             result = self._generate_nasa(n, noise)
        else:
            error = ValueError(f"Unbekannter Datensatz für multiple Regression: {dataset}")
            log_error_with_context(
                logger=logger,
                error=error,
                context="Multiple regression dataset validation",
                service="DataFetcher",
                operation="fetch_multiple",
                dataset=dataset,
                available_datasets=list(MULTIPLE_DATASETS.keys())
            )
            raise error
            
        if result:
            result.extra["dataset"] = dataset
            
        return result
    
    # =========================================================
    # PRIVATE: External Data Fetchers (Mocked for Stability)
    # =========================================================

    def _fetch_world_bank(self, indicators: List[str], countries: List[str] = None, years: List[int] = None) -> pd.DataFrame:
        """Fetch/Mock World Bank data."""
        try:
            if countries is None:
                countries = ['USA', 'CHN', 'DEU', 'JPN', 'GBR', 'FRA', 'ITA', 'CAN', 'AUS', 'ESP']
            if years is None:
                years = list(range(2010, 2021))

            logger.info(f"World Bank API: Fetching {len(indicators)} indicators for {len(countries)} countries")

            # Mock data
            mock_data = []
            for country in countries:
                for year in years:
                    for indicator in indicators:
                        value = np.random.normal(1000, 200) if 'GDP' in indicator else np.random.normal(50, 10)
                        mock_data.append({
                            'country': country, 'year': year, 'indicator': indicator,
                            'value': max(0, value)
                        })
            
            df = pd.DataFrame(mock_data)
            return df.pivot(index=['country', 'year'], columns='indicator', values='value').reset_index()
        except Exception as e:
            log_error_with_context(
                logger=logger,
                error=e,
                context="World Bank API fetch",
                service="DataFetcher",
                operation="_fetch_worldbank",
                countries=countries,
                years=years,
                indicators=indicators
            )
            return pd.DataFrame()

    def _fetch_fred(self, series_ids: List[str], start_date: str = '2010-01-01', end_date: str = '2023-12-31') -> pd.DataFrame:
        """Fetch/Mock FRED data."""
        try:
            logger.info(f"FRED API: Fetching {len(series_ids)} series")
            date_range = pd.date_range(start=start_date, end=end_date, freq='QS')
            mock_data = {'date': date_range}

            for series_id in series_ids:
                if 'GDP' in series_id:
                    values = np.cumsum(np.random.normal(100, 20, len(date_range))) + 20000
                elif 'UNRATE' in series_id:
                    values = np.clip(np.random.normal(5, 2, len(date_range)), 0, 15)
                else:
                    values = np.random.normal(100, 20, len(date_range))
                mock_data[series_id] = values

            return pd.DataFrame(mock_data)
        except Exception as e:
            log_error_with_context(
                logger=logger,
                error=e,
                context="FRED API fetch",
                service="DataFetcher",
                operation="_fetch_fred",
                series_ids=series_ids,
                start_date=start_date,
                end_date=end_date
            )
            return pd.DataFrame()

    def _fetch_who(self, indicators: List[str], countries: List[str] = None, years: List[int] = None) -> pd.DataFrame:
        """Fetch/Mock WHO data."""
        try:
            if countries is None:
                 countries = ['USA', 'CHN', 'DEU', 'JPN', 'GBR', 'FRA', 'ITA', 'CAN']
            if years is None:
                years = list(range(2010, 2021))
                
            logger.info(f"WHO API: Fetching {len(indicators)} indicators")
            
            mock_data = []
            for country in countries:
                for year in years:
                    for indicator in indicators:
                        if 'WHOSIS_000001' in indicator: value = np.clip(np.random.normal(75, 5), 50, 90)
                        else: value = np.random.normal(100, 20)
                        mock_data.append({'country': country, 'year': year, 'indicator': indicator, 'value': value})
            
            df = pd.DataFrame(mock_data)
            return df.pivot(index=['country', 'year'], columns='indicator', values='value').reset_index()
        except Exception as e:
            log_error_with_context(
                logger=logger,
                error=e,
                context="WHO API fetch",
                service="DataFetcher",
                operation="_fetch_who",
                indicators=indicators,
                countries=countries,
                years=years
            )
            return pd.DataFrame()

    def _fetch_eurostat(self, dataset_codes: List[str], countries: List[str] = None, years: List[int] = None) -> pd.DataFrame:
        """Fetch/Mock Eurostat data."""
        try:
            if countries is None: countries = ['DE', 'FR', 'IT', 'ES', 'NL', 'BE', 'AT', 'SE', 'DK', 'FI']
            if years is None: years = list(range(2010, 2021))
            
            logger.info(f"Eurostat API: Fetching {len(dataset_codes)} datasets")
            mock_data = []
            for country in countries:
                for year in years:
                    for ds in dataset_codes:
                        if 'gdp' in ds: val = np.random.normal(2000000, 500000)
                        elif 'emp' in ds: val = np.clip(np.random.normal(70, 5), 50, 90)
                        else: val = np.random.normal(30, 5)
                        mock_data.append({'country': country, 'year': year, 'dataset': ds, 'value': max(0, val)})
            
            df = pd.DataFrame(mock_data)
            return df.pivot(index=['country', 'year'], columns='dataset', values='value').reset_index()
        except Exception as e:
            log_error_with_context(
                logger=logger,
                error=e,
                context="Eurostat API fetch",
                service="DataFetcher",
                operation="_fetch_eurostat",
                dataset_codes=dataset_codes,
                countries=countries,
                years=years
            )
            return pd.DataFrame()

    def _fetch_nasa(self, variables: List[str], locations: int = 50) -> pd.DataFrame:
        """Fetch/Mock NASA POWER data."""
        try:
            logger.info(f"NASA POWER API: Fetching data for {locations} locations")
            # Lat/Lon grid
            lats = np.random.uniform(-50, 50, locations)
            # Solar Radiation (kWh/m^2/day)
            solar = np.abs(lats) * -0.1 + 8 + np.random.normal(0, 1, locations)
            # Temperature (C)
            temp = 30 - np.abs(lats) * 0.5 + np.random.normal(0, 3, locations)
            
            return pd.DataFrame({'lat': lats, 'solar': solar, 'temp': temp})
        except Exception as e:
            log_error_with_context(
                logger=logger,
                error=e,
                context="NASA POWER API fetch",
                service="DataFetcher",
                operation="_fetch_nasa",
                variables=variables,
                locations=locations
            )
            return pd.DataFrame()

    def _generate_eurostat(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """Generate Eurostat data."""
        es = self._fetch_eurostat(['nama_10_gdp', 'lfsi_emp_a'])
        if es.empty: return self._generate_cities(n, noise)
        
        # Predict GDP based on Employment
        emp = es['lfsi_emp_a'].values
        gdp = es['nama_10_gdp'].values
        # 2nd predictor: Education Index (mock)
        edu = np.clip(0.00001 * gdp + np.random.normal(0, 5, len(gdp)), 10, 60)
        
        config = get_multiple_config("eurostat")
        return MultipleRegressionDataResult(
            x1=emp, x2=edu, y=gdp,
            x1_label=config.x1_label,
            x2_label=config.x2_label,
            y_label=config.y_label,
            extra={"context": config.context_description}
        )

    def _generate_nasa(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """Generate NASA data."""
        nasa = self._fetch_nasa([], locations=n)
        if nasa.empty: return self._generate_cities(n, noise)
        
        temp = nasa['temp'].values
        solar = nasa['solar'].values
        # Predict Crop Yield (for example) using Temp and Solar
        yield_val = 20 + 2 * temp + 5 * solar + np.random.normal(0, noise, len(temp))
        
        config = get_multiple_config("nasa_weather")
        return MultipleRegressionDataResult(
            x1=temp, x2=solar, y=yield_val,
            x1_label=config.x1_label,
            x2_label=config.x2_label,
            y_label=config.y_label,
            extra={"context": config.context_description}
        )

    def _generate_world_bank(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """Generate World Bank data (GDP -> Life Expectancy)."""
        wb_data = self._fetch_world_bank(['NY.GDP.PCAP.KD', 'SP.DYN.LE00.IN'])
        
        if wb_data.empty: return self._generate_cities(n, noise) # Fallback

        gdp = wb_data['NY.GDP.PCAP.KD'].fillna(wb_data['NY.GDP.PCAP.KD'].mean()).values
        life = wb_data['SP.DYN.LE00.IN'].fillna(wb_data['SP.DYN.LE00.IN'].mean()).values
        
        # We need 2 predictors for MultipleRegressionDataResult. 
        # API expects x1, x2. Let's create a dummy or split GDP if needed.
        # Or better: "Education" as 2nd predictor (mocked)
        education = np.clip(0.0005 * gdp + np.random.normal(0, 2, len(gdp)), 5, 20)
        
        config = get_multiple_config("world_bank")
        return MultipleRegressionDataResult(
            x1=gdp, x2=education, y=life,
            x1_label=config.x1_label,
            x2_label=config.x2_label,
            y_label=config.y_label,
            extra={"context": config.context_description}
        )

    def _generate_fred(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """Generate FRED data (Unemployment -> GDP)."""
        fred = self._fetch_fred(['GDP', 'UNRATE'])
        
        if fred.empty: return self._generate_cities(n, noise)
        
        unrate = fred['UNRATE'].values
        gdp = fred['GDP'].values
        # 2nd predictor: Interest Rate (Inverse to GDP usually)
        interest = np.clip(5 - 0.0001 * gdp + np.random.normal(0, 1, len(gdp)), 0, 10)
        
        config = get_multiple_config("fred_economic")
        return MultipleRegressionDataResult(
            x1=unrate, x2=interest, y=gdp,
            x1_label=config.x1_label,
            x2_label=config.x2_label,
            y_label=config.y_label,
            extra={"context": config.context_description}
        )

    def _generate_who(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """Generate WHO data."""
        who = self._fetch_who(['WHOSIS_000001'])
        if who.empty: return self._generate_cities(n, noise)
        
        life_exp = who['WHOSIS_000001'].fillna(75).values
        # Inverse problem: We usually predict Life Exp. 
        # Let's mock Predictors: Healthcare Spend, Sanitation
        spend = (life_exp - 50) * 100 + np.random.normal(0, 500, len(life_exp))
        sanitation = np.clip((life_exp - 40) * 2 + np.random.normal(0, 5, len(life_exp)), 0, 100)
        
        config = get_multiple_config("who_health")
        return MultipleRegressionDataResult(
            x1=spend, x2=sanitation, y=life_exp,
            x1_label=config.x1_label,
            x2_label=config.x2_label,
            y_label=config.y_label,
            extra={"context": config.context_description}
        )
    
    # =========================================================
    # PRIVATE: Daten-Generatoren (Simulationen)
    # =========================================================
    
    def _generate_cantons(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """
        Generiert Daten für Schweizer Kantone (Dichte, Ausländeranteil -> BIP).
        Realitätsnahe Parameter basierend auf sozioökonomischen Daten der Schweiz.
        """
        # Feature 1: Bevölkerungsdichte (Log-Normal verteilt)
        # Die meisten Kantone < 500, einige (Zürich, Genf, Basel) sehr hoch.
        x1 = np.random.lognormal(5.5, 0.8, n)
        x1 = np.clip(x1, 50, 5000)
        
        # Feature 2: Ausländeranteil % (15% bis 50%)
        x2 = np.random.normal(25, 8, n)
        x2 = np.clip(x2, 10, 50)
        
        # Feature 3: Arbeitslosenrate (1.5% bis 5.0%) - Korreliert leicht mit Ausländeranteil
        x3 = 1.5 + 0.05 * x2 + np.random.normal(0, 0.5, n)
        x3 = np.clip(x3, 0.5, 6.0)
        
        # BIP pro Kopf (CHF)
        # Basis 60k + Bonus für Dichte + Bonus für Ausländeranteil (Spezialisten-Effekt) - Malus für Arbeitslosigkeit
        y = 55000 + 5 * x1 + 800 * x2 - 2000 * x3 + np.random.normal(0, noise * 2000, n)
        
        config = get_multiple_config("cantons")
        return MultipleRegressionDataResult(
            x1=x1, x2=x2, y=y,
            x1_label=config.x1_label,
            x2_label=config.x2_label,
            y_label=config.y_label,
            extra={
                "true_b0": 55000, "true_b1": 5, "true_b2": 800,
                "context": config.context_description
            }
        )

    def _generate_weather(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """
        Generiert Daten für Schweizer Wetterstationen (Höhe, Sonnendauer -> Temperatur).
        """
        # x1: Höhe über Meer (Meter) - 300 bis 3500m
        x1 = np.random.uniform(300, 2500, n)
        
        # x2: Sonnenscheindauer (h/Jahr) - 1200 bis 2500h
        # Höhere Lagen haben oft mehr Sonne (über dem Nebel)
        x2 = 1500 + 0.1 * x1 + np.random.normal(0, 200, n)
        x2 = np.clip(x2, 1000, 3000)
        
        # Temperatur sinkt mit der Höhe (~ -0.65°C pro 100m) und steigt mit Sonne
        y = 15 - 0.0065 * x1 + 0.002 * x2 + np.random.normal(0, noise, n)
        
        config = get_multiple_config("weather")
        return MultipleRegressionDataResult(
            x1=x1, x2=x2, y=y,
            x1_label=config.x1_label,
            x2_label=config.x2_label,
            y_label=config.y_label,
            extra={
                "true_b0": 15, "true_b1": -0.0065, "true_b2": 0.002,
                "context": config.context_description
            }
        )
    
    def _generate_electronics(
        self, n: int, noise: float, intercept: float, slope: float
    ) -> DataResult:
        """Generiert Daten für einen Elektronikmarkt (Verkaufsfläche vs. Umsatz)."""
        x = np.random.uniform(2, 10, n)  # Verkaufsfläche in 100 qm
        y = intercept + slope * x + np.random.normal(0, noise, n)
        
        config = get_simple_config("electronics")
        return DataResult(
            x=x, y=y,
            x_label=config.x_label,
            y_label=config.y_label,
            x_unit=config.x_unit,
            y_unit=config.y_unit,
            context_title=config.context_title,
            context_description=config.context_description,
            extra={"true_intercept": intercept, "true_slope": slope}
        )
    
    def _generate_advertising(self, n: int, noise: float) -> DataResult:
        """Generiert Daten für eine Werbestudie."""
        x = np.random.uniform(1000, 10000, n)  # Werbeausgaben in $
        y = 50000 + 5.0 * x + np.random.normal(0, noise * 5000, n)
        
        config = get_simple_config("advertising")
        return DataResult(
            x=x, y=y,
            x_label=config.x_label,
            y_label=config.y_label,
            x_unit=config.x_unit,
            y_unit=config.y_unit,
            context_title=config.context_title,
            context_description=config.context_description
        )
    
    def _generate_temperature(self, n: int, noise: float) -> DataResult:
        """Generiert Daten für Temperatur vs. Eisverkauf."""
        x = np.random.uniform(15, 35, n)  # Temperatur in °C
        y = 20 + 3.0 * x + np.random.normal(0, noise * 10, n)
        
        config = get_simple_config("temperature")
        return DataResult(
            x=x, y=y,
            x_label=config.x_label,
            y_label=config.y_label,
            x_unit=config.x_unit,
            y_unit=config.y_unit,
            context_title=config.context_title,
            context_description=config.context_description
        )
    
    def _generate_synthetic(
        self, n: int, noise: float, intercept: float, slope: float
    ) -> DataResult:
        """Generiert rein synthetische Daten zu Demonstrationszwecken."""
        x = np.random.uniform(0, 100, n)
        y = intercept + slope * x + np.random.normal(0, noise, n)
        
        config = get_simple_config("synthetic")
        return DataResult(
            x=x, y=y,
            x_label=config.x_label,
            y_label=config.y_label,
            context_title=config.context_title,
            context_description=config.context_description
        )
    
    def _generate_cities(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """Generiert Daten für eine Städte-Marketingstudie (Preis, Werbung -> Umsatz)."""
        x1 = np.random.normal(5.69, 0.52, n)
        x1 = np.clip(x1, 4.5, 7.0) # Preis in CHF
        
        x2 = np.random.normal(1.84, 0.83, n)
        x2 = np.clip(x2, 0.5, 3.5) # Werbung in 1000 CHF
        
        # Umsatz = 120 - 8*Preis + 4*Werbung + noise
        y = 120 - 8 * x1 + 4 * x2 + np.random.normal(0, noise, n)
        
        config = get_multiple_config("cities")
        return MultipleRegressionDataResult(
            x1=x1, x2=x2, y=y,
            x1_label=config.x1_label,
            x2_label=config.x2_label,
            y_label=config.y_label,
            extra={"true_b0": 120, "true_b1": -8, "true_b2": 4}
        )
    
    def _generate_houses(self, n: int, noise: float) -> MultipleRegressionDataResult:
        """Generiert Daten für Immobilienpreise (Wohnfläche, Pool -> Preis)."""
        x1 = np.random.normal(25.21, 2.92, n)
        x1 = np.clip(x1, 18, 35) # Wohnfläche (Einheit sqft/10)
        
        # Dummy-Variable für Pool (0 = Nein, 1 = Ja)
        x2 = (np.random.random(n) < 0.204).astype(float)
        
        y = 50 + 8 * x1 + 30 * x2 + np.random.normal(0, noise, n)
        
        config = get_multiple_config("houses")
        return MultipleRegressionDataResult(
            x1=x1, x2=x2, y=y,
            x1_label=config.x1_label,
            x2_label=config.x2_label,
            y_label=config.y_label,
            extra={"true_b0": 50, "true_b1": 8, "true_b2": 30}
        )
    
    # =========================================================
    # KLASSIFIKATIONS-DATENSÄTZE
    # =========================================================
    
    def get_classification(
        self,
        dataset: str,
        n: int = 59,
        seed: int = 42,
    ) -> ClassificationDataResult:
        """
        Beschafft Daten für Klassifikationsaufgaben (KNN, Logistische Regression).
        """
        logger.info(f"Abruf von Klassifikationsdaten: {dataset}, n={n}")
        np.random.seed(seed)
        
        if dataset == "fruits":
            return self._generate_fruits(n)
        elif dataset == "digits":
            return self._generate_digits(n)
        elif dataset == "binary_electronics":
            return self._generate_binary_from_simple("electronics", n, seed)
        elif dataset == "binary_housing":
            return self._generate_binary_from_simple("houses", n, seed)
        
        # Rückfall: Konvertierung von Regressionsdaten in binäre Klassifikation (High vs Low)
        try:
            reg_data = self.get_multiple(dataset, n=n, seed=seed)
            if reg_data:
                 threshold = np.median(reg_data.y)
                 y_binary = (reg_data.y > threshold).astype(int)
                 return ClassificationDataResult(
                    X=np.column_stack([reg_data.x1, reg_data.x2]), 
                    y=y_binary,
                    feature_names=[reg_data.x1_label, reg_data.x2_label],
                    target_names=["Niedrig " + reg_data.y_label.split('(')[0].strip(), "Hoch " + reg_data.y_label.split('(')[0].strip()],
                    context_title=reg_data.extra.get("context", dataset) + " (Binär)",
                    context_description=f"Vorhersage Hoch/Niedrig {reg_data.y_label} basierend auf {reg_data.x1_label} & {reg_data.x2_label}"
                 )
        except Exception:
            pass
            
        return self._generate_fruits(n)
    
    def _generate_fruits(self, n: int) -> ClassificationDataResult:
        """
        Früchte-Datensatz (KNN-Fallbeispiel aus der Vorlesung).
        Merkmale: Höhe, Breite, Masse, Farbskala. Klassen: Apfel, Mandarine, Orange, Zitrone.
        """
        n_per_class = n // 4
        classes, heights, widths, masses, colors = [], [], [], [], []
        
        # Äpfel: rund, mittelgroß, eher rot
        for _ in range(n_per_class):
            classes.append(0)
            heights.append(np.random.normal(7.5, 0.8))
            widths.append(np.random.normal(7.3, 0.7))
            masses.append(np.random.normal(175, 25))
            colors.append(np.random.normal(0.75, 0.1))
        
        # Mandarinen: klein, rundlich, orange
        for _ in range(n_per_class):
            classes.append(1)
            heights.append(np.random.normal(4.5, 0.5))
            widths.append(np.random.normal(5.8, 0.4))
            masses.append(np.random.normal(85, 15))
            colors.append(np.random.normal(0.82, 0.08))
        
        # Orangen: rund, größer
        for _ in range(n_per_class):
            classes.append(2)
            heights.append(np.random.normal(7.0, 0.6))
            widths.append(np.random.normal(7.2, 0.5))
            masses.append(np.random.normal(155, 20))
            colors.append(np.random.normal(0.78, 0.07))
        
        # Zitronen: länglich, gelb
        for _ in range(n - 3 * n_per_class):
            classes.append(3)
            heights.append(np.random.normal(8.5, 0.9))
            widths.append(np.random.normal(5.5, 0.6))
            masses.append(np.random.normal(120, 20))
            colors.append(np.random.normal(0.70, 0.1))
        
        X = np.column_stack([heights, widths, masses, colors])
        y = np.array(classes)
        
        idx = np.random.permutation(len(y))
        
        return ClassificationDataResult(
            X=X[idx], y=y[idx],
            feature_names=["height", "width", "mass", "color_score"],
            target_names=["apple", "mandarin", "orange", "lemon"],
            context_title="🍎 Früchte-Klassifikation",
            context_description="KNN Fallbeispiel: Klassifizierung von Früchten anhand physischer Merkmale",
            extra={"source": "Professor's Lecture"}
        )
    
    def _generate_digits(self, n: int) -> ClassificationDataResult:
        """Handgeschriebene Ziffern (8x8 Pixel, Fallbeispiel aus der Vorlesung)."""
        n_per_class = max(1, n // 10)
        X_list, y_list = [], []
        
        for digit in range(10):
            for _ in range(n_per_class if digit < 9 else n - 9 * n_per_class):
                img = np.zeros((8, 8))
                
                # Vereinfachte Muster für Ziffern
                if digit == 0:
                    img[1:7, 2:6] = np.random.uniform(8, 16, (6, 4))
                    img[2:6, 3:5] = 0
                elif digit == 1:
                    img[1:7, 3:5] = np.random.uniform(10, 16, (6, 2))
                elif digit == 2:
                    img[1:3, 2:6] = np.random.uniform(8, 14, (2, 4))
                    img[5:7, 2:6] = np.random.uniform(8, 14, (2, 4))
                elif digit == 3:
                     img[1:2, 2:6] = np.random.uniform(8, 14, (1, 4))
                     img[3:4, 2:6] = np.random.uniform(8, 14, (1, 4))
                     img[6:7, 2:6] = np.random.uniform(8, 14, (1, 4))
                else:
                    img[digit % 3:(digit % 3) + 4, 2:6] = np.random.uniform(8, 16, (4, 4))
                
                img += np.random.uniform(0, 2, (8, 8))
                X_list.append(img.flatten())
                y_list.append(digit)
        
        X, y = np.array(X_list), np.array(y_list)
        idx = np.random.permutation(len(y))
        
        return ClassificationDataResult(
            X=X[idx], y=y[idx],
            feature_names=[f"pixel_{i}" for i in range(64)],
            target_names=[str(d) for d in range(10)],
            context_title="🔢 Handgeschriebene Ziffern",
            context_description="Ziffern Fallbeispiel: Klassifizierung von 8x8 Pixel-Bildern (0-9)",
            extra={"image_shape": (8, 8)}
        )
    
    def _generate_binary_from_simple(self, base: str, n: int, seed: int) -> ClassificationDataResult:
        """Erzeugt eine binäre Klassifikation aus Regressionsdaten (Umsatz hoch/niedrig)."""
        if base == "electronics":
            data = self.get_simple("electronics", n=n, seed=seed)
            y_binary = (data.y > np.median(data.y)).astype(int)
            return ClassificationDataResult(
                X=data.x.reshape(-1, 1), y=y_binary,
                feature_names=[data.x_label],
                target_names=["low_sales", "high_sales"],
                context_title="🏪 Elektronikmarkt (Binär)",
                context_description="Vorhersage hoher/niedriger Umsatz basierend auf Ladengröße"
            )
        else:
            data = self.get_multiple("houses", n=n, seed=seed)
            y_binary = (data.y > np.median(data.y)).astype(int)
            return ClassificationDataResult(
                X=np.column_stack([data.x1, data.x2]), y=y_binary,
                feature_names=[data.x1_label, data.x2_label],
                target_names=["standard", "premium"],
                context_title="🏠 Immobilien (Binär)",
                context_description="Vorhersage Premium- oder Standard-Immobilien"
            )
