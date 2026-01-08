"""
Infrastructure: Implementierung des Data Providers.
Implementiert IDataProvider und nutzt den DataFetcher für die tatsächliche Datengenerierung.
"""
from typing import Dict, Any, List
from ...core.domain.interfaces import IDataProvider
from ...core.domain.value_objects import DatasetMetadata
from .generators import DataFetcher


class DataProviderImpl(IDataProvider):
    """Konkrete Implementierung von IDataProvider, die den DataFetcher kapselt."""
    
    def __init__(self):
        # Initialisierung des internen Fetchers (NumPy/Pandas basiert)
        self._fetcher = DataFetcher()
    
    def get_dataset(self, dataset_id: str, n: int, **kwargs) -> Dict[str, Any]:
        """
        Ruft Daten ab und konvertiert sie in ein Dictionary für den Use Case.
        Konvertiert NumPy-Arrays in Listen, um die Reinheit des Domain-Layers zu wahren.
        """
        noise = kwargs.get("noise", 0.4)
        seed = kwargs.get("seed", 42)
        true_intercept = kwargs.get("true_intercept", 0.6)
        true_slope = kwargs.get("true_slope", 0.52)
        regression_type = kwargs.get("regression_type", "simple")
        analysis_type = kwargs.get("analysis_type", "regression")
        
        # Fallunterscheidung: Klassifikation vs. Regression
        if analysis_type == "classification":
            result = self._fetcher.get_classification(dataset_id, n=n, seed=seed)
            return {
                "X": result.X.tolist(),
                "y": result.y.tolist(),
                "feature_names": result.feature_names,
                "target_names": result.target_names,
                "name": result.context_title, 
                "description": result.context_description,
                "extra": result.extra,
                "metadata": DatasetMetadata(
                    id=dataset_id,
                    name=result.context_title,
                    description=result.context_description,
                    source="generated",
                    variables=result.feature_names + ["Target"],
                    n_observations=result.n
                )
            }
        
        # Fallunterscheidung innerhalb der Regression: Multipel vs. Einfach
        if regression_type == "multiple":
            result = self._fetcher.get_multiple(dataset_id, n=n, noise=noise, seed=seed)
            return {
                "x1": result.x1.tolist(),
                "x2": result.x2.tolist(),
                "y": result.y.tolist(),
                "x1_label": result.x1_label,
                "x2_label": result.x2_label,
                "y_label": result.y_label,
                "context_title": result.extra.get("context", "Multiple Regression"),
                "context_description": f"Datensatz: {dataset_id}",
                "extra": result.extra,
                "metadata": DatasetMetadata(
                    id=dataset_id,
                    name=dataset_id,
                    description="",
                    source="generated",
                    variables=[result.x1_label, result.x2_label, result.y_label],
                    n_observations=result.n
                )
            }
        else:
            # Einfache lineare Regression
            result = self._fetcher.get_simple(
                dataset_id, n=n, noise=noise, seed=seed,
                true_intercept=true_intercept, true_slope=true_slope
            )
            return {
                "x": result.x.tolist(),
                "y": result.y.tolist(),
                "x_label": result.x_label,
                "y_label": result.y_label,
                "context_title": result.context_title,
                "context_description": result.context_description,
                "extra": result.extra,
                "metadata": DatasetMetadata(
                    id=dataset_id,
                    name=dataset_id,
                    description=result.context_description,
                    source="generated",
                    variables=[result.x_label, result.y_label],
                    n_observations=result.n
                )
            }
    
    def get_all_datasets(self) -> Dict[str, List[Dict[str, str]]]:
        """Gibt eine Liste aller verfügbaren Datensätze zurück, gruppiert nach Analyse-Typ."""
        datasets = {
            "simple": [
                {"id": "electronics", "name": "Elektronikmarkt (Umsatz)"},
                {"id": "advertising", "name": "Werbebudget (Umsatz)"},
                {"id": "temperature", "name": "Temperatur (Eisverkauf)"},
                {"id": "productivity", "name": "Produktivität"},
                {"id": "learning_time", "name": "Lernzeit"},
                {"id": "cantons", "name": "Schweizer Kantone (Einkommen)"},
                {"id": "linear_growth", "name": "Lineares Wachstum"},
                {"id": "exponential_growth", "name": "Exponentielles Wachstum"},
                {"id": "seasonal", "name": "Saisonale Daten"},
                {"id": "outliers", "name": "Ausreisser"},
                {"id": "heteroscedasticity", "name": "Heteroskedastizität"},
            ],
            "multiple": [
                {"id": "cities", "name": "Städte (Marketing)"},
                {"id": "houses", "name": "Immobilienpreise"},
                {"id": "startup", "name": "Startup Profit"},
                {"id": "quality_control", "name": "Qualitätskontrolle"},
                {"id": "car_prices", "name": "Autopreise"},
                {"id": "cantons", "name": "Schweizer Kantone (Multipel)"},
                {"id": "weather", "name": "Schweizer Wetter"},
                {"id": "world_bank", "name": "World Bank Dev"},
                {"id": "fred_economic", "name": "US Wirtschaft (FRED)"},
                {"id": "who_health", "name": "WHO Gesundheit"},
                {"id": "eurostat", "name": "Eurostat EU"},
                {"id": "nasa_weather", "name": "NASA Agrar-Klima"},
            ],
            "classification": [
                {"id": "fruits", "name": "Früchte (Logistic/KNN)"},
                {"id": "ad_click", "name": "Werbeklick (Logistic)"},
                {"id": "loan_approval", "name": "Kreditvergabe"},
                {"id": "iris", "name": "Iris Schwertlilien"},
                {"id": "wine", "name": "Weinqualität"},
                {"id": "cancer", "name": "Brustkrebs Diagnostik"},
            ]
        }
        return datasets

    def get_raw_data(self, dataset_id: str) -> Dict[str, Any]:
        """Gibt rohe tabellarische Daten für einen Datensatz zurück (für den Data Explorer)."""
        # Wir laden standardmäßig 100 Zeilen für die Vorschau
        n = 100
        
        # Logik zur Bestimmung des Datensatz-Typs anhand der ID
        if dataset_id in [d["id"] for d in self.get_all_datasets()["classification"]]:
             result = self._fetcher.get_classification(dataset_id, n=n)
             # Konvertierung in eine Liste von Dictionaries für die Tabellenansicht
             data = []
             for i in range(len(result.y)):
                 row = {name: result.X[i, j] for j, name in enumerate(result.feature_names)}
                 row["Target"] = int(result.y[i])
                 if result.target_names and int(result.y[i]) < len(result.target_names):
                     row["Target Name"] = result.target_names[int(result.y[i])]
                 data.append(row)
             return {"data": data, "columns": result.feature_names + ["Target", "Target Name"]}
             
        elif dataset_id in [d["id"] for d in self.get_all_datasets()["multiple"]]:
             result = self._fetcher.get_multiple(dataset_id, n=n)
             data = []
             for i in range(len(result.y)):
                 data.append({
                     result.x1_label: result.x1[i],
                     result.x2_label: result.x2[i],
                     result.y_label: result.y[i]
                 })
             return {"data": data, "columns": [result.x1_label, result.x2_label, result.y_label]}
             
        else:
             # Standard: Einfache Regression
             result = self._fetcher.get_simple(dataset_id, n=n)
             data = []
             for i in range(len(result.y)):
                 data.append({
                     result.x_label: result.x[i],
                     result.y_label: result.y[i]
                 })
             return {"data": data, "columns": [result.x_label, result.y_label]}

    def list_datasets(self) -> List[DatasetMetadata]:
        """Legacy-Methode (für Kompatibilität beibehalten)."""
        return []
