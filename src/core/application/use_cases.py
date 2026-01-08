"""
Application Use Cases.
Orchestrate domain objects and infrastructure services using dependency injection.
"""
from typing import Dict, Any
from .dtos import RegressionRequestDTO, RegressionResponseDTO
from ..domain.interfaces import IDataProvider, IRegressionService
from ..domain.entities import RegressionModel
from ..domain.value_objects import RegressionParameters, RegressionMetrics

class RunRegressionUseCase:
    """Use Case: Run a regression analysis (Simple or Multiple)."""
    
    def __init__(self, data_provider: IDataProvider, regression_service: IRegressionService):
        self.data_provider = data_provider
        self.regression_service = regression_service
        
    def execute(self, request: RegressionRequestDTO) -> RegressionResponseDTO:
        # 1. Fetch Data via Interface
        data_result = self.data_provider.get_dataset(
            dataset_id=request.dataset_id,
            n=request.n_observations,
            noise=request.noise_level,
            seed=request.seed,
            true_intercept=request.true_intercept or 0.6,
            true_slope=request.true_slope or 0.52,
            regression_type=request.regression_type
        )
        
        # 2. Perform Regression via Service
        if request.regression_type == "multiple":
            # Extract x matrix (assuming x1, x2 are in data_result)
            # This logic depends on dictionary structure contract
            x_data = [data_result["x1"], data_result["x2"]] # Simplified for now
            y_data = data_result["y"]
            variable_names = [data_result["x1_label"], data_result["x2_label"]]
            
            model = self.regression_service.train_multiple(x_data, y_data, variable_names)
        else:
            # Simple
            x_data = [data_result["x"]] # Wrap in list for consistency if needed, or keeping it flat
            y_data = data_result["y"]
            
            model = self.regression_service.train_simple(data_result["x"], y_data)
        
        # 3. Add Metadata
        model.dataset_metadata = data_result.get("metadata")
        
        # 4. Construct Response DTO
        return self._build_response(model, data_result)

    def _build_response(self, model: RegressionModel, data_raw: Dict[str, Any]) -> RegressionResponseDTO:
        params = model.parameters
        metrics = model.metrics
        
        return RegressionResponseDTO(
            model_id=model.id,
            success=model.is_trained(),
            coefficients=params.coefficients,
            metrics={
                "r_squared": metrics.r_squared,
                "r_squared_adj": metrics.r_squared_adj,
                "mse": metrics.mse,
                "rmse": metrics.rmse,
                "f_statistic": metrics.f_statistic,
                "p_value": metrics.p_value
            },
            x_data=data_raw.get("x", []) or [data_raw.get("x1", []), data_raw.get("x2", [])],
            y_data=data_raw.get("y", []),
            residuals=model.residuals,
            predictions=model.predictions,
            x_label=data_raw.get("x_label") or f"{data_raw.get('x1_label')} & {data_raw.get('x2_label')}",
            y_label=data_raw.get("y_label", ""),
            title=data_raw.get("context_title", ""),
            description=data_raw.get("context_description", ""),
            extra=data_raw.get("extra", {})
        )
