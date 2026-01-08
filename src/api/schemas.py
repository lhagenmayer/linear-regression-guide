from enum import Enum
from pydantic import BaseModel, Field, field_validator


class DatasetType(str, Enum):
    """Available dataset types."""
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
    """Base request parameters."""
    dataset: DatasetType = Field(default=DatasetType.ELECTRONICS, description="Dataset identifier")
    n: int = Field(default=50, ge=10, le=1000, description="Sample size (10-1000)")
    noise: float = Field(default=0.4, ge=0.0, le=100.0, description="Noise level")
    seed: int = Field(default=42, description="Random seed")
    include_predictions: bool = Field(default=True, description="Include predictions in response")

    @field_validator("n")
    def validate_n(cls, v):
        if v % 1 != 0:
            raise ValueError("n must be an integer")
        return v


class SimpleRegressionRequest(BaseRegressionRequest):
    """Request parameters for simple regression."""
    true_intercept: float = Field(default=0.6, description="True y-intercept (β₀)")
    true_slope: float = Field(default=0.52, description="True slope (β₁)")


class MultipleRegressionRequest(BaseRegressionRequest):
    """Request parameters for multiple regression."""
    # Inherits common fields, specific dataset validation can occur in endpoints if needed
    pass


class AIInterpretationRequest(BaseModel):
    """Request for AI interpretation."""
    stats: dict = Field(..., description="Statistical results dictionary")
    use_cache: bool = Field(default=True, description="Use cached interpretation if available")
