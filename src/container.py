"""
Dependency Injection Container.
Wires up all implementations to interfaces.
This is the ONLY place where concrete classes are instantiated.
"""
from src.core.application import RunRegressionUseCase
from src.infrastructure import DataProviderImpl, RegressionServiceImpl


class Container:
    """
    Simple DI Container for the application.
    Provides configured Use Cases with injected dependencies.
    """
    
    def __init__(self):
        # Infrastructure implementations
        self._data_provider = DataProviderImpl()
        self._regression_service = RegressionServiceImpl()
    
    @property
    def run_regression_use_case(self) -> RunRegressionUseCase:
        """Get configured RunRegressionUseCase."""
        return RunRegressionUseCase(
            data_provider=self._data_provider,
            regression_service=self._regression_service
        )


# Singleton instance
_container = None

def get_container() -> Container:
    """Get or create the DI container singleton."""
    global _container
    if _container is None:
        _container = Container()
    return _container
