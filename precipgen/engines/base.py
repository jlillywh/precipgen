"""
Abstract base classes for engines.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from datetime import datetime


class Engine(ABC):
    """
    Abstract base class for all engines.
    
    Defines the common interface that all engines must implement
    to maintain separation between algorithmic components.
    """
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """
        Initialize the engine with required parameters.
        
        Args:
            **kwargs: Engine-specific initialization parameters
        """
        pass
    
    @abstractmethod
    def reset(self, **kwargs) -> None:
        """
        Reset the engine to initial state.
        
        Args:
            **kwargs: Engine-specific reset parameters
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get current engine state for serialization/debugging.
        
        Returns:
            Dictionary containing current engine state
        """
        pass