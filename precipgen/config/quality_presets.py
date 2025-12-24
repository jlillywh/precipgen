"""
Quality configuration presets for different use cases.

Provides pre-configured quality settings for common scenarios,
eliminating the need for users to understand GHCN quality flags.
"""

from .quality_config import QualityConfig
from typing import Dict


class QualityPresets:
    """Pre-configured quality settings for different use cases."""
    
    @staticmethod
    def strict() -> QualityConfig:
        """
        Strict quality requirements for research-grade analysis.
        
        Rejects any data with quality flags indicating potential issues.
        Use for: Academic research, climate studies, official reports.
        """
        return QualityConfig(
            max_missing_percentage=5.0,
            min_years_required=15,
            max_consecutive_missing_days=7,
            quality_flags_to_reject=['X', 'W', 'I', 'O', 'Z']
        )
    
    @staticmethod
    def standard() -> QualityConfig:
        """
        Standard quality requirements for most applications.
        
        Accepts minor quality issues but rejects major problems.
        Use for: Engineering applications, general analysis, planning.
        """
        return QualityConfig(
            max_missing_percentage=10.0,
            min_years_required=10,
            max_consecutive_missing_days=30,
            quality_flags_to_reject=['X', 'W']  # Default behavior
        )
    
    @staticmethod
    def lenient() -> QualityConfig:
        """
        Lenient quality requirements for exploratory analysis.
        
        Accepts most data with quality flags, only rejects severe issues.
        Use for: Preliminary analysis, data exploration, proof of concept.
        """
        return QualityConfig(
            max_missing_percentage=20.0,
            min_years_required=5,
            max_consecutive_missing_days=90,
            quality_flags_to_reject=['X']  # Only reject failed bounds checks
        )
    
    @staticmethod
    def permissive() -> QualityConfig:
        """
        Very permissive quality requirements.
        
        Accepts all data regardless of quality flags.
        Use for: Data availability assessment, historical reconstruction.
        """
        return QualityConfig(
            max_missing_percentage=50.0,
            min_years_required=1,
            max_consecutive_missing_days=365,
            quality_flags_to_reject=[]  # Accept all quality flags
        )
    
    @staticmethod
    def get_preset(level: str) -> QualityConfig:
        """
        Get a quality preset by name.
        
        Args:
            level: Quality level ('strict', 'standard', 'lenient', 'permissive')
            
        Returns:
            QualityConfig for the specified level
            
        Raises:
            ValueError: If level is not recognized
        """
        presets = {
            'strict': QualityPresets.strict,
            'standard': QualityPresets.standard,
            'lenient': QualityPresets.lenient,
            'permissive': QualityPresets.permissive
        }
        
        if level not in presets:
            available = ', '.join(presets.keys())
            raise ValueError(f"Unknown quality level '{level}'. Available: {available}")
        
        return presets[level]()
    
    @staticmethod
    def list_presets() -> Dict[str, str]:
        """
        List available quality presets with descriptions.
        
        Returns:
            Dictionary mapping preset names to descriptions
        """
        return {
            'strict': 'Research-grade quality (rejects most flagged data)',
            'standard': 'Standard quality (default, rejects major issues)',
            'lenient': 'Lenient quality (accepts minor issues)',
            'permissive': 'Very permissive (accepts all data)'
        }