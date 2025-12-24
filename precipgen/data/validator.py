"""
Data validation components for PrecipGen library.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from ..config.quality_config import QualityConfig
from ..utils.exceptions import (
    DataQualityError, PhysicalBoundsError, ValidationError,
    InsufficientDataError, create_error_context, handle_graceful_degradation
)
from ..utils.logging_config import get_logger


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


@dataclass 
class QualityReport:
    """Comprehensive data quality assessment."""
    completeness_percentage: float
    missing_data_count: int
    total_data_count: int
    consecutive_missing_max: int
    physical_bounds_violations: int
    quality_flag_issues: int
    time_period_years: float
    is_acceptable: bool
    issues: List[str]
    recommendations: List[str] = None
    
    def __post_init__(self):
        """Generate recommendations based on issues."""
        if self.recommendations is None:
            self.recommendations = self._generate_recommendations()
    
    def _generate_recommendations(self) -> List[str]:
        """Generate specific recommendations based on identified issues."""
        recommendations = []
        
        if self.completeness_percentage < 80:
            recommendations.append("Consider using a different time period with higher data completeness")
            recommendations.append("Implement gap-filling techniques for missing data")
        
        if self.consecutive_missing_max > 30:
            recommendations.append("Long data gaps detected - consider splitting analysis periods")
        
        if self.physical_bounds_violations > 0:
            recommendations.append("Review extreme values for data entry errors")
            recommendations.append("Consider adjusting physical bounds for extreme climate events")
        
        if self.time_period_years < 10:
            recommendations.append("Extend time period for more reliable parameter estimation")
        
        if self.quality_flag_issues > 0:
            flag_percentage = (self.quality_flag_issues / self.total_data_count) * 100
            if flag_percentage < 1.0:
                recommendations.append(f"Minor quality issues ({self.quality_flag_issues} records, {flag_percentage:.2f}%) - consider using 'lenient' quality preset")
            elif flag_percentage < 5.0:
                recommendations.append(f"Moderate quality issues ({self.quality_flag_issues} records, {flag_percentage:.2f}%) - review data or use 'lenient' quality preset")
            else:
                recommendations.append(f"Significant quality issues ({self.quality_flag_issues} records, {flag_percentage:.2f}%) - review data source reliability")
        
        return recommendations


class DataValidator:
    """
    Data validation and quality assessment for precipitation data.
    
    Implements configurable quality thresholds and comprehensive
    validation checks for precipitation time series with robust error handling.
    """
    
    def __init__(self, quality_config: QualityConfig = None):
        """
        Initialize data validator.
        
        Args:
            quality_config: Quality configuration parameters. If None, uses standard preset.
        """
        if quality_config is None:
            from ..config.quality_presets import QualityPresets
            quality_config = QualityPresets.standard()
            
        self.quality_config = quality_config
        self.logger = get_logger('data_validator')
        
        # Validation statistics for reporting
        self._validation_stats = {
            'total_validations': 0,
            'failed_validations': 0,
            'warnings_generated': 0
        }
    
    def validate_completeness(self, data: pd.Series, 
                            site_id: Optional[str] = None) -> ValidationResult:
        """
        Validate data completeness against configured thresholds.
        
        Args:
            data: Time series of precipitation data
            site_id: Optional site identifier for logging
            
        Returns:
            ValidationResult with completeness assessment
            
        Raises:
            ValidationError: If validation cannot be performed
        """
        self._validation_stats['total_validations'] += 1
        
        try:
            errors = []
            warnings = []
            metadata = {}
            
            if len(data) == 0:
                error_msg = "No data provided for completeness validation"
                self.logger.error(error_msg)
                raise InsufficientDataError(0, 1, parameter_type="completeness validation")
            
            # Calculate completeness statistics
            total_count = len(data)
            missing_count = data.isna().sum()
            completeness_pct = ((total_count - missing_count) / total_count) * 100
            
            metadata.update({
                'total_count': total_count,
                'missing_count': missing_count,
                'completeness_percentage': completeness_pct,
                'site_id': site_id
            })
            
            self.logger.debug(f"Completeness validation for {site_id or 'unknown site'}: "
                            f"{completeness_pct:.1f}% complete")
            
            # Check against threshold
            required_completeness = 100 - self.quality_config.max_missing_percentage
            if completeness_pct < required_completeness:
                error_msg = (f"Data completeness {completeness_pct:.1f}% below threshold "
                           f"{required_completeness:.1f}%")
                errors.append(error_msg)
                
                # Add guidance based on severity
                if completeness_pct < 50:
                    warnings.append("Very low data completeness - consider different data source")
                elif completeness_pct < 70:
                    warnings.append("Low data completeness - results may be unreliable")
            
            # Check consecutive missing data
            try:
                consecutive_missing = self._find_max_consecutive_missing(data)
                metadata['max_consecutive_missing'] = consecutive_missing
                
                if consecutive_missing > self.quality_config.max_consecutive_missing_days:
                    error_msg = (f"Maximum consecutive missing days {consecutive_missing} exceeds "
                               f"threshold {self.quality_config.max_consecutive_missing_days}")
                    errors.append(error_msg)
                    
                    if consecutive_missing > 365:
                        warnings.append("Very long data gaps detected - consider gap-filling methods")
                
            except Exception as e:
                warning_msg = f"Could not calculate consecutive missing days: {str(e)}"
                warnings.append(warning_msg)
                self.logger.warning(warning_msg)
                metadata['max_consecutive_missing'] = -1
            
            # Check time period length
            try:
                if hasattr(data.index, 'min') and hasattr(data.index, 'max'):
                    time_span = data.index.max() - data.index.min()
                    years = time_span.days / 365.25
                    metadata['time_period_years'] = years
                    
                    if years < self.quality_config.min_years_required:
                        error_msg = (f"Time period {years:.1f} years below minimum "
                                   f"{self.quality_config.min_years_required} years")
                        errors.append(error_msg)
                        
                        if years < 5:
                            warnings.append("Very short time period - parameter estimates may be unstable")
                else:
                    warnings.append("Cannot determine time period from data index")
                    metadata['time_period_years'] = 0
                    
            except Exception as e:
                warning_msg = f"Could not calculate time period: {str(e)}"
                warnings.append(warning_msg)
                self.logger.warning(warning_msg)
                metadata['time_period_years'] = 0
            
            is_valid = len(errors) == 0
            
            if not is_valid:
                self._validation_stats['failed_validations'] += 1
            
            if warnings:
                self._validation_stats['warnings_generated'] += len(warnings)
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                metadata=metadata
            )
            
        except Exception as e:
            context = create_error_context(
                'completeness_validation',
                site_id=site_id,
                data_length=len(data) if data is not None else 0
            )
            self.logger.error(f"Completeness validation failed: {str(e)}", 
                            extra={'context': context})
            
            # Return graceful degradation result
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation failed: {str(e)}"],
                warnings=["Using fallback validation due to error"],
                metadata={'error': str(e)}
            )
    
    def validate_physical_bounds(self, data: pd.Series,
                               site_id: Optional[str] = None) -> ValidationResult:
        """
        Validate precipitation values against physical bounds.
        
        Args:
            data: Time series of precipitation data
            site_id: Optional site identifier for logging
            
        Returns:
            ValidationResult with bounds validation
        """
        try:
            errors = []
            warnings = []
            metadata = {}
            
            # Remove missing values for bounds checking
            valid_data = data.dropna()
            
            if len(valid_data) == 0:
                error_msg = "No valid data available for bounds checking"
                self.logger.warning(error_msg)
                return ValidationResult(
                    is_valid=False,
                    errors=[error_msg],
                    warnings=["Consider checking data loading process"],
                    metadata={'valid_data_count': 0}
                )
            
            self.logger.debug(f"Physical bounds validation for {site_id or 'unknown site'}: "
                            f"{len(valid_data)} valid data points")
            
            # Check minimum bounds
            below_min = (valid_data < self.quality_config.physical_bounds_min).sum()
            metadata['below_minimum_count'] = below_min
            
            if below_min > 0:
                error_msg = (f"{below_min} values below physical minimum "
                           f"{self.quality_config.physical_bounds_min} mm")
                errors.append(error_msg)
                
                # Log extreme negative values
                extreme_negatives = valid_data[valid_data < -1.0]
                if len(extreme_negatives) > 0:
                    warnings.append(f"Extreme negative values detected: min = {extreme_negatives.min():.2f} mm")
            
            # Check maximum bounds
            above_max = (valid_data > self.quality_config.physical_bounds_max).sum()
            metadata['above_maximum_count'] = above_max
            
            if above_max > 0:
                warning_msg = (f"{above_max} values above physical maximum "
                             f"{self.quality_config.physical_bounds_max} mm (may be extreme events)")
                warnings.append(warning_msg)
                
                # Log the most extreme values
                extreme_values = valid_data[valid_data > self.quality_config.physical_bounds_max]
                if len(extreme_values) > 0:
                    max_extreme = extreme_values.max()
                    warnings.append(f"Maximum extreme value: {max_extreme:.2f} mm")
                    
                    if max_extreme > 1000:  # > 1 meter of rain in a day
                        warnings.append("Extremely high precipitation detected - verify data accuracy")
            
            # Statistical outlier detection
            try:
                q99 = valid_data.quantile(0.99)
                extreme_count = (valid_data > q99 * 3).sum()  # Values > 3x 99th percentile
                metadata['extreme_outlier_count'] = extreme_count
                
                if extreme_count > 0:
                    warnings.append(f"{extreme_count} potential extreme outliers detected")
                    
                    if extreme_count > len(valid_data) * 0.01:  # > 1% of data
                        warnings.append("High number of outliers - review data quality")
                
            except Exception as e:
                self.logger.warning(f"Could not perform outlier detection: {str(e)}")
                metadata['extreme_outlier_count'] = -1
            
            # Basic statistics
            try:
                metadata.update({
                    'min_value': float(valid_data.min()),
                    'max_value': float(valid_data.max()),
                    'mean_value': float(valid_data.mean()),
                    'median_value': float(valid_data.median()),
                    'std_value': float(valid_data.std())
                })
                
                # Check for suspicious patterns
                if valid_data.std() == 0:
                    warnings.append("All precipitation values are identical - check data source")
                elif valid_data.std() < 0.1:
                    warnings.append("Very low precipitation variability detected")
                
            except Exception as e:
                self.logger.warning(f"Could not calculate basic statistics: {str(e)}")
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                metadata=metadata
            )
            
        except Exception as e:
            context = create_error_context(
                'physical_bounds_validation',
                site_id=site_id,
                bounds_min=self.quality_config.physical_bounds_min,
                bounds_max=self.quality_config.physical_bounds_max
            )
            self.logger.error(f"Physical bounds validation failed: {str(e)}", 
                            extra={'context': context})
            
            return ValidationResult(
                is_valid=False,
                errors=[f"Bounds validation failed: {str(e)}"],
                warnings=["Could not validate physical bounds"],
                metadata={'error': str(e)}
            )
    
    def assess_data_quality(self, data: pd.Series, 
                           quality_flags: pd.Series = None,
                           site_id: Optional[str] = None) -> QualityReport:
        """
        Perform comprehensive data quality assessment.
        
        Args:
            data: Time series of precipitation data
            quality_flags: Optional quality flags for each data point
            site_id: Optional site identifier for logging
            
        Returns:
            QualityReport with comprehensive assessment
        """
        try:
            self.logger.info(f"Starting comprehensive quality assessment for {site_id or 'unknown site'}")
            
            # Basic completeness metrics
            total_count = len(data)
            missing_count = data.isna().sum()
            completeness_pct = ((total_count - missing_count) / total_count) * 100 if total_count > 0 else 0
            
            # Consecutive missing data
            consecutive_missing_max = self._find_max_consecutive_missing(data)
            
            # Physical bounds violations
            valid_data = data.dropna()
            bounds_violations = 0
            if len(valid_data) > 0:
                bounds_violations = (
                    (valid_data < self.quality_config.physical_bounds_min).sum() +
                    (valid_data > self.quality_config.physical_bounds_max).sum()
                )
            
            # Quality flag issues
            quality_flag_issues = 0
            if quality_flags is not None:
                try:
                    quality_flag_issues = quality_flags.isin(
                        self.quality_config.quality_flags_to_reject
                    ).sum()
                except Exception as e:
                    self.logger.warning(f"Could not assess quality flags: {str(e)}")
            
            # Time period calculation
            time_period_years = 0
            try:
                if hasattr(data.index, 'min') and hasattr(data.index, 'max') and len(data) > 0:
                    time_span = data.index.max() - data.index.min()
                    time_period_years = time_span.days / 365.25
            except Exception as e:
                self.logger.warning(f"Could not calculate time period: {str(e)}")
            
            # Determine acceptability and issues
            issues = []
            
            required_completeness = 100 - self.quality_config.max_missing_percentage
            if completeness_pct < required_completeness:
                issues.append(f"Low completeness: {completeness_pct:.1f}%")
            
            if consecutive_missing_max > self.quality_config.max_consecutive_missing_days:
                issues.append(f"Long gaps: {consecutive_missing_max} days")
            
            if time_period_years < self.quality_config.min_years_required:
                issues.append(f"Short period: {time_period_years:.1f} years")
            
            if bounds_violations > 0:
                issues.append(f"Bounds violations: {bounds_violations}")
            
            if quality_flag_issues > 0:
                issues.append(f"Quality flag issues: {quality_flag_issues}")
            
            # Additional quality checks
            if len(valid_data) > 0:
                # Check for unrealistic patterns
                zero_precip_pct = (valid_data == 0).sum() / len(valid_data) * 100
                if zero_precip_pct > 95:
                    issues.append(f"Excessive zero precipitation: {zero_precip_pct:.1f}%")
                elif zero_precip_pct < 10:
                    issues.append(f"Unusually low zero precipitation: {zero_precip_pct:.1f}%")
                
                # Check for repeated values (potential data quality issue)
                value_counts = valid_data.value_counts()
                if len(value_counts) > 0:
                    most_common_count = value_counts.iloc[0]
                    if most_common_count > len(valid_data) * 0.5 and value_counts.index[0] != 0:
                        issues.append(f"Repeated non-zero value: {most_common_count} occurrences")
            
            is_acceptable = len(issues) == 0
            
            quality_report = QualityReport(
                completeness_percentage=completeness_pct,
                missing_data_count=missing_count,
                total_data_count=total_count,
                consecutive_missing_max=consecutive_missing_max,
                physical_bounds_violations=bounds_violations,
                quality_flag_issues=quality_flag_issues,
                time_period_years=time_period_years,
                is_acceptable=is_acceptable,
                issues=issues
            )
            
            # Log quality assessment results
            if is_acceptable:
                self.logger.info(f"Data quality assessment passed for {site_id or 'unknown site'}")
            else:
                self.logger.warning(f"Data quality issues found for {site_id or 'unknown site'}: {', '.join(issues)}")
            
            return quality_report
            
        except Exception as e:
            context = create_error_context(
                'quality_assessment',
                site_id=site_id,
                data_length=len(data) if data is not None else 0
            )
            self.logger.error(f"Quality assessment failed: {str(e)}", extra={'context': context})
            
            # Return degraded quality report
            return QualityReport(
                completeness_percentage=0,
                missing_data_count=len(data) if data is not None else 0,
                total_data_count=len(data) if data is not None else 0,
                consecutive_missing_max=0,
                physical_bounds_violations=0,
                quality_flag_issues=0,
                time_period_years=0,
                is_acceptable=False,
                issues=[f"Quality assessment failed: {str(e)}"],
                recommendations=["Review data loading and validation process"]
            )
    
    def assess_with_fallback(self, data: pd.Series, 
                           quality_flags: pd.Series = None,
                           site_id: Optional[str] = None,
                           quality_levels: List[str] = None) -> QualityReport:
        """
        Assess data quality with automatic fallback to more lenient standards.
        
        Tries quality levels in order until data passes or all levels are exhausted.
        
        Args:
            data: Time series of precipitation data
            quality_flags: Optional quality flags for each data point
            site_id: Optional site identifier for logging
            quality_levels: Quality levels to try in order. Defaults to ['standard', 'lenient', 'permissive']
            
        Returns:
            QualityReport from the first passing quality level, or the most lenient attempt
        """
        from ..config.quality_presets import QualityPresets
        
        if quality_levels is None:
            quality_levels = ['standard', 'lenient', 'permissive']
        
        last_report = None
        
        for level in quality_levels:
            self.logger.info(f"Trying quality level: {level}")
            
            # Create validator with this quality level
            temp_config = QualityPresets.get_preset(level)
            temp_validator = DataValidator(temp_config)
            
            # Assess quality
            report = temp_validator.assess_data_quality(data, quality_flags, site_id)
            last_report = report
            
            if report.is_acceptable:
                self.logger.info(f"Data acceptable with '{level}' quality standards")
                # Add info about which level was used
                report.recommendations.insert(0, f"Data passed with '{level}' quality standards")
                return report
            else:
                self.logger.info(f"Data not acceptable with '{level}' quality standards")
        
        # If we get here, no quality level passed
        self.logger.warning(f"Data did not pass any quality level. Using most lenient results.")
        if last_report:
            last_report.recommendations.insert(0, "Data failed all quality levels - review data source or use custom quality configuration")
        
        return last_report or self.assess_data_quality(data, quality_flags, site_id)
    
    def validate_time_series_structure(self, data: pd.Series,
                                     expected_frequency: Optional[str] = 'D') -> ValidationResult:
        """
        Validate time series structure and frequency.
        
        Args:
            data: Time series data
            expected_frequency: Expected frequency ('D' for daily)
            
        Returns:
            ValidationResult with structure validation
        """
        try:
            errors = []
            warnings = []
            metadata = {}
            
            if not isinstance(data.index, pd.DatetimeIndex):
                errors.append("Data index is not a DatetimeIndex")
                return ValidationResult(False, errors, warnings, metadata)
            
            # Check for duplicate dates
            duplicates = data.index.duplicated().sum()
            metadata['duplicate_dates'] = duplicates
            
            if duplicates > 0:
                errors.append(f"{duplicates} duplicate dates found in time series")
            
            # Check frequency consistency
            if expected_frequency and len(data) > 1:
                try:
                    inferred_freq = pd.infer_freq(data.index)
                    metadata['inferred_frequency'] = inferred_freq
                    
                    if inferred_freq != expected_frequency:
                        if inferred_freq is None:
                            warnings.append("Could not infer consistent frequency from data")
                        else:
                            warnings.append(f"Inferred frequency {inferred_freq} differs from expected {expected_frequency}")
                
                except Exception as e:
                    warnings.append(f"Could not infer frequency: {str(e)}")
            
            # Check for large gaps in time series
            if len(data) > 1:
                time_diffs = data.index.to_series().diff().dropna()
                if expected_frequency == 'D':
                    expected_diff = pd.Timedelta(days=1)
                    large_gaps = (time_diffs > expected_diff * 7).sum()  # Gaps > 1 week
                    
                    metadata['large_gaps'] = large_gaps
                    if large_gaps > 0:
                        warnings.append(f"{large_gaps} large time gaps (>1 week) detected")
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Time series structure validation failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Structure validation failed: {str(e)}"],
                warnings=[],
                metadata={'error': str(e)}
            )
    
    def _find_max_consecutive_missing(self, data: pd.Series) -> int:
        """
        Find maximum consecutive missing values in time series.
        
        Args:
            data: Time series data
            
        Returns:
            Maximum consecutive missing count
        """
        if len(data) == 0:
            return 0
        
        try:
            is_missing = data.isna()
            consecutive_counts = []
            current_count = 0
            
            for missing in is_missing:
                if missing:
                    current_count += 1
                else:
                    if current_count > 0:
                        consecutive_counts.append(current_count)
                    current_count = 0
            
            # Don't forget the last sequence if it ends with missing values
            if current_count > 0:
                consecutive_counts.append(current_count)
            
            return max(consecutive_counts) if consecutive_counts else 0
            
        except Exception as e:
            self.logger.warning(f"Error calculating consecutive missing values: {str(e)}")
            return 0
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """
        Get validation statistics for monitoring and debugging.
        
        Returns:
            Dictionary with validation statistics
        """
        return self._validation_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset validation statistics."""
        self._validation_stats = {
            'total_validations': 0,
            'failed_validations': 0,
            'warnings_generated': 0
        }