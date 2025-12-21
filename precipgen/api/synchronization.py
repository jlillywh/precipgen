"""
External simulation synchronization for PrecipGen library.

Provides capabilities to synchronize internal date tracking
with external simulation clocks and coordinate state management.
"""

from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ExternalSimulationSync:
    """
    Synchronization interface for external simulation environments.
    
    Manages coordination between PrecipGen internal state and
    external simulation clocks, ensuring consistent time tracking.
    """
    
    def __init__(self, 
                 time_step_callback: Optional[Callable[[datetime], None]] = None,
                 state_sync_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize external simulation synchronization.
        
        Args:
            time_step_callback: Callback function called on each time step
            state_sync_callback: Callback function for state synchronization
        """
        self.time_step_callback = time_step_callback
        self.state_sync_callback = state_sync_callback
        self.external_clock: Optional[datetime] = None
        self.internal_clock: Optional[datetime] = None
        self.sync_enabled = False
        self.sync_tolerance_seconds = 60  # 1 minute tolerance
        self.sync_history: List[Dict[str, Any]] = []
        self.max_history_length = 1000
    
    def enable_synchronization(self, 
                              external_start_time: datetime,
                              internal_start_time: Optional[datetime] = None) -> None:
        """
        Enable synchronization with external simulation clock.
        
        Args:
            external_start_time: Starting time from external simulation
            internal_start_time: Starting time for internal clock (defaults to external)
        """
        self.external_clock = external_start_time
        self.internal_clock = internal_start_time or external_start_time
        self.sync_enabled = True
        
        logger.info(f"Synchronization enabled: external={external_start_time.isoformat()}, "
                   f"internal={self.internal_clock.isoformat()}")
        
        self._record_sync_event("sync_enabled", {
            'external_time': external_start_time.isoformat(),
            'internal_time': self.internal_clock.isoformat()
        })
    
    def disable_synchronization(self) -> None:
        """Disable synchronization with external simulation."""
        self.sync_enabled = False
        logger.info("Synchronization disabled")
        
        self._record_sync_event("sync_disabled", {})
    
    def update_external_clock(self, external_time: datetime) -> bool:
        """
        Update external simulation clock and check synchronization.
        
        Args:
            external_time: Current time from external simulation
            
        Returns:
            True if synchronization is maintained, False if drift detected
        """
        if not self.sync_enabled:
            return True
        
        previous_external = self.external_clock
        self.external_clock = external_time
        
        # Calculate expected internal time based on external advancement
        if previous_external is not None:
            time_delta = external_time - previous_external
            expected_internal = self.internal_clock + time_delta
        else:
            expected_internal = external_time
        
        # Check for synchronization drift
        if self.internal_clock is not None:
            drift_seconds = abs((expected_internal - self.internal_clock).total_seconds())
            
            if drift_seconds > self.sync_tolerance_seconds:
                logger.warning(f"Clock drift detected: {drift_seconds:.1f} seconds")
                self._record_sync_event("drift_detected", {
                    'drift_seconds': drift_seconds,
                    'external_time': external_time.isoformat(),
                    'internal_time': self.internal_clock.isoformat(),
                    'expected_internal': expected_internal.isoformat()
                })
                return False
        
        # Call time step callback if provided
        if self.time_step_callback:
            try:
                self.time_step_callback(external_time)
            except Exception as e:
                logger.error(f"Time step callback failed: {str(e)}")
        
        return True
    
    def advance_internal_clock(self, time_delta: timedelta) -> datetime:
        """
        Advance internal clock by specified time delta.
        
        Args:
            time_delta: Time delta to advance
            
        Returns:
            New internal clock time
        """
        if self.internal_clock is None:
            raise RuntimeError("Internal clock not initialized")
        
        previous_internal = self.internal_clock
        self.internal_clock += time_delta
        
        logger.debug(f"Internal clock advanced: {previous_internal.isoformat()} -> "
                    f"{self.internal_clock.isoformat()}")
        
        # Check synchronization if enabled
        if self.sync_enabled and self.external_clock is not None:
            self.update_external_clock(self.external_clock)
        
        return self.internal_clock
    
    def synchronize_to_external(self) -> bool:
        """
        Force synchronization of internal clock to external clock.
        
        Returns:
            True if synchronization successful, False otherwise
        """
        if not self.sync_enabled or self.external_clock is None:
            return False
        
        previous_internal = self.internal_clock
        self.internal_clock = self.external_clock
        
        logger.info(f"Forced synchronization: {previous_internal.isoformat() if previous_internal else 'None'} -> "
                   f"{self.internal_clock.isoformat()}")
        
        self._record_sync_event("forced_sync", {
            'previous_internal': previous_internal.isoformat() if previous_internal else None,
            'new_internal': self.internal_clock.isoformat(),
            'external_time': self.external_clock.isoformat()
        })
        
        # Call state sync callback if provided
        if self.state_sync_callback:
            try:
                sync_state = self.get_sync_state()
                self.state_sync_callback(sync_state)
            except Exception as e:
                logger.error(f"State sync callback failed: {str(e)}")
        
        return True
    
    def get_sync_state(self) -> Dict[str, Any]:
        """
        Get current synchronization state.
        
        Returns:
            Dictionary containing synchronization state information
        """
        state = {
            'sync_enabled': self.sync_enabled,
            'external_clock': self.external_clock.isoformat() if self.external_clock else None,
            'internal_clock': self.internal_clock.isoformat() if self.internal_clock else None,
            'sync_tolerance_seconds': self.sync_tolerance_seconds,
            'has_time_step_callback': self.time_step_callback is not None,
            'has_state_sync_callback': self.state_sync_callback is not None
        }
        
        # Calculate current drift if both clocks are available
        if self.sync_enabled and self.external_clock and self.internal_clock:
            drift_seconds = (self.external_clock - self.internal_clock).total_seconds()
            state.update({
                'current_drift_seconds': drift_seconds,
                'is_synchronized': abs(drift_seconds) <= self.sync_tolerance_seconds
            })
        
        return state
    
    def get_sync_history(self, max_entries: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get synchronization event history.
        
        Args:
            max_entries: Maximum number of entries to return
            
        Returns:
            List of synchronization events
        """
        if max_entries is None:
            return self.sync_history.copy()
        else:
            return self.sync_history[-max_entries:]
    
    def clear_sync_history(self) -> None:
        """Clear synchronization event history."""
        self.sync_history.clear()
        logger.info("Synchronization history cleared")
    
    def set_sync_tolerance(self, tolerance_seconds: float) -> None:
        """
        Set synchronization tolerance.
        
        Args:
            tolerance_seconds: Maximum allowed drift in seconds
        """
        self.sync_tolerance_seconds = tolerance_seconds
        logger.info(f"Synchronization tolerance set to {tolerance_seconds} seconds")
    
    def validate_synchronization(self) -> Dict[str, Any]:
        """
        Validate current synchronization state.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        if self.sync_enabled:
            if self.external_clock is None:
                validation['is_valid'] = False
                validation['errors'].append("External clock not set")
            
            if self.internal_clock is None:
                validation['is_valid'] = False
                validation['errors'].append("Internal clock not set")
            
            if self.external_clock and self.internal_clock:
                drift_seconds = abs((self.external_clock - self.internal_clock).total_seconds())
                
                if drift_seconds > self.sync_tolerance_seconds:
                    validation['warnings'].append(
                        f"Clock drift ({drift_seconds:.1f}s) exceeds tolerance ({self.sync_tolerance_seconds}s)"
                    )
                
                validation['info'].append(f"Current drift: {drift_seconds:.1f} seconds")
        else:
            validation['info'].append("Synchronization disabled")
        
        return validation
    
    def _record_sync_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Record synchronization event in history."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'data': event_data
        }
        
        self.sync_history.append(event)
        
        # Limit history length
        if len(self.sync_history) > self.max_history_length:
            self.sync_history = self.sync_history[-self.max_history_length:]


class SimulationCoordinator:
    """
    High-level coordinator for managing multiple PrecipGen engines
    with external simulation synchronization.
    """
    
    def __init__(self):
        self.engines: Dict[str, Any] = {}
        self.sync_manager = ExternalSimulationSync()
        self.coordination_state: Dict[str, Any] = {}
    
    def register_engine(self, engine_id: str, engine: Any) -> None:
        """
        Register an engine for coordination.
        
        Args:
            engine_id: Unique identifier for the engine
            engine: Engine instance to register
        """
        self.engines[engine_id] = engine
        logger.info(f"Engine registered: {engine_id}")
    
    def unregister_engine(self, engine_id: str) -> bool:
        """
        Unregister an engine.
        
        Args:
            engine_id: Engine identifier to unregister
            
        Returns:
            True if engine was found and removed, False otherwise
        """
        if engine_id in self.engines:
            del self.engines[engine_id]
            logger.info(f"Engine unregistered: {engine_id}")
            return True
        return False
    
    def synchronize_all_engines(self, external_time: datetime) -> Dict[str, bool]:
        """
        Synchronize all registered engines to external time.
        
        Args:
            external_time: External simulation time
            
        Returns:
            Dictionary mapping engine_id to synchronization success
        """
        results = {}
        
        # Update sync manager
        self.sync_manager.update_external_clock(external_time)
        
        # Synchronize each engine
        for engine_id, engine in self.engines.items():
            try:
                if hasattr(engine, 'current_date'):
                    # Update engine's internal date
                    engine.current_date = external_time
                    results[engine_id] = True
                    logger.debug(f"Engine {engine_id} synchronized to {external_time.isoformat()}")
                else:
                    results[engine_id] = False
                    logger.warning(f"Engine {engine_id} does not support date synchronization")
            except Exception as e:
                results[engine_id] = False
                logger.error(f"Failed to synchronize engine {engine_id}: {str(e)}")
        
        return results
    
    def get_coordination_state(self) -> Dict[str, Any]:
        """
        Get current coordination state for all engines.
        
        Returns:
            Dictionary with coordination state information
        """
        state = {
            'registered_engines': list(self.engines.keys()),
            'engine_count': len(self.engines),
            'sync_state': self.sync_manager.get_sync_state(),
            'engine_states': {}
        }
        
        # Get state from each engine
        for engine_id, engine in self.engines.items():
            try:
                if hasattr(engine, 'get_state'):
                    state['engine_states'][engine_id] = engine.get_state()
                else:
                    state['engine_states'][engine_id] = {'status': 'no_state_method'}
            except Exception as e:
                state['engine_states'][engine_id] = {'status': 'error', 'error': str(e)}
        
        return state