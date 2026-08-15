# src/evaluation/constraints.py
"""
Constraint checker for blade optimization.
"""

import numpy as np


class ConstraintChecker:
    """Check design constraints and compute penalties."""
    
    def __init__(self, R: float, max_thrust: float = 2000, max_flap_moment: float = 1500):
        self.R = R
        self.max_thrust = max_thrust
        self.max_flap_moment = max_flap_moment
    
    def check_all(self, geometry_data, power_curve, wind_speeds):
        """
        Check all constraints.
        
        Returns
        -------
        list of violation messages
        """
        violations = []
        
        # 1. Geometric constraints
        chord = geometry_data['chord']
        twist = geometry_data['twist']
        
        if np.any(chord < 0.05):
            violations.append("Chord below 0.05m")
        if np.any(chord > 0.60):
            violations.append("Chord above 0.60m")
        if np.any(twist < 0):
            violations.append("Twist below 0°")
        if np.any(twist > 35):
            violations.append("Twist above 35°")
        
        # 2. Aerodynamic constraints (if power_curve provided)
        if power_curve is not None and len(power_curve) > 0:
            # Check if power is reasonable at rated wind speed (8 m/s)
            # Find index closest to 8 m/s
            v_rated_idx = np.argmin(np.abs(wind_speeds - 8.0))
            power_at_rated = power_curve[v_rated_idx]
            if power_at_rated < 5000:  # 5 kW minimum
                violations.append(f"Power at 8 m/s is too low: {power_at_rated/1000:.2f} kW")
        
        return violations
    
    def compute_penalty(self, violations, weight=1e6):
        """Compute penalty from violations."""
        return len(violations) * weight