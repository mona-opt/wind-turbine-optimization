# src/aero/airfoil_database.py
"""
Airfoil database for NACA 4412 with 2D interpolation.
Supports variable Reynolds number and angle of attack.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional


class AirfoilDatabase:
    """
    Database for NACA 4412 airfoil coefficients.
    Uses linear interpolation in both alpha and Re dimensions.
    
    Usage:
        db = AirfoilDatabase()
        cl, cd = db.get_coeffs(alpha=6.0, Re=500000)
    """
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        Parameters
        ----------
        csv_path : str, optional
            Path to CSV file with columns: alpha, Re, cl, cd
            If None, uses built-in mock data for testing.
        """
        if csv_path is not None:
            self._load_from_csv(csv_path)
        else:
            self._generate_mock_data()
    
    def _generate_mock_data(self):
        """
        Generate realistic mock data for NACA 4412.
        Based on typical wind turbine airfoil behavior.
        """
        self.Re_values = np.array([300000, 500000, 1000000])
        self.alpha_values = np.linspace(-5, 20, 15)
        
        self.cl_data = {}
        self.cd_data = {}
        
        for Re in self.Re_values:
            alpha = self.alpha_values
            # CL: linear until stall, then drop
            cl = 0.11 * alpha + 0.35
            # Stall at ~14 degrees
            stall_idx = np.where(alpha > 14)[0]
            if len(stall_idx) > 0:
                cl[stall_idx] = cl[stall_idx[0]-1] * np.exp(-0.05 * (alpha[stall_idx] - 14))
            # Clamp
            cl = np.clip(cl, -0.5, 1.6)
            
            # CD: parabolic with minimum at zero lift
            cd = 0.008 + 0.00008 * (alpha + 2)**2
            cd = np.clip(cd, 0.006, 0.035)
            
            self.cl_data[Re] = cl
            self.cd_data[Re] = cd
    
    def _load_from_csv(self, csv_path: str):
        """Load data from CSV file."""
        df = pd.read_csv(csv_path)
        self.Re_values = np.sort(df['Re'].unique())
        self.alpha_values = np.sort(df['alpha'].unique())
        self.cl_data = {}
        self.cd_data = {}
        for Re in self.Re_values:
            mask = df['Re'] == Re
            self.cl_data[Re] = df[mask]['cl'].values
            self.cd_data[Re] = df[mask]['cd'].values
    
    def get_coeffs(self, alpha: float, Re: float) -> Tuple[float, float]:
        """
        Get CL and CD for given alpha (degrees) and Reynolds number.
        
        Returns
        -------
        tuple (cl, cd)
        """
        # Clamp alpha and Re to valid range
        alpha = np.clip(alpha, self.alpha_values.min(), self.alpha_values.max())
        Re = np.clip(Re, self.Re_values.min(), self.Re_values.max())
        
        # Find nearest Re values
        Re_low = self.Re_values[self.Re_values <= Re]
        Re_high = self.Re_values[self.Re_values >= Re]
        
        if len(Re_low) == 0:
            Re_low = self.Re_values[0]
            Re_high = self.Re_values[0]
        elif len(Re_high) == 0:
            Re_low = self.Re_values[-1]
            Re_high = self.Re_values[-1]
        else:
            Re_low = Re_low[-1]
            Re_high = Re_high[0]
        
        if Re_low == Re_high:
            cl = np.interp(alpha, self.alpha_values, self.cl_data[Re_low])
            cd = np.interp(alpha, self.alpha_values, self.cd_data[Re_low])
        else:
            # Interpolate between two Re values
            cl_low = np.interp(alpha, self.alpha_values, self.cl_data[Re_low])
            cl_high = np.interp(alpha, self.alpha_values, self.cl_data[Re_high])
            cd_low = np.interp(alpha, self.alpha_values, self.cd_data[Re_low])
            cd_high = np.interp(alpha, self.alpha_values, self.cd_data[Re_high])
            
            weight = (Re - Re_low) / (Re_high - Re_low)
            cl = cl_low + weight * (cl_high - cl_low)
            cd = cd_low + weight * (cd_high - cd_low)
        
        return float(cl), float(cd)
    
    def get_re_range(self) -> Tuple[float, float]:
        """Return min and max Re in database."""
        return self.Re_values.min(), self.Re_values.max()
    
    def get_alpha_range(self) -> Tuple[float, float]:
        """Return min and max alpha in database."""
        return self.alpha_values.min(), self.alpha_values.max()