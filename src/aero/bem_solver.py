# src/aero/bem_solver.py
"""
Blade Element Momentum (BEM) solver for wind turbine aerodynamics.
Includes Prandtl tip loss and Glauert correction for high axial induction.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
from .airfoil_database import AirfoilDatabase

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class BEMSolver:
    """
    BEM solver for wind turbine rotor performance.
    
    Usage:
        solver = BEMSolver(R=5.0, B=3, air_density=1.225)
        results = solver.solve(geometry, wind_speed=8.0, rpm=110)
    """
    
    def __init__(self, R: float, B: int = 3, air_density: float = 1.225,
                 airfoil_db: Optional[AirfoilDatabase] = None):
        self.R = R
        self.B = B
        self.rho = air_density
        self.airfoil_db = airfoil_db if airfoil_db is not None else AirfoilDatabase()
        
        # Convergence settings
        self.max_iter = 200
        self.tolerance = 1e-5
        self.a_max = 0.9
        self.relaxation = 0.3  # Under-relaxation for stability
    
    def solve(self, geometry: Dict, wind_speed: float, rpm: float,
              include_tip_loss: bool = True, include_glauert: bool = True) -> Dict:
        """
        Run BEM solver.
        
        Returns
        -------
        dict with: thrust, torque, power, axial_induction, tangential_induction
        """
        r = geometry['r']
        chord = geometry['chord']
        twist = geometry['twist']
        
        omega = rpm * 2 * np.pi / 60.0
        V0 = wind_speed
        
        n_elem = len(r)
        sigma = self.B * chord / (2 * np.pi * r)
        
        # Initialize arrays
        a = np.zeros(n_elem)
        a_prime = np.zeros(n_elem)
        dT = np.zeros(n_elem)
        dQ = np.zeros(n_elem)
        phi = np.zeros(n_elem)
        alpha = np.zeros(n_elem)
        F = np.ones(n_elem)
        
        for i in range(n_elem):
            a_i = 0.0
            a_prime_i = 0.0
            
            for it in range(self.max_iter):
                a_old = a_i
                a_prime_old = a_prime_i
                
                # Relative velocity
                v_axial = V0 * (1 - a_i)
                v_tang = omega * r[i] * (1 + a_prime_i)
                v_rel = np.sqrt(v_axial**2 + v_tang**2)
                
                if v_rel < 1e-8:
                    break
                
                phi_i = np.arctan2(v_axial, v_tang)
                alpha_i = np.degrees(phi_i - np.radians(twist[i]))
                
                # Airfoil coefficients
                Re = self.rho * v_rel * chord[i] / 1.81e-5
                cl, cd = self.airfoil_db.get_coeffs(alpha_i, Re)
                
                # Prandtl tip loss
                if include_tip_loss:
                    sin_phi = np.sin(phi_i)
                    if sin_phi > 1e-6:
                        f = self.B / 2.0 * (self.R - r[i]) / (r[i] * sin_phi)
                        F_i = 2.0 / np.pi * np.arccos(np.clip(np.exp(-f), 0, 1))
                    else:
                        F_i = 1.0
                else:
                    F_i = 1.0
                
                # Force coefficients
                cos_phi = np.cos(phi_i)
                sin_phi = np.sin(phi_i)
                Cn = cl * cos_phi + cd * sin_phi
                Ct = cl * sin_phi - cd * cos_phi
                
                # Axial induction
                if Cn > 0 and sigma[i] > 0:
                    denom = 4.0 * F_i * sin_phi**2 / (sigma[i] * Cn)
                    a_new = 1.0 / (denom + 1.0)
                else:
                    a_new = 0.0
                
                # Glauert correction
                if include_glauert and a_new > 0.33:
                    CT = sigma[i] * Cn * (1 - a_new)**2 / (sin_phi**2 + 1e-9)
                    if CT > 0.96:
                        a_new = 0.9
                    else:
                        a_new = 0.33 + 0.2 * (a_new - 0.33)
                
                a_new = np.clip(a_new, 0, self.a_max)
                
                # Tangential induction
                denom_ap = 4.0 * F_i * sin_phi * cos_phi
                if abs(denom_ap) > 1e-8 and Ct > 0:
                    a_prime_new = sigma[i] * Ct / (denom_ap - sigma[i] * Ct)
                else:
                    a_prime_new = 0.0
                a_prime_new = np.clip(a_prime_new, 0, 0.9)
                
                # Under-relaxation
                a_i = a_i + self.relaxation * (a_new - a_i)
                a_prime_i = a_prime_i + self.relaxation * (a_prime_new - a_prime_i)
                
                # Convergence check
                if abs(a_i - a_old) < self.tolerance and abs(a_prime_i - a_prime_old) < self.tolerance:
                    break
            
            # Store results
            a[i] = a_i
            a_prime[i] = a_prime_i
            phi[i] = phi_i
            alpha[i] = alpha_i
            F[i] = F_i
            
            # Calculate forces with final values
            v_axial_f = V0 * (1 - a_i)
            v_tang_f = omega * r[i] * (1 + a_prime_i)
            v_rel_f = np.sqrt(v_axial_f**2 + v_tang_f**2)
            
            if v_rel_f > 1e-8:
                Re_f = self.rho * v_rel_f * chord[i] / 1.81e-5
                cl_f, cd_f = self.airfoil_db.get_coeffs(alpha_i, Re_f)
                cos_phi_f = np.cos(phi_i)
                sin_phi_f = np.sin(phi_i)
                Cn_f = cl_f * cos_phi_f + cd_f * sin_phi_f
                Ct_f = cl_f * sin_phi_f - cd_f * cos_phi_f
                
                q = 0.5 * self.rho * v_rel_f**2 * chord[i] * self.B
                dT[i] = q * Cn_f
                dQ[i] = q * Ct_f * r[i]
        
        # Integrate (compatible with numpy 1.x and 2.x)
        try:
            total_thrust = np.trapezoid(dT, r)
            total_torque = np.trapezoid(dQ, r)
        except AttributeError:
            total_thrust = np.trapz(dT, r)
            total_torque = np.trapz(dQ, r)
        
        total_power = total_torque * omega
        
        return {
            'thrust': total_thrust,
            'torque': total_torque,
            'power': total_power,
            'axial_induction': a,
            'tangential_induction': a_prime,
            'angle_phi': phi,
            'angle_alpha': alpha,
            'tip_loss': F,
            'thrust_distribution': dT,
            'torque_distribution': dQ
        }