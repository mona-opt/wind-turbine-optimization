
import numpy as np
from typing import Tuple, Optional

class AirfoilDatabase:
    def __init__(self, csv_path: Optional[str] = None):
        self.use_aerodynamics = False
        if csv_path is not None:
            self._load_from_csv(csv_path)
        else:
            # استفاده از کتابخانه aerodynamics
            try:
                from aerodynamics import Airfoil
                self.airfoil = Airfoil('naca4412')
                self.use_aerodynamics = True
                print("✅ Using aerodynamics library for NACA 4412")
            except ImportError:
                print("⚠️ aerodynamics not available, using mock data")
                self._generate_mock_data()
    
    def get_coeffs(self, alpha: float, Re: float) -> Tuple[float, float]:
        if self.use_aerodynamics:
            try:
                # aerodynamics library expects alpha in degrees and returns (cl, cd)
                cl, cd = self.airfoil.cl_cd(alpha, Re)
                return float(cl), float(cd)
            except Exception as e:
                print(f"⚠️ aerodynamics failed: {e}, falling back to mock")
                return self._interpolate_mock(alpha, Re)
        return self._interpolate_mock(alpha, Re)
    
    def _interpolate_mock(self, alpha, Re):
        """Improved mock data for fallback"""
        # CL: linear with stall at 15 degrees
        cl = 0.11 * alpha + 0.35
        if alpha > 12:
            cl = cl * (1 - 0.05 * (alpha - 12))
        cl = np.clip(cl, -0.5, 1.5)
        # CD: parabolic with minimum at zero lift
        cd = 0.008 + 0.0001 * (alpha + 1)**2
        cd = np.clip(cd, 0.006, 0.035)
        return cl, cd
    
    def _generate_mock_data(self):
        self.Re_values = np.array([300000, 500000, 1000000])
        self.alpha_values = np.linspace(-10, 20, 30)
        self.cl_data = {}
        self.cd_data = {}
        for Re in self.Re_values:
            cl_list = []
            cd_list = []
            for alpha in self.alpha_values:
                cl, cd = self._interpolate_mock(alpha, Re)
                cl_list.append(cl)
                cd_list.append(cd)
            self.cl_data[Re] = np.array(cl_list)
            self.cd_data[Re] = np.array(cd_list)
    
    def _load_from_csv(self, csv_path: str):
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        self.Re_values = np.unique(data[:, 1])
        self.alpha_values = np.unique(data[:, 0])
        self.cl_data = {}
        self.cd_data = {}
        for Re in self.Re_values:
            mask = data[:, 1] == Re
            self.cl_data[Re] = data[mask, 2]
            self.cd_data[Re] = data[mask, 3]
EOF