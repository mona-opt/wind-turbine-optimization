
import numpy as np
from typing import Tuple, Optional

class AirfoilDatabase:
    def __init__(self, csv_path: Optional[str] = None):
        if csv_path is not None:
            self._load_from_csv(csv_path)
        else:
            # بارگذاری داده‌های واقعی از فایل داخلی
            try:
                data_path = "data/naca4412_re500k.csv"
                self._load_from_csv(data_path)
                print(f"✅ Loaded real airfoil data from {data_path}")
            except:
                print("⚠️ Real data not found, using mock data")
                self._generate_mock_data()
    
    def get_coeffs(self, alpha: float, Re: float) -> Tuple[float, float]:
        # فقط یک رینولدز برای سادگی
        if hasattr(self, 'alpha_values') and hasattr(self, 'cl_data'):
            # درون‌یابی با داده‌های واقعی
            cl = np.interp(alpha, self.alpha_values, self.cl_data)
            cd = np.interp(alpha, self.alpha_values, self.cd_data)
            return float(cl), float(cd)
        return self._interpolate_mock(alpha, Re)
    
    def _interpolate_mock(self, alpha, Re):
        cl = 0.11 * alpha + 0.35
        if alpha > 12:
            cl = cl * (1 - 0.05 * (alpha - 12))
        cl = np.clip(cl, -0.5, 1.5)
        cd = 0.008 + 0.0001 * (alpha + 1)**2
        cd = np.clip(cd, 0.006, 0.035)
        return cl, cd
    
    def _load_from_csv(self, csv_path: str):
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        self.alpha_values = data[:, 0]
        self.Re_values = data[:, 1]
        self.cl_data = data[:, 2]
        self.cd_data = data[:, 3]
    
    def _generate_mock_data(self):
        self.alpha_values = np.linspace(-10, 20, 30)
        self.cl_data = np.array([self._interpolate_mock(a, 500000)[0] for a in self.alpha_values])
        self.cd_data = np.array([self._interpolate_mock(a, 500000)[1] for a in self.alpha_values])

