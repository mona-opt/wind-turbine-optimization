import numpy as np
from typing import Tuple, Optional

class AirfoilDatabase:
    def __init__(self, csv_path: Optional[str] = None):
        if csv_path is not None:
            self._load_from_csv(csv_path)
        else:
            self._generate_mock_data()
    
    def _generate_mock_data(self):
        # داده‌های بهبودیافته برای NACA 4412
        self.Re_values = np.array([300000, 500000, 1000000])
        self.alpha_values = np.linspace(-10, 20, 30)
        
        self.cl_data = {}
        self.cd_data = {}
        
        for Re in self.Re_values:
            alpha = self.alpha_values
            # CL: خطی با شیب مناسب در ناحیه خطی
            cl = 0.1 * alpha + 0.4
            
            # استال در حدود 14 درجه (با کاهش تدریجی)
            stall_start = 12
            stall_end = 18
            for i, a in enumerate(alpha):
                if a > stall_start:
                    factor = 1.0 - (a - stall_start) / (stall_end - stall_start) * 0.5
                    cl[i] = cl[i] * max(factor, 0.5)
            
            cl = np.clip(cl, -0.5, 1.6)
            
            # CD: سهمی با مینیمم در زاویه صفر
            cd = 0.008 + 0.0001 * (alpha + 1)**2
            cd = np.clip(cd, 0.005, 0.040)
            
            self.cl_data[Re] = cl
            self.cd_data[Re] = cd
    
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
    
    def get_coeffs(self, alpha: float, Re: float) -> Tuple[float, float]:
        alpha = np.clip(alpha, self.alpha_values.min(), self.alpha_values.max())
        Re = np.clip(Re, self.Re_values.min(), self.Re_values.max())
        
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
            cl_low = np.interp(alpha, self.alpha_values, self.cl_data[Re_low])
            cl_high = np.interp(alpha, self.alpha_values, self.cl_data[Re_high])
            cd_low = np.interp(alpha, self.alpha_values, self.cd_data[Re_low])
            cd_high = np.interp(alpha, self.alpha_values, self.cd_data[Re_high])
            weight = (Re - Re_low) / (Re_high - Re_low)
            cl = cl_low + weight * (cl_high - cl_low)
            cd = cd_low + weight * (cd_high - cd_low)
        
        return float(cl), float(cd)
