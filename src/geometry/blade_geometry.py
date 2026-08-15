# src/geometry/blade_geometry.py
"""
Parametric wind turbine blade geometry generator.
Project: 10 kW HAWT for Manjil site
Design variables (8): [chord_cp1..chord_cp4, twist_cp1..twist_cp4]
Control point locations (r/R): [0.20, 0.45, 0.70, 0.95]
Airfoil: NACA 4412 (fixed for all sections)
Only depends on NumPy and SciPy (for cubic spline).
Designed for use inside an optimization loop.
"""

import numpy as np
from scipy.interpolate import make_interp_spline
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


@dataclass
class BladeSection:
    """Container for one blade cross-section."""
    r: float                     # radial position (m)
    chord: float                 # chord length (m)
    twist: float                 # twist angle (deg)
    airfoil_points: np.ndarray   # 2D array (x, y) with chord=1 (unit)


class BladeGeometry:
    """
    Parametric blade geometry generator.

    Usage:
        blade = BladeGeometry(R=5.0, hub_radius=0.8, num_stations=30)
        data = blade.build_from_vector(design_vector)
    """

    def __init__(self, R: float, hub_radius: float, num_stations: int):
        """
        Parameters
        ----------
        R : float
            Rotor tip radius (m)
        hub_radius : float
            Hub radius (m)
        num_stations : int
            Number of radial stations for geometry discretization (e.g., 30)
        """
        if hub_radius >= R:
            raise ValueError("hub_radius must be smaller than R")
        self.R = R
        self.hub_radius = hub_radius
        self.num_stations = num_stations
        self.r_grid = np.linspace(hub_radius, R, num_stations)

        # Control point locations (normalized by blade length)
        self.cp_r_over_R = np.array([0.20, 0.45, 0.70, 0.95])
        self.cp_radii = self.hub_radius + self.cp_r_over_R * (self.R - self.hub_radius)

        # Bounds for design variables (according to plan)
        self.chord_bounds = (0.05, 0.60)   # meters
        self.twist_bounds = (0.0, 35.0)    # degrees

        # Fixed airfoil: NACA 4412
        self.airfoil_params = (0.04, 0.4, 0.12)   # m, p, t

        # Storage
        self.params = None
        self.sections: List[BladeSection] = []
        self.chord_grid: Optional[np.ndarray] = None
        self.twist_grid: Optional[np.ndarray] = None
        self.valid = False
        self.messages: List[str] = []

    # ------------------------------------------------------------------
    def build_from_vector(self, x) -> Dict:
        """
        Build blade geometry from an 8-element design vector.

        x = [chord_cp1, chord_cp2, chord_cp3, chord_cp4,
             twist_cp1, twist_cp2, twist_cp3, twist_cp4]

        chord values in meters, twist values in degrees.
        Control points are located at r/R = [0.20, 0.45, 0.70, 0.95].
        """
        if len(x) != 8:
            raise ValueError(f"Expected 8 design variables, got {len(x)}")

        x = np.asarray(x, dtype=float)
        chord_vals = x[0:4]
        twist_vals = x[4:8]

        cp_chord = np.column_stack([self.cp_radii, chord_vals])
        cp_twist = np.column_stack([self.cp_radii, twist_vals])

        return self.build(cp_chord, cp_twist, airfoil_params=self.airfoil_params)

    # ------------------------------------------------------------------
    def build(self,
              cp_chord: np.ndarray,
              cp_twist: np.ndarray,
              airfoil_params: Tuple[float, float, float] = (0.04, 0.4, 0.12)
              ) -> Dict:
        """
        Build the blade geometry from control points for chord and twist.

        Parameters
        ----------
        cp_chord : ndarray shape (n_cp, 2) with columns [radius, chord]
        cp_twist : ndarray shape (n_cp, 2) with columns [radius, twist]
        airfoil_params : tuple (m, p, t) for NACA 4-digit; default NACA 4412.

        Returns
        -------
        dict with keys: r, chord, twist, sections, valid, messages
        """
        if cp_chord.shape[0] < 4:
            raise ValueError("cp_chord must have at least 4 control points")
        if cp_twist.shape[0] < 4:
            raise ValueError("cp_twist must have at least 4 control points")

        # Sort control points by radius
        cp_chord = cp_chord[cp_chord[:, 0].argsort()]
        cp_twist = cp_twist[cp_twist[:, 0].argsort()]

        # Interpolate chord and twist onto r_grid using clamped cubic spline
        def _interp(cp):
            r_cp = cp[:, 0]
            val_cp = cp[:, 1]
            spline = make_interp_spline(r_cp, val_cp, k=3, bc_type='clamped')
            return spline(self.r_grid)

        chord_grid = _interp(cp_chord)
        twist_grid = _interp(cp_twist)

        # Generate airfoil coordinates for each station (unit chord)
        self.sections = []
        for i, r in enumerate(self.r_grid):
            chord = chord_grid[i]
            twist = twist_grid[i]
            # NACA 4-digit airfoil (unit chord)
            x_unit, y_unit = self._naca4(
                airfoil_params[0], airfoil_params[1], airfoil_params[2]
            )
            section = BladeSection(
                r=r,
                chord=chord,
                twist=twist,
                airfoil_points=np.column_stack([x_unit, y_unit])  # unit chord
            )
            self.sections.append(section)

        self.chord_grid = chord_grid
        self.twist_grid = twist_grid

        # Validate physical constraints
        self.valid, self.messages = self.validate()

        return {
            'r': self.r_grid,
            'chord': self.chord_grid,
            'twist': self.twist_grid,
            'sections': self.sections,
            'valid': self.valid,
            'messages': self.messages,
        }

    # ------------------------------------------------------------------
    def _naca4(self, m: float, p: float, t: float, num_points: int = 100):
        """
        Generate NACA 4-digit airfoil coordinates (unit chord).
        Returns (x, y) arrays with x from 0 to 1.
        """
        # Clamp parameters to valid range
        m = max(0.0, min(0.1, m))
        p = max(0.0, min(0.9, p))
        t = max(0.05, min(0.30, t))

        x = np.linspace(0, 1, num_points)
        # Thickness distribution
        yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)

        # Camber line and slope
        yc = np.where(
            x < p,
            m / p**2 * (2*p*x - x**2),
            m / (1-p)**2 * ((1-2*p) + 2*p*x - x**2)
        )
        dyc_dx = np.where(
            x < p,
            2*m / p**2 * (p - x),
            2*m / (1-p)**2 * (p - x)
        )
        theta = np.arctan(dyc_dx)

        # Upper and lower surfaces
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        # Combine (upper reversed, then lower)
        X = np.concatenate([xu[::-1], xl[1:]])
        Y = np.concatenate([yu[::-1], yl[1:]])
        return X, Y

    # ------------------------------------------------------------------
    def validate(self) -> Tuple[bool, List[str]]:
        """Check geometric constraints according to plan."""
        msgs = []

        if self.chord_grid is None or self.twist_grid is None:
            msgs.append("Geometry not built yet")
            return False, msgs

        # Chord bounds
        if np.any(self.chord_grid < self.chord_bounds[0]):
            msgs.append(f"Chord below lower bound {self.chord_bounds[0]} m")
        if np.any(self.chord_grid > self.chord_bounds[1]):
            msgs.append(f"Chord exceeds upper bound {self.chord_bounds[1]} m")

        # Twist bounds
        if np.any(self.twist_grid < self.twist_bounds[0]):
            msgs.append(f"Twist below lower bound {self.twist_bounds[0]} deg")
        if np.any(self.twist_grid > self.twist_bounds[1]):
            msgs.append(f"Twist exceeds upper bound {self.twist_bounds[1]} deg")

        # Additional sanity checks
        if np.any(self.chord_grid <= 0):
            msgs.append("Chord distribution contains non-positive values")

        if len(msgs) == 0:
            return True, msgs
        else:
            return False, msgs

    # ------------------------------------------------------------------
    def get_section_3d_points(self, section_index: int) -> np.ndarray:
        """
        Return 3D coordinates of a blade section at given index.
        The airfoil is placed at radial position r, with chord and twist applied.
        Coordinates: (x, y, z) where x is chordwise, y is flapwise, z is spanwise (radial).
        """
        if section_index < 0 or section_index >= len(self.sections):
            raise IndexError("section_index out of range")
        sec = self.sections[section_index]
        x_unit, y_unit = sec.airfoil_points[:, 0], sec.airfoil_points[:, 1]
        # Scale by chord
        x = x_unit * sec.chord
        y = y_unit * sec.chord
        # Twist rotation about radial axis (z)
        angle = np.radians(sec.twist)
        x_rot = x * np.cos(angle) - y * np.sin(angle)
        y_rot = x * np.sin(angle) + y * np.cos(angle)
        z = np.full_like(x_rot, sec.r)
        return np.column_stack([x_rot, y_rot, z])

    def get_all_section_points(self) -> List[np.ndarray]:
        """Return list of 3D points for all sections."""
        return [self.get_section_3d_points(i) for i in range(len(self.sections))]

    # ------------------------------------------------------------------
    def estimate_mass(self, density_kg_m3: float = 1700.0) -> float:
        """
        Rough mass estimate assuming a rectangular cross-section.
        For more accurate mass, use actual airfoil area integration.
        """
        if self.chord_grid is None:
            raise ValueError("Geometry not built yet")
        avg_chord = np.mean(self.chord_grid)
        avg_thickness = 0.12 * avg_chord   # approximate thickness from NACA 4412
        blade_length = self.R - self.hub_radius
        volume = avg_chord * avg_thickness * blade_length
        mass = volume * density_kg_m3
        return mass

    # ------------------------------------------------------------------
    def export_dxf(self, filename: str):
        """Export a simple 2D DXF of the blade planform (chord vs radius)."""
        try:
            import ezdxf
        except ImportError:
            raise ImportError("ezdxf is required for DXF export. Install with: pip install ezdxf")

        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        # Draw leading and trailing edges
        for i in range(len(self.r_grid)-1):
            r1, c1 = self.r_grid[i], self.chord_grid[i]
            r2, c2 = self.r_grid[i+1], self.chord_grid[i+1]
            # Leading edge at x=0, trailing edge at x=c
            msp.add_line((0, r1), (0, r2), dxfattribs={'layer': 'LE'})
            msp.add_line((c1, r1), (c2, r2), dxfattribs={'layer': 'TE'})
        doc.saveas(filename)
        return True


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create blade geometry with R=5 m, hub=0.8 m, 30 stations
    blade = BladeGeometry(R=5.0, hub_radius=0.8, num_stations=30)

    # Example design vector (8 variables)
    # chord: 0.35, 0.30, 0.25, 0.15 (m)
    # twist: 25, 20, 12, 5 (deg)
    design = [0.35, 0.30, 0.25, 0.15, 25.0, 20.0, 12.0, 5.0]

    data = blade.build_from_vector(design)
    print(f"Valid: {data['valid']}")
    for msg in data['messages']:
        print(" -", msg)
    print(f"Number of sections: {len(blade.sections)}")
    print(f"Average chord: {np.mean(blade.chord_grid):.3f} m")
    print(f"Estimated mass (density=1700 kg/m3): {blade.estimate_mass():.2f} kg")