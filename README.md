# Wind Turbine Blade Optimization (10 kW HAWT)

## 📌 Overview
Educational project for aerodynamic optimization of a 10 kW wind turbine blade using BEM and WOA.

## 🧠 Method
- **Solver:** BEM with Prandtl & Glauert corrections
- **Optimizer:** Whale Optimization Algorithm (WOA)
- **Airfoil:** SG6043 (suitable for small wind turbines)

## 📐 Design Variables
- 4 chord control points (0.05–0.60 m)
- 4 twist control points (0°–35°)
- Control points at r/R: 0.20, 0.45, 0.70, 0.95

## ⚙️ Constraints
- **Structural:** Thrust ≤ 7000 N, Root moment ≤ 5500 N·m, Stress ≤ 70 MPa, Frequency > 6.5 Hz
- **Aerodynamic:** a ≤ 0.45, α ≤ α_stall - 2°
- **Operational:** 6.5 ≤ TSR ≤ 7.5, 0.025 ≤ Solidity ≤ 0.05
- **Manufacturing:** Blade mass ≤ 30 kg, Tip deflection ≤ 0.5 m

## 📊 Initial Results
- **Best AEP:** ~63,700 kWh/year (light test)
- **Optimal design:** Chord: [0.371, 0.360, 0.111, 0.282] m, Twist: [12.1, 19.8, 27.2, 19.9]°

## ⚠️ Note
This is an **educational project** and **needs validation** with CFD tools (e.g., OpenFOAM, Fluent) or QBlade.

## 🤝 Contributing
Feedback, suggestions, and contributions are welcome. Please open an issue or submit a pull request.

## 🙏 Acknowledgments
- This project was developed with the assistance of AI (DeepSeek).
- Thanks to the open-source Python community for the essential tools.

**Status:** ✅ Under development – Needs validation
