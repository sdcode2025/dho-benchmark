# DHO-Benchmark: Physics-Grounded AI Evaluation Suite

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-green.svg)
![Field](https://img.shields.io/badge/field-Computational%20Physics-orange.svg)

---

## 🎯 Overview

This repository implements a high-precision numerical simulation of a  
**Driven Damped Harmonic Oscillator (DDHO)**.

It is designed as a **physics-grounded benchmark** to evaluate an AI agent’s
ability to:

1. **Solve coupled ordinary differential equations (ODEs)** using stable
   numerical methods (RK4).
2. **Validate physical consistency**, including damping regimes and energy decay.
3. **Perform sensitivity analysis** via Monte Carlo simulations to handle
   real-world uncertainty.

The benchmark explicitly tests *reasoning quality*, not just numerical output.

---

## 🧮 Mathematical Model

The system is governed by the second-order non-autonomous differential equation:

\[
m\ddot{x} + c\dot{x} + kx = F_0 \cos(\omega t)
\]

For numerical integration, this is converted into a system of first-order
coupled ODEs:

\[
\frac{dx}{dt} = v
\]

\[
\frac{dv}{dt} =
\frac{1}{m}\left[F_0 \cos(\omega t) - cv - kx\right]
\]

---

## 🚀 Key Features

- **RK4 Integrator**  
  Uses the 4th-order Runge–Kutta method for superior numerical stability compared
  to the Euler method.

- **Resonance Scanning**  
  Automated frequency sweeps to identify resonance peaks and damping-induced
  frequency shifts (\omega_r).

- **Monte Carlo Suite**  
  Introduces Gaussian noise (~5%) to physical parameters (m, c, k) to produce
  probabilistic resonance curves and confidence intervals.

- **Visualization**  
  Clear plots of transient behavior, steady-state response, and
  frequency-domain sensitivity.

---

## 🛠️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/sdcode2025/dho-benchmark.git
cd dho-benchmark

# Install dependencies
pip install numpy matplotlib

# Run the benchmark suite
python3 dho_benchmark_suite.py

---

📊 Analysis & Results
1️⃣ Resonance Peak Shift
The simulation accurately captures the damping-induced resonance shift, where the resonance frequency is slightly lower than the system’s natural frequency.
2️⃣ Monte Carlo Uncertainty
Shaded regions in the resonance plots represent 1σ and 2σ confidence intervals, demonstrating increased variance near resonance where the system is most sensitive to parameter fluctuations.

---

🧠 Why This for AI Benchmarking?
Standard LLMs and autonomous agents often struggle with:
Numerical Drift — incorrect time-step or integrator choice.
Sign Errors — misinterpreting damping or restoring forces.
Physical Plausibility — producing unbounded or non-decaying solutions.
Statistical Interpretation — failing to explain increased variance near resonance.
This benchmark provides a ground-truth physics reference to evaluate whether an AI system can reason about numerical stability, physical laws, and uncertainty, not just compute equations.

---
📄 License
This project is licensed under the MIT License.

---

## 👤 Author

**Suvankar Das**  
M.Sc. Physics | Computational Physics | AI Benchmarking  
GitHub: https://github.com/sdcode2025  
LinkedIn: https://www.linkedin.com/in/suvankar-das-51916711b
