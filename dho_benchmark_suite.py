import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List

# ============================================================================
# 1. CORE PHYSICS ENGINE
# ============================================================================

@dataclass
class OscillatorParams:
    """Stores physical parameters of the system."""
    mass: float = 1.0
    damping_coeff: float = 0.15
    spring_constant: float = 1.0
    f_amplitude: float = 1.0
    f_frequency: float = 1.0  # ω (driving frequency)

    @property
    def natural_frequency(self) -> float:
        """ω0 = √(k/m)"""
        return np.sqrt(self.spring_constant / self.mass)

class DrivenODESystem:
    """Defines the 1st order coupled ODEs for the system."""
    def __init__(self, params: OscillatorParams):
        self.p = params

    def derivative(self, state: np.ndarray, t: float) -> np.ndarray:
        x, v = state
        # F(t) = F0 * cos(ωt)
        f_ext = self.p.f_amplitude * np.cos(self.p.f_frequency * t)
        
        dxdt = v
        dvdt = (f_ext - self.p.damping_coeff * v - self.p.spring_constant * x) / self.p.mass
        return np.array([dxdt, dvdt])

class RK4Integrator:
    """Runge-Kutta 4th Order Numerical Solver."""
    def __init__(self, ode_system: DrivenODESystem, dt: float = 0.05):
        self.ode = ode_system
        self.dt = dt

    def step(self, state: np.ndarray, t: float) -> np.ndarray:
        dt = self.dt
        k1 = self.ode.derivative(state, t)
        k2 = self.ode.derivative(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.ode.derivative(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.ode.derivative(state + dt * k3, t + dt)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ============================================================================
# 2. ANALYSIS MODULES
# ============================================================================

def get_steady_state_amplitude(params: OscillatorParams, t_max: float = 120.0, dt: float = 0.05) -> float:
    """Runs simulation until steady state and extracts peak amplitude."""
    ode = DrivenODESystem(params)
    integrator = RK4Integrator(ode, dt)
    state = np.array([0.0, 0.0]) # Initial x, v
    x_history = []

    for step in range(int(t_max/dt)):
        t = step * dt
        state = integrator.step(state, t)
        if t > t_max * 0.8: # Consider only the last 20% of the timeline
            x_history.append(state[0])
            
    return (np.max(x_history) - np.min(x_history)) / 2

def run_monte_carlo_analysis(freq_range: np.ndarray, num_trials: int = 30):
    """Performs sensitivity analysis using Monte Carlo trials."""
    results = []
    print(f"--- Running Monte Carlo Scan ({num_trials} trials/freq) ---")
    
    for freq in freq_range:
        trial_amps = []
        for _ in range(num_trials):
            # 5% uncertainty in physical properties
            m_noisy = np.random.normal(1.0, 0.05)
            c_noisy = np.random.normal(0.15, 0.03)
            k_noisy = np.random.normal(1.0, 0.05)
            
            p = OscillatorParams(mass=m_noisy, damping_coeff=c_noisy, 
                                 spring_constant=k_noisy, f_frequency=freq)
            trial_amps.append(get_steady_state_amplitude(p))
        results.append(trial_amps)
    return np.array(results)

# ============================================================================
# 3. VISUALIZATION & MAIN EXECUTION
# ============================================================================

def main():
    print("Damped Harmonic Oscillator Benchmark Suite Starting...")
    
    # 1. Individual Simulation Case (Near Resonance)
    p_case = OscillatorParams(f_frequency=0.95)
    ode = DrivenODESystem(p_case)
    integrator = RK4Integrator(ode, dt=0.05)
    
    t_vals, x_vals = [], []
    state = np.array([0.0, 0.0])
    for s in range(int(100/0.05)):
        t = s * 0.05
        t_vals.append(t)
        x_vals.append(state[0])
        state = integrator.step(state, t)

    # 2. Monte Carlo Resonance Scan
    freqs = np.linspace(0.5, 1.5, 40)
    mc_data = run_monte_carlo_analysis(freqs, num_trials=30)
    means = np.mean(mc_data, axis=1)
    stds = np.std(mc_data, axis=1)

    # Plotting Results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Top Plot: Time Series
    ax1.plot(t_vals, x_vals, color='tab:blue', label='Displacement (x)')
    ax1.set_title("Time Domain: Transient to Steady State (ω=0.95)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Bottom Plot: Monte Carlo Resonance Curve
    
    ax2.plot(freqs, means, 'b-', label='Mean Steady-State Amp', linewidth=2)
    ax2.fill_between(freqs, means - 2*stds, means + 2*stds, color='blue', alpha=0.1, label='95% Confidence (2σ)')
    ax2.axvline(x=1.0, color='red', linestyle='--', label='Natural Freq (ω0)')
    ax2.set_title("Frequency Domain: Monte Carlo Resonance & Sensitivity")
    ax2.set_xlabel("Driving Frequency (ω)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    print("\nBenchmark Complete. Displaying Plots.")
    plt.show()

if __name__ == "__main__":
    main()
