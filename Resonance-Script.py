import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ============================================================================
# 1. PHYSICS MODEL & PARAMETERS
# ============================================================================

@dataclass
class DrivenOscillatorParams:
    mass: float = 1.0
    damping_coeff: float = 0.15   # Low damping for a sharper peak
    spring_constant: float = 1.0  # ω0 = 1.0
    f_amplitude: float = 1.0      # F0
    f_frequency: float = 1.0      # This will vary during the scan

    @property
    def natural_frequency(self) -> float:
        return np.sqrt(self.spring_constant / self.mass)

# ============================================================================
# 2. ODE SYSTEM & INTEGRATOR
# ============================================================================

class DrivenODESystem:
    def __init__(self, params: DrivenOscillatorParams):
        self.p = params

    def state_derivative(self, state: np.ndarray, t: float) -> np.ndarray:
        x, v = state
        f_ext = self.p.f_amplitude * np.cos(self.p.f_frequency * t)
        dxdt = v
        dvdt = (f_ext - self.p.damping_coeff * v - self.p.spring_constant * x) / self.p.mass
        return np.array([dxdt, dvdt])

class RK4Integrator:
    def __init__(self, ode_system: DrivenODESystem, dt: float = 0.05):
        self.ode = ode_system
        self.dt = dt

    def step(self, state: np.ndarray, t: float) -> np.ndarray:
        dt = self.dt
        k1 = self.ode.state_derivative(state, t)
        k2 = self.ode.state_derivative(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.ode.state_derivative(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.ode.state_derivative(state + dt * k3, t + dt)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ============================================================================
# 3. ANALYSIS FUNCTIONS
# ============================================================================

def get_steady_state_amplitude(params, t_max=120.0, dt=0.05):
    """Simulates until steady state and returns amplitude."""
    ode = DrivenODESystem(params)
    integrator = RK4Integrator(ode, dt)
    state = np.array([0.0, 0.0])
    x_history = []

    for step in range(int(t_max/dt)):
        t = step * dt
        state = integrator.step(state, t)
        # Steady state starts after transients decay (last 20% of time)
        if t > t_max * 0.8:
            x_history.append(state[0])
            
    return (np.max(x_history) - np.min(x_history)) / 2

# ============================================================================
# 4. EXECUTION: FREQUENCY SCAN
# ============================================================================

if __name__ == "__main__":
    # Parameters for scanning
    frequencies = np.linspace(0.1, 2.5, 60)
    amplitudes = []
    
    # Base configuration
    base_params = DrivenOscillatorParams(damping_coeff=0.15)
    
    print(f"Starting Resonance Scan (ω₀ = {base_params.natural_frequency:.2f})...")
    
    for freq in frequencies:
        base_params.f_frequency = freq
        amp = get_steady_state_amplitude(base_params)
        amplitudes.append(amp)
        print(f"Freq: {freq:.2f} -> Amp: {amp:.4f}")

    # Theoretical peak calculation for verification
    # ω_r = sqrt(ω₀² - 2β²) where β = c/(2m)
    beta = base_params.damping_coeff / (2 * base_params.mass)
    theoretical_peak_freq = np.sqrt(base_params.natural_frequency**2 - 2 * beta**2)

    # Visualization
    
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, amplitudes, 'b-', label='Numerical Simulation (RK4)', linewidth=2)
    plt.fill_between(frequencies, amplitudes, color='blue', alpha=0.1)
    
    plt.axvline(x=base_params.natural_frequency, color='red', linestyle='--', label='Natural Freq ($ω_0$)')
    plt.axvline(x=theoretical_peak_freq, color='green', linestyle=':', label='Theoretical Resonant Peak ($ω_r$)')

    plt.title("Resonance Analysis Benchmark", fontsize=14)
    plt.xlabel("Driving Frequency ($ω$)", fontsize=12)
    plt.ylabel("Steady-State Amplitude ($A$)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print(f"\nScan complete. Theoretical peak frequency: {theoretical_peak_freq:.4f}")
    plt.show()
