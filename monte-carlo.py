import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ============================================================================
# 1. CORE PHYSICS CLASSES (Definitions)
# ============================================================================

@dataclass
class DrivenOscillatorParams:
    mass: float = 1.0
    damping_coeff: float = 0.15
    spring_constant: float = 1.0
    f_amplitude: float = 1.0
    f_frequency: float = 1.0

    @property
    def natural_frequency(self) -> float:
        return np.sqrt(self.spring_constant / self.mass)

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
# 2. ANALYSIS FUNCTIONS
# ============================================================================

def get_steady_state_amplitude(params, t_max=100.0, dt=0.05):
    ode = DrivenODESystem(params)
    integrator = RK4Integrator(ode, dt)
    state = np.array([0.0, 0.0])
    x_history = []

    for step in range(int(t_max/dt)):
        t = step * dt
        state = integrator.step(state, t)
        if t > t_max * 0.8: # Only take last 20% of time
            x_history.append(state[0])
            
    return (np.max(x_history) - np.min(x_history)) / 2

# ============================================================================
# 3. MONTE CARLO CORE LOGIC
# ============================================================================

def run_monte_carlo_scan(freq_range, num_trials=30):
    all_results = []
    
    print(f"Starting Monte Carlo Scan with {num_trials} trials per frequency...")
    
    for freq in freq_range:
        trial_amps = []
        for _ in range(num_trials):
            # Adding 5% Gaussian noise to physical parameters
            m_noisy = np.random.normal(1.0, 0.05)
            c_noisy = np.random.normal(0.15, 0.03)
            k_noisy = np.random.normal(1.0, 0.05)
            
            p = DrivenOscillatorParams(mass=m_noisy, damping_coeff=c_noisy, 
                                       spring_constant=k_noisy, f_frequency=freq)
            
            amp = get_steady_state_amplitude(p)
            trial_amps.append(amp)
        
        all_results.append(trial_amps)
        print(f"Frequency {freq:.2f} [Trial Data Collected]")

    return np.array(all_results)

# ============================================================================
# 4. EXECUTION & VISUALIZATION
# ============================================================================

if __name__ == "__main__":
    # Frequency range near resonance
    freqs = np.linspace(0.5, 1.5, 40)
    
    # Run the Monte Carlo simulation
    raw_results = run_monte_carlo_scan(freqs, num_trials=30)
    
    # Statistical analysis
    means = np.mean(raw_results, axis=1)
    stds = np.std(raw_results, axis=1)

    # Plotting
    
    plt.figure(figsize=(12, 7))
    
    # Mean curve
    plt.plot(freqs, means, 'b-', label='Mean Amplitude', linewidth=2)
    
    # Confidence Interval (Shaded region)
    plt.fill_between(freqs, means - stds, means + stds, color='blue', alpha=0.2, label='1σ Uncertainty (68%)')
    plt.fill_between(freqs, means - 2*stds, means + 2*stds, color='blue', alpha=0.1, label='2σ Uncertainty (95%)')

    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Nominal $\omega_0$')
    
    plt.title("Monte Carlo Sensitivity Analysis: Driven Damped Oscillator", fontsize=14)
    plt.xlabel("Driving Frequency ($\omega$)", fontsize=12)
    plt.ylabel("Steady-State Amplitude ($A$)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print("\nSimulation complete. Plotting results...")
    plt.show()
