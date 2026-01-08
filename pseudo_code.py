import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List

# ============================================================================
# 1. PHYSICS MODEL: Damped Harmonic Oscillator
# ============================================================================

@dataclass
class DampedOscillatorParams:
    """Physical parameters for the system"""
    mass: float = 1.0          # m (kg)
    damping_coeff: float = 0.5 # c (N·s/m)
    spring_constant: float = 1.0 # k (N/m)
    
    @property
    def natural_frequency(self) -> float:
        """ω₀ = √(k/m)"""
        return np.sqrt(self.spring_constant / self.mass)
    
    @property
    def damping_ratio(self) -> float:
        """ζ = c / (2√(mk)) — determines damping regime"""
        return self.damping_coeff / (2 * np.sqrt(self.spring_constant * self.mass))
    
    @property
    def damping_regime(self) -> str:
        """Classify the damping behavior"""
        zeta = self.damping_ratio
        if zeta < 1.0:
            return "Underdamped"      # oscillates + decays
        elif zeta == 1.0:
            return "Critically damped" # no oscillation, fastest decay
        else:
            return "Overdamped"        # slow exponential decay


# ============================================================================
# 2. ODE SYSTEM: Convert 2nd order → First order
# ============================================================================

class ODESystem:
    """
    Represent: m*ẍ + c*ẋ + k*x = 0
    As:
        ẋ = v
        v̇ = -(c/m)*v - (k/m)*x
    """
    
    def __init__(self, params: DampedOscillatorParams):
        self.params = params
        # Pre-compute coefficients for efficiency
        self.damping_coeff_normalized = params.damping_coeff / params.mass
        self.spring_coeff_normalized = params.spring_constant / params.mass
    
    def state_derivative(self, state: np.ndarray) -> np.ndarray:
        """
        Compute dstate/dt given current state
        
        Input:  state = [x, v] (position, velocity)
        Output: d_state/dt = [v, a] (velocity, acceleration)
        """
        x, v = state
        
        # ẋ = v
        dx_dt = v
        
        # v̇ = -(c/m)*v - (k/m)*x
        dv_dt = -self.damping_coeff_normalized * v - self.spring_coeff_normalized * x
        
        return np.array([dx_dt, dv_dt])


# ============================================================================
# 3. NUMERICAL INTEGRATION METHODS
# ============================================================================

class NumericalIntegrator:
    """Base class for ODE solvers"""
    
    def __init__(self, ode_system: ODESystem, dt: float = 0.01):
        self.ode = ode_system
        self.dt = dt
    
    def step(self, state: np.ndarray) -> np.ndarray:
        """Advance state by one timestep — implement in subclass"""
        raise NotImplementedError


class EulerIntegrator(NumericalIntegrator):
    """
    Euler's Forward Method (1st order, O(dt²) error per step)
    state_{n+1} = state_n + dt * f(state_n)
    """
    
    def step(self, state: np.ndarray) -> np.ndarray:
        derivative = self.ode.state_derivative(state)
        return state + self.dt * derivative


class RK4Integrator(NumericalIntegrator):
    """
    Runge–Kutta 4th Order (4th order, O(dt⁵) error per step)
    Much more stable than Euler for same timestep
    """
    
    def step(self, state: np.ndarray) -> np.ndarray:
        k1 = self.ode.state_derivative(state)
        k2 = self.ode.state_derivative(state + 0.5 * self.dt * k1)
        k3 = self.ode.state_derivative(state + 0.5 * self.dt * k2)
        k4 = self.ode.state_derivative(state + self.dt * k3)
        
        return state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ============================================================================
# 4. SIMULATION ENGINE
# ============================================================================

class Simulator:
    """Run the full numerical simulation"""
    
    def __init__(self, params: DampedOscillatorParams, integrator_type: str = "RK4", dt: float = 0.01):
        self.params = params
        self.dt = dt
        
        # Initialize ODE system
        ode_system = ODESystem(params)
        
        # Choose integrator
        if integrator_type.upper() == "EULER":
            self.integrator = EulerIntegrator(ode_system, dt)
        elif integrator_type.upper() == "RK4":
            self.integrator = RK4Integrator(ode_system, dt)
        else:
            raise ValueError(f"Unknown integrator: {integrator_type}")
        
        # Storage for trajectory
        self.t_history = []
        self.x_history = []
        self.v_history = []
    
    def run(self, t_span: float, initial_position: float = 1.0, initial_velocity: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate from t=0 to t=t_span
        
        Args:
            t_span: total simulation time
            initial_position: x(0)
            initial_velocity: v(0)
        
        Returns:
            (t_array, x_array, v_array)
        """
        # Initial state
        state = np.array([initial_position, initial_velocity])
        
        # Time array
        n_steps = int(t_span / self.dt)
        t_current = 0.0
        
        # Reset history
        self.t_history = [t_current]
        self.x_history = [state[0]]
        self.v_history = [state[1]]
        
        # Time-stepping loop
        for step_idx in range(n_steps):
            state = self.integrator.step(state)
            t_current += self.dt
            
            self.t_history.append(t_current)
            self.x_history.append(state[0])
            self.v_history.append(state[1])
        
        return np.array(self.t_history), np.array(self.x_history), np.array(self.v_history)


# ============================================================================
# 5. VALIDATION & CORRECTNESS CHECKS
# ============================================================================

class Validator:
    """Verify physical correctness of simulation"""
    
    @staticmethod
    def check_energy_conservation(x: np.ndarray, v: np.ndarray, params: DampedOscillatorParams) -> Tuple[bool, float]:
        """
        Energy should decay monotonically (dissipated by damping).
        E(t) = (1/2)*m*v² + (1/2)*k*x²
        
        Return: (is_valid, max_energy_increase_percent)
        """
        E = 0.5 * params.mass * v**2 + 0.5 * params.spring_constant * x**2
        E_initial = E[0]
        
        # Check for energy increase (non-physical)
        dE = np.diff(E)
        num_increases = np.sum(dE > 1e-10)  # tolerance for numerical errors
        
        max_increase_percent = np.max(dE) / E_initial * 100 if E_initial > 0 else 0
        
        is_valid = num_increases < len(dE) * 0.05  # allow <5% of steps to increase
        
        return is_valid, max_increase_percent
    
    @staticmethod
    def check_amplitude_decay(x: np.ndarray, params: DampedOscillatorParams) -> Tuple[bool, float]:
        """
        For underdamped case, amplitude should decay exponentially.
        Expected decay: A(t) = A₀ * exp(-ζ*ω₀*t)
        """
        zeta = params.damping_ratio
        omega_0 = params.natural_frequency
        
        if zeta < 1.0:  # Only underdamped shows oscillations
            # Find local maxima (crude approach)
            peaks_idx = np.where(np.diff(np.sign(np.diff(x))) == -2)[0]
            
            if len(peaks_idx) > 1:
                # Check if peak heights decay
                peak_values = np.abs(x[peaks_idx])
                peak_ratio = peak_values[1:] / (peak_values[:-1] + 1e-10)
                
                # Expected ratio: exp(-ζ*ω₀*T) where T ≈ 2π/ωd
                expected_decay_per_cycle = np.exp(-2 * np.pi * zeta / np.sqrt(1 - zeta**2))
                
                deviation = np.abs(np.mean(peak_ratio) - expected_decay_per_cycle)
                is_valid = deviation < 0.1  # allow 10% deviation
                
                return is_valid, deviation
        
        return True, 0.0
    
    @staticmethod
    def check_numerical_stability(x: np.ndarray, v: np.ndarray) -> Tuple[bool, str]:
        """
        Check for exploding solutions (sign of numerical instability)
        """
        max_displacement = np.max(np.abs(x))
        max_velocity = np.max(np.abs(v))
        
        has_nan = np.any(np.isnan(x)) or np.any(np.isnan(v))
        has_inf = np.any(np.isinf(x)) or np.any(np.isinf(v))
        
        # Heuristic: if amplitude grows, suspect instability
        is_growing = np.mean(np.abs(x[-100:])) > np.mean(np.abs(x[:100])) if len(x) > 200 else False
        
        is_stable = not (has_nan or has_inf or is_growing)
        reason = "OK" if is_stable else ("NaN/Inf detected" if (has_nan or has_inf) else "Solution growing")
        
        return is_stable, reason


# ============================================================================
# 6. VISUALIZATION
# ============================================================================

def plot_results(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: DampedOscillatorParams, title_suffix: str = ""):
    """Generate 3 subplots: displacement, velocity, phase space"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Displacement vs Time
    axes[0].plot(t, x, 'b-', linewidth=1.5)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Displacement x (m)')
    axes[0].set_title('Displacement vs Time')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Velocity vs Time
    axes[1].plot(t, v, 'r-', linewidth=1.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Velocity v (m/s)')
    axes[1].set_title('Velocity vs Time')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Phase Space (x vs v)
    axes[2].plot(x, v, 'g-', linewidth=1.5)
    axes[2].plot(x[0], v[0], 'go', markersize=8, label='Start')
    axes[2].plot(x[-1], v[-1], 'rx', markersize=8, label='End')
    axes[2].set_xlabel('Position x (m)')
    axes[2].set_ylabel('Velocity v (m/s)')
    axes[2].set_title('Phase Space Trajectory')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Overall title with regime info
    fig.suptitle(f"Damped Oscillator — {params.damping_regime} (ζ={params.damping_ratio:.2f}) {title_suffix}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# 7. MAIN BENCHMARK EXECUTION
# ============================================================================

def run_benchmark():
    """
    Execute full benchmark: simulate all 3 damping regimes, validate, plot
    """
    
    # Define the 3 damping regimes
    scenarios = [
        ("Underdamped", DampedOscillatorParams(mass=1.0, damping_coeff=0.5, spring_constant=1.0)),
        ("Critically damped", DampedOscillatorParams(mass=1.0, damping_coeff=2.0, spring_constant=1.0)),
        ("Overdamped", DampedOscillatorParams(mass=1.0, damping_coeff=4.0, spring_constant=1.0)),
    ]
    
    results = {}
    
    for name, params in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {name}")
        print(f"  m={params.mass}, c={params.damping_coeff}, k={params.spring_constant}")
        print(f"  ζ (damping ratio) = {params.damping_ratio:.3f}")
        print(f"  ω₀ (natural freq) = {params.natural_frequency:.3f} rad/s")
        print(f"{'='*60}")
        
        # Run simulation
        simulator = Simulator(params, integrator_type="RK4", dt=0.01)
        t, x, v = simulator.run(t_span=20.0, initial_position=1.0, initial_velocity=0.0)
        
        # Validate
        print("\n✓ Validation Checks:")
        
        energy_valid, energy_increase = Validator.check_energy_conservation(x, v, params)
        print(f"  Energy conservation: {'PASS' if energy_valid else 'FAIL'} (max increase: {energy_increase:.2f}%)")
        
        amplitude_valid, amplitude_dev = Validator.check_amplitude_decay(x, params)
        print(f"  Amplitude decay: {'PASS' if amplitude_valid else 'FAIL'} (deviation: {amplitude_dev:.4f})")
        
        stability_valid, stability_msg = Validator.check_numerical_stability(x, v)
        print(f"  Numerical stability: {'PASS' if stability_valid else 'FAIL'} ({stability_msg})")
        
        # Store results
        results[name] = {
            't': t, 'x': x, 'v': v, 'params': params,
            'energy_valid': energy_valid, 'amplitude_valid': amplitude_valid, 'stability_valid': stability_valid
        }
        
        # Plot
        fig = plot_results(t, x, v, params)
        plt.savefig(f"oscillator_{name.replace(' ', '_')}.png", dpi=100, bbox_inches='tight')
        print(f"\n  📊 Saved plot: oscillator_{name.replace(' ', '_')}.png")
    
    return results


# ============================================================================
# 8. RUN EVERYTHING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DAMPED HARMONIC OSCILLATOR — BENCHMARK SUITE")
    print("="*60)
    
    results = run_benchmark()
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    for scenario, data in results.items():
        all_pass = data['energy_valid'] and data['amplitude_valid'] and data['stability_valid']
        status = "✅ PASS" if all_pass else "❌ FAIL"
        print(f"{scenario:20s} {status}")
    
    plt.show()

