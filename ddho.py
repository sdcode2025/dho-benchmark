import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple

# ============================================================================
# 1. UPDATED PHYSICS PARAMETERS
# ============================================================================

@dataclass
class DrivenOscillatorParams:
    mass: float = 1.0
    damping_coeff: float = 0.2    # Lower damping to see resonance clearly
    spring_constant: float = 1.0   # Natural freq ω0 = 1.0 rad/s
    f_amplitude: float = 0.5       # Driving force magnitude (F0)
    f_frequency: float = 1.2       # Driving frequency (ω)

    @property
    def natural_frequency(self) -> float:
        return np.sqrt(self.spring_constant / self.mass)

# ============================================================================
# 2. UPDATED ODE SYSTEM (Time-Dependent)
# ============================================================================

class DrivenODESystem:
    def __init__(self, params: DrivenOscillatorParams):
        self.p = params

    def state_derivative(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Calculates: 
        dx/dt = v
        dv/dt = [F0*cos(ωt) - c*v - k*x] / m
        """
        x, v = state
        
        # External driving force at time t
        f_ext = self.p.f_amplitude * np.cos(self.p.f_frequency * t)
        
        dxdt = v
        dvdt = (f_ext - self.p.damping_coeff * v - self.p.spring_constant * x) / self.p.mass
        
        return np.array([dxdt, dvdt])

# ============================================================================
# 3. UPDATED RK4 INTEGRATOR (Time-Aware)
# ============================================================================

class RK4Integrator:
    def __init__(self, ode_system: DrivenODESystem, dt: float = 0.01):
        self.ode = ode_system
        self.dt = dt

    def step(self, state: np.ndarray, t: float) -> np.ndarray:
        dt = self.dt
        # RK4 steps now include time t
        k1 = self.ode.state_derivative(state, t)
        k2 = self.ode.state_derivative(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.ode.state_derivative(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.ode.state_derivative(state + dt * k3, t + dt)
        
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ============================================================================
# 4. SIMULATION ENGINE & VISUALIZATION
# ============================================================================

def run_driven_simulation(t_max=50.0, dt=0.01):
    # Setup parameters (Case: Near Resonance)
    params = DrivenOscillatorParams(f_frequency=1.1) # Natural freq is 1.0
    ode = DrivenODESystem(params)
    integrator = RK4Integrator(ode, dt)
    
    # Initial state [x, v]
    state = np.array([0.0, 0.0])
    
    t_hist, x_hist, f_hist = [], [], []
    
    for step in range(int(t_max/dt)):
        t = step * dt
        t_hist.append(t)
        x_hist.append(state[0])
        f_hist.append(params.f_amplitude * np.cos(params.f_frequency * t))
        
        state = integrator.step(state, t)
        
    return np.array(t_hist), np.array(x_hist), np.array(f_hist), params

# Plotting
t, x, f_ext, p = run_driven_simulation()

fig, ax1 = plt.subplots(figsize=(12, 5))

# Plot Displacement
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Displacement (x)', color='tab:blue')
ax1.plot(t, x, label='Oscillator Position (x)', color='tab:blue', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Plot Driving Force on same graph to see Phase Lag
ax2 = ax1.twinx()
ax2.set_ylabel('Driving Force (F)', color='tab:red', alpha=0.5)
ax2.plot(t, f_ext, label='Driving Force (F)', color='tab:red', linestyle='--', alpha=0.4)
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title(f"Driven Damped Oscillator: ω={p.f_frequency}, ω₀={p.natural_frequency}")
fig.tight_layout()
plt.show()
