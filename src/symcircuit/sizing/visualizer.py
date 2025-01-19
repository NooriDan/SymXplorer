import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

# Define the design variables
R1, C1, s, f = sp.symbols('R1 C1 s f')

# Define your transfer function (example: a simple RC low-pass filter)
H = (R1 * C1 * s) / (R1 * C1 * s + 1)

# Substitute design variable values (example)
R1_val = 1e3  # Ohms
C1_val = 1e-9  # Farads

# Define the transfer function in terms of frequency
H_numeric = H.subs({R1: R1_val, C1: C1_val, s: 2 * sp.pi * sp.I * f})

# Compute the magnitude (in dB) and phase (in degrees)
magnitude_expr = 20 * sp.log(abs(H_numeric), 10)  # Magnitude in dB
phase_expr = sp.arg(H_numeric) * 180 / sp.pi   # Phase in degrees

# Define the frequency range
frequencies = np.logspace(1, 6, 100)  # From 1 Hz to 1 MHz

# Evaluate the magnitude and phase for each frequency
magnitude_vals = [magnitude_expr.subs(f, freq).evalf() for freq in frequencies]
phase_vals = [phase_expr.subs(f, freq).evalf() for freq in frequencies]

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot Magnitude
axs[0].semilogx(frequencies, [float(m) for m in magnitude_vals], label='Magnitude')
axs[0].set_title("Bode Plot - Magnitude")
axs[0].set_xlabel("Frequency (Hz)")
axs[0].set_ylabel("Magnitude (dB)")
axs[0].grid(True)  # Add grid
axs[0].legend()

# Plot Phase
axs[1].semilogx(frequencies, [float(p) for p in phase_vals], label='Phase')
axs[1].set_title("Bode Plot - Phase")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel("Phase (degrees)")
axs[1].grid(True)  # Add grid
axs[1].legend()

# Show the plot with interactive features
plt.tight_layout()
plt.show()
