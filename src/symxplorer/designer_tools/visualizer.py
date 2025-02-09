import logging
import sympy as sp
import numpy as np

import matplotlib.pyplot as plt
from   matplotlib.ticker import LogLocator
from   tqdm import tqdm
from typing import Dict, List, Tuple
from copy   import deepcopy

# Custom Imports
from symxplorer.symbolic_solver.domains import Filter_Classification

# Suppress info logging for matplotlib
logging.getLogger('matplotlib').setLevel(logging.ERROR)

class Symbolic_Visualizer:
    def __init__(self, filter_classification: Filter_Classification = None, tf: sp.Expr = None):
        """Either pass the TF or the filter classification object for more visualization"""
        if filter_classification is None and tf is None:
            raise NotImplementedError("Visualization works with only either a transfer function or a Filter_Classification. Cannot pass both at the same time.")
        
        self.classification_given = not (filter_classification is None)

        if self.classification_given:
            self.name: str            = f"{str(filter_classification.zCombo)}"
            self.tf_original: sp.Expr = filter_classification.transferFunc
            self.tf: sp.Expr          = filter_classification.transferFunc
            self.tf_params_original: Dict[str, sp.Expr] = deepcopy(filter_classification.parameters)
            self.tf_params: Dict[str, sp.Expr]          = deepcopy(filter_classification.parameters)
        else:
            self.name: str = "TF Visualizer"
            self.tf_original: sp.Expr = tf
            self.tf: sp.Expr          = tf
            self.tf_params_original: Dict[str, sp.Expr] = None
            self.tf_params: Dict[str, sp.Expr] = None


        s, f = sp.symbols('s f')
        self._str_to_param:    Dict[str, sp.Basic] = {str(sym) : sym for sym in self.tf.free_symbols if sym != s}
        self.params_to_value:  Dict[sp.Basic, float] = {s: 2 * sp.pi * sp.I * f}

        self.magnitude_expr: sp.Basic = None
        self.phase_expr: sp.Basic     = None

    def get_parameters(self) -> List[str]:
        return sorted([sym_key for sym_key in self._str_to_param.keys()])

    def is_defined_numerically(self) -> bool:
        for sym in self.tf.free_symbols:
            if self.params_to_value.get(sym) is None:
                return False
        return True
    
    def set_params(self, param_str_to_value: Dict[str, float]) -> Dict[sp.Basic, float]:

        for param_str, val in param_str_to_value.items():
            if self._str_to_param.get(param_str) is None:
                raise KeyError(f"{param_str} does not exist in the list of free symbols. Choose from {self._str_to_param.keys()}")
            self.params_to_value[self._str_to_param.get(param_str)] = val

        self.get_bode_expression() # To update the

        return self.params_to_value
    
    def set_equal_c(self) -> sp.Expr:
        sub_dict = {sym: sp.symbols("C", real=True, positive=True) for sym in self.tf.free_symbols if "C_" in str(sym)}
        self.tf = self.tf.subs(sub_dict)

        if self.classification_given:
            for param in self.tf_params.keys():
                if self.tf_params.get(param) is not None and isinstance(self.tf_params.get(param), sp.Basic):
                    self.tf_params[param] = self.tf_params[param].subs(sub_dict)
        
        # Update the design variables
        self.reset_params()
        return self.tf

    def set_equal_r(self) -> sp.Expr:
        sub_dict = {sym: sp.symbols("R", real=True, positive=True) for sym in self.tf.free_symbols if "R_" in str(sym)}

        self.tf = self.tf.subs(sub_dict)

        if self.classification_given:
            for param in self.tf_params.keys():
                if self.tf_params.get(param) is not None and isinstance(self.tf_params.get(param), sp.Basic):
                    self.tf_params[param] = self.tf_params[param].subs(sub_dict)

        # Update the design variables
        self.reset_params()
        return self.tf

    def reset(self) -> sp.Expr:
        self.tf        = deepcopy(self.tf_original)
        self.tf_params = deepcopy(self.tf_params_original)
        self.magnitude_expr: sp.Basic = None
        self.phase_expr: sp.Basic     = None
        self.reset_params()
        return self.tf
    
    def reset_params(self):
        s, f = sp.symbols('s f')
        self._str_to_param:    Dict[str, sp.Basic] = {str(sym) : sym for sym in self.tf.free_symbols if sym != s}
        self.params_to_value:  Dict[sp.Basic, float] = {s: 2 * sp.pi * sp.I * f}


    def get_bode_expression(self) -> Tuple[sp.Basic, sp.Basic, sp.Basic]:
        """Substitutes the parameters in self.params_to_value into the symbolic TF in self.tf, 
        and returns magnitude_expr, phase_expr, H_numeric."""

        if not self.is_defined_numerically():
            print("!!Set the parameters of the TF through --self.set_params--!!!")
            raise RuntimeError(f"Cannot evaluate the TF since the design parameters are not resolved. Provided {self.params_to_value} but need {self.tf.free_symbols}")

        H_numeric = self.tf.subs(self.params_to_value)

        # Compute the magnitude
        self.magnitude_expr = sp.Abs(H_numeric)  # Magnitude in dB

        # Compute the unwrapped phase (in degrees)
        self.phase_expr = sp.arg(H_numeric) * 180 / sp.pi         # Phase in degress

        return self.magnitude_expr, self.phase_expr, H_numeric

    
    def eval_freq(self, frequency: float) -> Tuple[float, float]:

        if self.magnitude_expr is None and self.phase_expr is  None:
            self.magnitude_expr, self.phase_expr, _ = self.get_bode_expression()

        f = sp.symbols('f')
        magnitude_val = self.magnitude_expr.subs(f, frequency).evalf()
        phase_val     = self.phase_expr.subs(f, frequency).evalf()

        return np.log10(float(sp.Abs(magnitude_val))), float(phase_val)
    
    def visualize(self, start_freq_order: float = 1, end_freq_order: float = 7, num_of_points: int = 20, title: str = ""):

        # Get the magnitude and phase expressions
        magnitude_expr, phase_expr, H_numeric = self.get_bode_expression()

        # Define the frequency range
        frequencies = np.logspace(start_freq_order, end_freq_order, num_of_points)

        # Evaluate the magnitude and phase for each frequency
        f = sp.symbols('f')

        magnitude_vals = [20*np.log10(float(sp.Abs(magnitude_expr.subs(f, freq).evalf()))) for freq in tqdm(frequencies, desc="Calculating Magnitudes", total=len(frequencies))]
        phase_vals     = [phase_expr.subs(f, freq).evalf() for freq in tqdm(frequencies, desc="Calculating Phases", total=len(frequencies))]
        phase_vals     = [float(p) for p in phase_vals]
        phase_vals     = np.unwrap(np.radians(phase_vals)) * 180 / np.pi

        # Plotting
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Magnitude Plot
        axs[0].semilogx(frequencies, [float(m) for m in magnitude_vals], label='Magnitude', color='blue', linewidth=1.5)
        axs[0].set_title(f"{title} - Magnitude Response")
        axs[0].set_xlabel("Frequency (Hz)")
        axs[0].set_ylabel("Magnitude (dB)")
        axs[0].grid(True, which='major', linestyle='-', linewidth=0.75)  # Major grid
        axs[0].grid(True, which='minor', linestyle=':', linewidth=0.5)  # Minor grid

        # Adjust minor ticks for log scale
        axs[0].xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
        axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), framealpha=0.8)

        # Phase Plot
        axs[1].semilogx(frequencies, [float(p) for p in phase_vals], label='Phase', color='orange', linewidth=1.5)
        axs[1].set_title(f"{title} - Phase Response")
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("Phase (degrees)")
        axs[1].grid(True, which='major', linestyle='-', linewidth=0.75)  # Major grid
        axs[1].grid(True, which='minor', linestyle=':', linewidth=0.5)  # Minor grid

        # Adjust minor ticks for log scale
        axs[1].xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
        axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), framealpha=0.8)


        # Show the plot with interactive features
        plt.tight_layout()
        plt.show()

    def get_filter_param(self):
        return sorted([param_name for param_name in self.tf_params.keys() if self.tf_params.get(param_name) is not None])

    def eval_filter_parameter(self, param_name: str, num_of_decimals: int = 3) -> Tuple[sp.Expr, float]:

        if self.tf_params is None:
            raise RuntimeError(f"The transfer function's filter parameters are not defined in self.tf_params")
        
        if not self.is_defined_numerically():
            print("!!Set the parameters of the TF through --self.set_params--!!!")
            raise RuntimeError(f"Cannot evaluate the TF since the design parameters are not resolved. provided {self.params_to_value} but need {self.tf.free_symbols}")

        if self.tf_params.get(param_name) is None:
            raise KeyError(f"Invalid filter parameter. Choose from {self.get_filter_param()}")

        expression = self.tf_params.get(param_name)
        value = expression.subs(self.params_to_value)

        return value, round(float(value.evalf()), num_of_decimals)

class Bode_Visualizer:
    def __init__(self, frequencies: np.ndarray, complex_response: np.ndarray):
        """
        Initialize the Bode_Visualizer with frequency data and complex response.

        Parameters:
        frequencies (np.ndarray): Array of frequency values in Hz.
        complex_response (np.ndarray): Array of complex response values (H(jw)).
        """
        self.frequencies = frequencies
        self.complex_response = complex_response

    def plot_bode(self):
        """
        Plot the Bode magnitude and phase plots.
        """
        magnitude = 20 * np.log10(np.abs(self.complex_response))  # Convert to dB
        phase = np.unwrap(np.angle(self.complex_response, deg=True))  # Convert to degrees

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        
        # Magnitude plot
        ax1.semilogx(self.frequencies, magnitude, 'b')
        ax1.set_title("Bode Magnitude Plot")
        ax1.set_ylabel("Magnitude (dB)")
        ax1.grid(True, which='both', linestyle='--')
        
        # Phase plot
        ax2.semilogx(self.frequencies, phase, 'r')
        ax2.set_title("Bode Phase Plot")
        ax2.set_ylabel("Phase (degrees)")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.grid(True, which='both', linestyle='--')
        
        plt.tight_layout()
        plt.show()