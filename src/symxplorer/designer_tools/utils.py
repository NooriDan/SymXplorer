import torch
import numpy  as np
import control as ctrl
import sympy  as sp
from   typing import Dict, List, Tuple

# Plotting Tools
import plotly.graph_objects as go
from   plotly.subplots import make_subplots

UNIT_DICT: Dict[str, float] ={
    'p' : 1e-12,
    'n' : 1e-9,
    'u' : 1e-6,
    'k' : 1e3
}

def weighted_mse_loss(
    response: torch.Tensor, 
    target_response: torch.Tensor, 
    weights: torch.Tensor, 
    normalize_method: str = None, 
    epsilon: float = 1e-10
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    """Computes the weighted mean squared error loss between the response and the target response."""

    norm_params = {}

    if normalize_method is None:
        norm_params = None
        loss = torch.mean(weights * (response - target_response) ** 2)

    elif normalize_method == "z-score":
        mean = torch.mean(target_response)
        std = torch.std(target_response)

        # Avoid division by zero
        std = torch.clamp(std, min=epsilon)

        target_response_norm = (target_response - mean) / std
        response_norm = (response - mean) / std

        norm_params = {"mean": mean, "std": std}
        loss = torch.mean(weights * (response_norm - target_response_norm) ** 2)

    elif normalize_method == "min-max":
        min_val = torch.min(target_response)
        max_val = torch.max(target_response)

        norm_params = {"min": min_val, "max": max_val}

        loss = torch.mean(weights * (response - target_response) ** 2 / ((max_val-min_val)** 0.5))

    else:
        raise ValueError("Invalid normalization method. Choose 'z-score' or 'min-max' or None.")

    return loss, norm_params

def weighted_mae_loss(
    response: torch.Tensor, 
    target_response: torch.Tensor, 
    weights: torch.Tensor, 
    normalize_method: str = None, 
    epsilon: float = 1e-10
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    """Computes the weighted absolute error loss between the response and the target response."""

    norm_params = {}

    if normalize_method is None:
        return torch.mean(weights * torch.abs(response - target_response)), norm_params

    elif normalize_method == "z-score":
        mean = torch.mean(target_response)
        std = torch.std(target_response)

        # Avoid division by zero
        std = torch.clamp(std, min=epsilon)

        target_response_norm = (target_response - mean) / std
        response_norm = (response - mean) / std

        norm_params = {"mean": mean, "std": std}
        loss = torch.mean(weights * torch.abs(response_norm - target_response_norm))

    elif normalize_method == "min-max":
        min_val = torch.min(target_response)
        max_val = torch.max(target_response)

        norm_params = {"min": min_val, "max": max_val}
        loss = torch.mean(weights * torch.abs(response - target_response)/ ((max_val-min_val)** 0.5))

    else:
        raise ValueError("Invalid normalization method. Choose 'z-score' or 'min-max' or None.")

    return loss, norm_params

def get_bode_fitness_loss( current_complex_response: torch.Tensor, target_complex_response: torch.Tensor, freq_weights: torch.Tensor = None, loss_type: str = 'mae',norm_method: str = "min-max", rescale:bool = True, epsilon: float = 1e-10) -> Dict[str, torch.Tensor]:
    # Ensure inputs are tensors
    if not isinstance(current_complex_response, torch.Tensor):
        current_complex_response = torch.tensor(current_complex_response, dtype=torch.cfloat)
    if not isinstance(target_complex_response, torch.Tensor):
        target_complex_response = torch.tensor(target_complex_response, dtype=torch.cfloat)
    
    # Set freq_weights to an array of ones if not provided, matching the dtype of current_complex_response
    if freq_weights is None:
        freq_weights = torch.ones_like(current_complex_response, dtype=torch.float64)

    helper = Transfer_Func_Helper()

    # Extract magnitude and phase
    curr_mag, curr_phase     = helper.get_mag_phase_from_complex_response(current_complex_response)
    target_mag, target_phase = helper.get_mag_phase_from_complex_response(target_complex_response)
    
    fit_summary = {}
    if rescale:
        curr_max_mag    = torch.max(curr_mag)
        target_max_mag  = torch.max(target_mag)

        # mag is in dB so we normalize to
        curr_mag   -= curr_max_mag  
        target_mag -= target_max_mag

        # log in the summary
        fit_summary['curr_max_mag'] = curr_max_mag
        fit_summary['target_max_mag'] = target_max_mag

    # Compute losses
    if loss_type == 'mae':
        mag_loss, _   = weighted_mae_loss(curr_mag, target_mag, freq_weights, norm_method, epsilon=epsilon)
        phase_loss, _ = weighted_mae_loss(curr_phase, target_phase, freq_weights, norm_method, epsilon=epsilon) 
    elif loss_type == 'mse':
        mag_loss, _   = weighted_mse_loss(curr_mag, target_mag, freq_weights, norm_method, epsilon=epsilon)
        phase_loss, _ = weighted_mse_loss(curr_phase, target_phase, freq_weights, norm_method, epsilon=epsilon)
    else:
        raise KeyError(f"{loss_type} is a loss type option... choose from: ['mae', 'mse']")  
    
    fit_summary['mag_loss']   = mag_loss
    fit_summary['phase_loss'] = phase_loss

    return fit_summary

# Plotting 
def plot_ac_response(frequencies: torch.Tensor, mag_list: list, phase_list: list, labels: list = None, title: str = "Frequency Response"):
    """Plots multiple AC responses on the same plot using Plotly for interactivity.

    Args:
        frequencies: A PyTorch tensor of frequencies (shared by all responses).
        H_f_list: A list of PyTorch tensors, each representing the complex frequency response (H_f) of a circuit.
        phase_list: A list of PyTorch tensors, each representing the phase response of a circuit.
        labels: A list of strings, each representing the label for a circuit's response.
        title: The title of the plot.
    """

    if labels is None:
        labels = [f"Series {i}" for i in range(len(mag_list))]

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Gain (dB)", "Phase (deg)"))
    helper = Transfer_Func_Helper()
    for H_f, phase, label in zip(mag_list, phase_list, labels):
        fig.add_trace(go.Scatter(x=frequencies.tolist(), y=helper.convert_to_dB(H_f).tolist(), mode='lines', name=f"{label}-mag"), row=1, col=1)  # Gain
        fig.add_trace(go.Scatter(x=frequencies.tolist(), y=phase.tolist(), mode='lines', name=f"{label}-phase"), row=2, col=1)  # Phase

    fig.update_layout(
        title=title,
        xaxis_type="log",  # Logarithmic frequency axis
        xaxis_title="Frequency (Hz)",
        yaxis_title="Gain (dB)",
        xaxis2_type="log", # Logarithmic frequency axis for phase plot
        xaxis2_title="Frequency (Hz)",
        yaxis2_title="Phase (deg)",
        height=800,  # Set the height (in pixels) - Increase this value
        width=1000   # Set the width (in pixels)
    )

    fig.show()

def plot_complex_response(frequencies: torch.Tensor, complex_response_list: list, labels: list = None, title: str = "Frequency Response"):
    """Plots multiple AC responses on the same plot using Plotly for interactivity.

    Args:
        frequencies: A PyTorch tensor of frequencies (shared by all responses).
        complex_response_list: A list of PyTorch tensors, each representing the complex frequency response (H_f) of a circuit.
        labels: A list of strings, each representing the label for a circuit's response.
        title: The title of the plot.
    """

    if labels is None:
        labels = [f"Series {i}" for i in range(len(complex_response_list))]

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Gain (dB)", "Phase (deg)"))

    helper = Transfer_Func_Helper()

    for complex_response, label in zip(complex_response_list, labels):

        magnitude_dB, phase_deg = helper.get_mag_phase_from_complex_response(complex_response)

        # Plot magnitude response
        fig.add_trace(go.Scatter(
            x=frequencies.tolist(), 
            y=magnitude_dB.tolist(), 
            mode='lines', 
            name=f"{label}-mag"
        ), row=1, col=1)

        # Plot phase response
        fig.add_trace(go.Scatter(
            x=frequencies.tolist(), 
            y=phase_deg.tolist(), 
            mode='lines', 
            name=f"{label}-phase"
        ), row=2, col=1)

    fig.update_layout(
        title=title,
        xaxis_type="log",  # Logarithmic frequency axis
        xaxis_title="Frequency (Hz)",
        yaxis_title="Gain (dB)",
        xaxis2_type="log",  # Logarithmic frequency axis for phase plot
        xaxis2_title="Frequency (Hz)",
        yaxis2_title="Phase (deg)",
        height=800,  # Set the height (in pixels)
        width=1000   # Set the width (in pixels)
    )

    fig.show()

# Helper Functions
class Transfer_Func_Helper:
    def __init__(self):
        pass

    def convert_from_dB(self, val: torch.Tensor) -> torch.Tensor:
        return torch.pow(10, val / 20)

    def convert_to_dB(self, val: torch.Tensor) -> torch.Tensor:
        return 20 * torch.log10(val)

    def convert_to_omega(self, f: torch.Tensor) -> torch.Tensor:
        return 2 * torch.pi * f

    def convert_to_f(self, omega: torch.Tensor) -> torch.Tensor:
        return omega / (2 * torch.pi)

    def eval_tf(self, tf: sp.Expr, f_val: torch.Tensor) -> torch.Tensor:
        s = sp.symbols("s")
        
        # Convert torch tensor to NumPy before passing to lambdify
        H_f = sp.lambdify(s, tf, "numpy")
        f_numpy = f_val.cpu().numpy()  # Ensure f_val is a NumPy array
        
        # Evaluate transfer function
        H_result = H_f(f_numpy * 2 * np.pi * 1j)  # Keep NumPy operations
        
        # Convert back to PyTorch tensor using torch.from_numpy
        return torch.from_numpy(np.asarray(H_result, dtype=np.complex64)).to(f_val.device)
    
    def get_mag_phase_from_complex_response(self, complex_response_array: torch.Tensor, epsilon: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
        mag   = 20 * torch.log10(torch.clamp(torch.abs(complex_response_array), min=epsilon))
        phase = torch.tensor(np.unwrap(torch.angle(complex_response_array)) * 180.0 / np.pi)
        return mag, phase

    def get_ac_response_from_symbolic(self, tf: sp.Expr, frequencies: torch.Tensor, epsilon: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
        complex_response_array = self.eval_tf(tf, frequencies)
        mag, phase  = self.get_mag_phase_from_complex_response(complex_response_array=complex_response_array, epsilon=epsilon)
        return mag, phase
    
    def control_tf_to_sympy(self, tf_sys):
        """
        Converts a control.TransferFunction to a sympy symbolic transfer function.
        
        Parameters:
        tf_sys (control.TransferFunction): The transfer function from the control module.
        
        Returns:
        sympy.Expr: The symbolic transfer function H(s).
        """
        s = sp.symbols('s')  # Define the Laplace variable

        # Extract numerator and denominator coefficients
        num_coeffs = tf_sys.num[0][0]  # Extract numerator coefficients
        den_coeffs = tf_sys.den[0][0]  # Extract denominator coefficients

        # Construct symbolic numerator and denominator polynomials
        num_expr = sum(c * s**i for i, c in enumerate(reversed(num_coeffs)))
        den_expr = sum(c * s**i for i, c in enumerate(reversed(den_coeffs)))

        # Construct and return the symbolic transfer function
        return num_expr / den_expr

    def sympy_tf_to_control(self, H_s):
        """
        Converts a sympy symbolic transfer function to a control.TransferFunction.
        
        Parameters:
        H_s (sympy.Expr): The symbolic transfer function.
        s (sympy.Symbol): The Laplace variable.
        
        Returns:
        control.TransferFunction: Equivalent transfer function in the control module.
        """
        s = sp.symbols("s")

        # Get numerator and denominator
        num_expr, den_expr = sp.fraction(H_s)  # Extract numerator and denominator

        # Convert to polynomials
        num_poly = sp.Poly(num_expr, s)
        den_poly = sp.Poly(den_expr, s)

        # Get coefficients in order of decreasing powers
        num_coeffs = [float(c) for c in num_poly.all_coeffs()]
        den_coeffs = [float(c) for c in den_poly.all_coeffs()]

        # Create control.TransferFunction
        return ctrl.TransferFunction(num_coeffs, den_coeffs)
class Frequency_Weight:
    def __init__(self, lower: float, upper: float, frequency_array: torch.Tensor = None, bias: float = 10):
        """
        Initializes the Frequency_Weight object.

        Args:
            lower (float): The lower bound of the frequency range to get the bias.
            upper (float): The upper bound of the frequency range to get the bias.
            frequency_array (torch.Tensor): The input tensor of frequencies.
            bias (float, optional): The weight assigned to frequencies within the bounds. Default is 10.
        """
        self.lower = lower
        self.upper = upper
        self.bias  = bias
        self.parent_frequency_array = frequency_array
        if frequency_array is not None:
            self.weights = torch.where((frequency_array >= lower) & (frequency_array <= upper), bias, torch.tensor(1.0))
        else: 
            self.weights = None

    def compute_weights(self) -> torch.Tensor:
        if self.parent_frequency_array is not None:
            self.weights = torch.where((self.parent_frequency_array >= self.lower) & (self.parent_frequency_array <= self.upper), self.bias, torch.tensor(1.0))
            return self.weights
        return None

    def __add__(self, other):
        """
        Takes the element-wise maximum of the weight tensors of two Frequency_Weight objects.

        Args:
            other (Frequency_Weight): Another Frequency_Weight object.

        Returns:
            Frequency_Weight: A new object with the maximum weights at each index.
        """
        if not isinstance(other, Frequency_Weight):
            raise TypeError("Can only add Frequency_Weight objects.")
        
        if self.parent_frequency_array is None:
            return other

        new_obj = Frequency_Weight(frequency_array=self.parent_frequency_array, lower=-1, upper=-1)  # Dummy instance
        new_obj.weights = torch.maximum(self.weights, other.weights)
        return new_obj

    def __repr__(self):
        return f"Frequency_Weight(weights={self.weights})"
