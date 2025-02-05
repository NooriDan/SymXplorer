import torch
import numpy  as np
import control as ctrl
import sympy  as sp
from   typing import Dict, List, Tuple

# Plotting Tools
import plotly.graph_objects as go
from   plotly.subplots import make_subplots

def weighted_mse_loss(
    response: torch.Tensor, 
    target_response: torch.Tensor, 
    weights: torch.Tensor, 
    normalize_method: str = None, 
    epsilon: float = 1e-10
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    """Computes the weighted mean squared error loss between the response and the target response."""

    # Clamp response and target_response to avoid log10 instability for phase response
    if epsilon > 0: 
        response = torch.log10(torch.clamp(response, min=epsilon))
        target_response = torch.log10(torch.clamp(target_response, min=epsilon))

    norm_params = {}

    # weights = weights / torch.sum(weights)  # Normalize weights

    if normalize_method is None:
        return torch.mean(weights * (response - target_response) ** 2), norm_params

    elif normalize_method == "z-score":
        mean = torch.mean(target_response)
        std = torch.std(target_response)

        # Avoid division by zero
        std = torch.clamp(std, min=epsilon)

        target_response_norm = (target_response - mean) / std
        response_norm = (response - mean) / std

        norm_params = {"mean": mean, "std": std}

    elif normalize_method == "min-max":
        min_val = torch.min(target_response)
        max_val = torch.max(target_response)

        # Avoid division by zero
        range_val = torch.clamp(max_val - min_val, min=epsilon)

        target_response_norm = (target_response - min_val) / range_val
        response_norm = (response - min_val) / range_val

        norm_params = {"min": min_val, "max": max_val}

    else:
        raise ValueError("Invalid normalization method. Choose 'z-score' or 'min-max' or None.")

    return torch.mean(weights * (response_norm - target_response_norm) ** 2), norm_params


# Plotting 
def plot_ac_response(frequencies: torch.Tensor, H_f_list: list, phase_list: list, labels: list = None, title: str = "Frequency Response"):
    """Plots multiple AC responses on the same plot using Plotly for interactivity.

    Args:
        frequencies: A PyTorch tensor of frequencies (shared by all responses).
        H_f_list: A list of PyTorch tensors, each representing the complex frequency response (H_f) of a circuit.
        phase_list: A list of PyTorch tensors, each representing the phase response of a circuit.
        labels: A list of strings, each representing the label for a circuit's response.
        title: The title of the plot.
    """

    if labels is None:
        labels = [f"Series {i}" for i in range(len(H_f_list))]

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Gain (dB)", "Phase (deg)"))
    helper = Transfer_Func_Helper()
    for H_f, phase, label in zip(H_f_list, phase_list, labels):
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

    def get_ac_response_from_symbolic(self, tf: sp.Expr, frequencies: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        H_f = self.eval_tf(tf, frequencies)
        return torch.abs(H_f), torch.tensor(np.unwrap(torch.angle(H_f)) * 180.0 / np.pi)
    
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

