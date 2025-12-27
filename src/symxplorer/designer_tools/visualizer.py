import logging
import sympy as sp
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from   matplotlib.ticker import LogLocator

import plotly.express as px
import plotly.graph_objects as go

from   tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from copy   import deepcopy
from dataclasses import dataclass

# Custom Imports
from symxplorer.symbolic_exploration.domains import Filter_Classification

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
        """This function adds the given values to the substitution dictionary. it does not erase previous values. will need to call self.reset to clear past history"""
        for param_str, val in param_str_to_value.items():
            if self._str_to_param.get(param_str) is None:
                print(f"{param_str} does not exist in the list of free symbols. Choose from {self._str_to_param.keys()}")
                continue
                # raise KeyError(f"{param_str} does not exist in the list of free symbols. Choose from {self._str_to_param.keys()}")
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

        # if not self.is_defined_numerically():
        #     print("!!Set the parameters of the TF through --self.set_params--!!!")
        #     raise RuntimeError(f"Cannot evaluate the TF since the design parameters are not resolved. Provided {self.params_to_value} but need {self.tf.free_symbols}")

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
        """Returns an alphabetically sorted list of performance metrics of the filter's transfer function"""
        return sorted([param_name for param_name in self.tf_params.keys() if self.tf_params.get(param_name) is not None])

    def eval_filter_parameter(self, param_name: str, num_of_decimals: int = 3) -> Tuple[sp.Expr, float]:
        """Evaluates the given filter perfomance metric up to the given decimal point. Returns a tuple of symbolic expression and float. 
        If the result cannot be evaluated numerically the float value is np.nan."""
        if self.tf_params is None:
            raise RuntimeError(f"The transfer function's filter parameters are not defined in self.tf_params")
        
        expression = self.tf_params.get(param_name)
        if expression is None:
            raise KeyError(f"Invalid filter parameter. Choose from {self.get_filter_param()}")

        value = expression.subs(self.params_to_value)

        if not self.is_defined_numerically():
            print(f"Cannot numerically evaluate the TF since the design parameters are not resolved. provided {self.params_to_value} but need {self.tf.free_symbols}")
            return value, np.nan
            
        return value, round(float(value.evalf()), num_of_decimals)
    
    def print_filter_parameters(self, num_of_decimals: int = 3) -> Dict[str, float]:
        """Prints a summary of perfomance metrics (symbolic values are returned if there is not enough information to resolve this numerically)"""
        out : Dict[str, float]= {}
        print("printing parameters:")
        print("-------------------------------------")
        for filter_param in self.get_filter_param():
            expr, val = self.eval_filter_parameter(param_name=filter_param, num_of_decimals=num_of_decimals)
            if filter_param == "wo": # The hack to convert wo to f in Hz.
                filter_param = "f"
                val = val/(2*np.pi)
            print(f"\t{filter_param}:\t{expr if np.isnan(val) else val :.3e}")
            out[filter_param] = val
        print("-------------------------------------")
        return out

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

 
# -----------------------
# Core Data Structures
# -----------------------

@dataclass
class ExplorationPoint:
    design_var_dict: Dict[str, float]
    performance_metric_dict: Dict[str, float]


@dataclass
class DesignSpaceExploration:
    points: List[ExplorationPoint]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert exploration points into a DataFrame with x_ prefix for design vars."""
        records = []
        for p in self.points:
            design_prefixed = {f"x_{k}": v for k, v in p.design_var_dict.items()}
            record = {**design_prefixed, **p.performance_metric_dict}
            records.append(record)
        return pd.DataFrame(records)


# -----------------------
# Analysis Suite (Plotly)
# -----------------------

class AnalysisSuite:
    def __init__(self, exploration: DesignSpaceExploration):
        self.df = exploration.to_dataframe()
        self.design_vars = [c for c in self.df.columns if c.startswith("x_")]
        self.metrics = [c for c in self.df.columns if not c.startswith("x_")]

    # ------------
    # Sensitivity (correlation)
    # ------------
    def plot_sensitivity(self, metric_name: str):
        corr = self.df[self.design_vars + [metric_name]].corr()[metric_name].drop(metric_name)
        corr_sorted = corr.sort_values(ascending=False)
        fig = px.bar(
            x=corr_sorted.values,
            y=corr_sorted.index,
            orientation='h',
            labels={'x': 'Correlation', 'y': 'Design Variable'},
            title=f"Sensitivity of {metric_name}",
        )
        fig.update_layout(yaxis=dict(categoryorder='total ascending'))
        fig.show()

    # ------------
    # Slice plot (1D sweep)
    # ------------
    def plot_slice(self, param: str, metric_name: str):
        fig = px.scatter(
            self.df,
            x=param,
            y=metric_name,
            trendline="lowess",
            title=f"{metric_name} vs {param}",
        )
        fig.show()

    # ------------
    # Contour plot (2D)
    # ------------
    def plot_contour(self, x_param: str, y_param: str, metric_name: str):
        df = self.df[[x_param, y_param, metric_name]].dropna()
        fig = go.Figure(
            data=go.Contour(
                x=df[x_param],
                y=df[y_param],
                z=df[metric_name],
                colorscale="Viridis",
                contours=dict(showlabels=True)
            )
        )
        fig.update_layout(
            title=f"{metric_name} Contour over {x_param} and {y_param}",
            xaxis_title=x_param,
            yaxis_title=y_param,
        )
        fig.show()

    # ------------
    # Progression plot (metric vs trial index)
    # ------------
    def plot_progression(self, metric_name: str):
        df = self.df.reset_index().rename(columns={"index": "iteration"})
        fig = px.line(
            df,
            x="iteration",
            y=metric_name,
            markers=True,
            title=f"Progression of {metric_name}",
        )
        fig.show()

    # ------------
    # Parallel coordinates plot (multi-metric + design vars)
    # ------------
    def plot_parallel(self, metrics: Optional[List[str]] = None):
        metrics = metrics or self.metrics
        fig = px.parallel_coordinates(
            self.df,
            dimensions=self.design_vars + metrics,
            color=metrics[0],
            color_continuous_scale=px.colors.sequential.Viridis,
            title="Parallel Coordinates (Design Variables + Metrics)"
        )
        fig.show()

    # ------------
    # Trade-off plot (2D metrics)
    # ------------
    def plot_tradeoff(self, metric1: str, metric2: str):
        """
        Creates an interactive 2D scatter plot to visualize the trade-off between two performance metrics.
        Hovering over a point reveals the design variables and other metrics for that exploration point.
        """
        if metric1 not in self.metrics or metric2 not in self.metrics:
            raise ValueError(f"Invalid metric names. Choose from: {self.metrics}")

        # All metrics and design variables will be available on hover
        hover_data = self.design_vars + [m for m in self.metrics if m not in [metric1, metric2]]

        fig = px.scatter(
            self.df,
            x=metric1,
            y=metric2,
            hover_data=hover_data,
            title=f"Trade-off between {metric1} and {metric2}",
        )
        fig.update_traces(
            mode='markers',
            marker=dict(
                size=10,
                opacity=0.7,
                line=dict(width=1, color='DarkSlateGrey')
            )
        )
        fig.update_layout(
            xaxis_title=metric1,
            yaxis_title=metric2,
            title_x=0.5
        )
        fig.show()

    # ------------
    # 3D Scatter plot (2 design vars vs. 1 metric)
    # ------------
    def plot_3d_scatter(self, design_var1: str, design_var2: str, metric_name: str):
        """
        Creates an interactive 3D scatter plot of two design variables against a specified metric.
        Hovering over a point reveals other metrics and design variables.
        """
        if design_var1 not in self.design_vars or design_var2 not in self.design_vars:
            raise ValueError(f"Invalid design variable names. Choose from: {self.design_vars}")
        if metric_name not in self.metrics:
            raise ValueError(f"Invalid metric name. Choose from: {self.metrics}")

        hover_data = self.design_vars + self.metrics

        fig = px.scatter_3d(
            self.df,
            x=design_var1,
            y=design_var2,
            z=metric_name,
            color=metric_name,
            color_continuous_scale=px.colors.sequential.Viridis,
            hover_data=hover_data,
            title=f"{metric_name} vs. {design_var1} and {design_var2}",
        )
        fig.update_layout(
            scene=dict(
                xaxis_title=design_var1,
                yaxis_title=design_var2,
                zaxis_title=metric_name,
            ),
            title_x=0.5,
        )
        fig.show()

    # ------------
    # Distribution plot (histogram)
    # ------------
    def plot_distribution(self, metric_name: str):
        """
        Creates an interactive histogram to visualize the distribution of a single performance metric.
        """
        if metric_name not in self.metrics:
            raise ValueError(f"Invalid metric name. Choose from: {self.metrics}")

        fig = px.histogram(
            self.df,
            x=metric_name,
            marginal="box",  # or "rug", "violin"
            title=f"Distribution of {metric_name}",
        )
        fig.update_layout(
            xaxis_title=metric_name,
            yaxis_title="Frequency",
            title_x=0.5
        )
        fig.show()

    # ------------
    # Box plot (metric vs. design var)
    # ------------
    def plot_box(self, design_var: str, metric_name: str):
        """
        Creates an interactive box plot of a metric grouped by a design variable.
        """
        if design_var not in self.design_vars:
            raise ValueError(f"Invalid design variable name. Choose from: {self.design_vars}")
        if metric_name not in self.metrics:
            raise ValueError(f"Invalid metric name. Choose from: {self.metrics}")

        fig = px.box(
            self.df,
            x=design_var,
            y=metric_name,
            title=f"Distribution of {metric_name} across {design_var}",
        )
        fig.update_layout(
            xaxis_title=design_var,
            yaxis_title=metric_name,
            title_x=0.5
        )
        fig.show()

