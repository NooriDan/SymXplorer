import logging
import os
import torch
import sympy  as sp
from   typing import Dict, List, Tuple, Any
from   tqdm   import tqdm
from   datetime import datetime
import nevergrad as ng
import plotly.graph_objects as go

from abc import ABC, abstractmethod

# Nevergrad Import
import nevergrad as ng


# Symxplorer Specific Imports
from   symxplorer.spice_engine.spicelib   import LTspice_Wrapper
from   symxplorer.spice_engine.spicelib   import Spicelib_Wrapper, Sim_Execution_Type

from   .symbolic_sizing import Symbolic_Sizing_Assist
from   .utils           import  plot_complex_response, get_bode_fitness_loss, Transfer_Func_Helper

s = sp.symbols("s")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype  = torch.double

torch.set_default_dtype(dtype)
torch.set_default_device(device)
print(f'Using device: {device} and dtype: {dtype}')

class Nevergrad_Base_Optimizer(ABC):
    def __init__(self,
                 budget: int = 10,
                 optimizer_name: str = "CMA",
                 random_seed: int = 42,
                 ):
        
        self.budget = budget
        self.optimizer_name = optimizer_name
        self.random_seed    = random_seed
        self.optimizer: ng.optimization.base.Optimizer | None = None
        self.optimizer_trace: List[Dict[str, Any]] = []
        self.loss_values : List[float] = []
        self.global_best_index: int = 0 # the index of the global best solution
        self.parametrization: ng.p.Dict | None = None
    
    def create_experiment(self, overwrite_optimizer:ng.optimization.base.Optimizer | None = None) -> bool:
        if self.parametrization is None:
            print("NEED TO CALL self.parameterize")
            return False

        elif overwrite_optimizer is not None:
            self.optimizer = overwrite_optimizer
            return True
        else:
            registry = ng.optimizers.registry.get(self.optimizer_name)
            if registry is not None:
                self.optimizer = registry(parametrization=self.parametrization, budget=self.budget)
                print(f"Optimizer is set to {self.optimizer.name} with budget = {self.budget}")
                return True
        return False
    
    @abstractmethod
    def parameterize(self, log_scale: bool = True) -> Tuple[Dict, Dict]:
        """Returns the parametrization dictionary for nevergrad and any denormalization factors needed"""
        pass

    @abstractmethod
    def denormalize_params(self, parameterization: Dict[str, float]) -> Dict[str, float]:
        """Convert the normalized parameters back to the original scale"""
        pass

    @abstractmethod
    def evaluate(self, parameterization: Dict[str, float]) -> float:
        """Evaluate the objective function for the given parameterization"""
        # 1 - Need to denormalize into their original scale
        parameterization_denorm = self.denormalize_params(parameterization)
        # TODO: The rest is to implemented in the subclass
        pass

        
    def optimize(self, budget: int, render_optimization_trace: bool = True) -> List[Dict[str, Any]] | None:
        """Run the optimization process for a given budget and returns the optimization trace as 
        a list of (parameterization, loss) tuples"""
        if self.optimizer is None:
            print("Need to set the optimizer by calling self.create_experiment")
            return None
        
        # Track the loss for plotting
        self.loss_values : List[float] = []
        self.optimizer_trace = []  # Store the optimization trace
        
        # Run the optimization process
        for trial in tqdm(range(budget), desc="Optimizing", unit="trial"):
            # Get a new candidate
            candidate : ng.p.Parameter = self.optimizer.ask()
            # Evaluate function
            curr_loss : float = self.evaluate(parameterization=candidate.value)
            # Provide feedback to optimizer
            self.optimizer.tell(candidate, curr_loss)
            
            # Log the achieved loss
            self.optimizer_trace.append({
                "params" : candidate.value, 
                "loss" : curr_loss
                })

            # Update the index of the global best solution (lowest loss)
            if curr_loss < self.optimizer_trace[self.global_best_index]["loss"]:
                self.global_best_index = trial
        
        # Plot the loss as a function of optimization step
        if render_optimization_trace:
            self._plot_loss()

        return self.optimizer_trace
    
    def get_best_params(self) -> Tuple[Dict[str, float], float] | None:

        if self.optimizer is None:
            print("Need to set the optimizer by calling self.create_experiment")
            return
        if len(self.optimizer_trace) < 1:
            print("need to run self.optimize")
            return
        
        point = self.optimizer_trace[self.global_best_index]
        best_solution : ng.p.Parameter = point['params']
        loss : float = point['loss']

        print("Optimized x - normalized:", best_solution.value)
        print("Optimized x - de-normalized:", self.denormalize_params(best_solution.value))
        print("loss:", loss)

        return self.denormalize_params(best_solution.value), loss
    
    def _plot_loss(self):
        """Plot the loss as a function of optimization steps with Plotly."""
        fig = go.Figure()

        if len(self.optimizer_trace) < 1:
            print("No optimization trace to plot")
            return
        loss_values = [entry["loss"] for entry in self.optimizer_trace]
        # Automatically generate x values (0, 1, 2, ..., n-1)
        x_values = list(range(len(loss_values)))
        # Add a line plot with trials on the x-axis and loss_values on the y-axis
        fig.add_trace(go.Scatter(x=x_values, y=loss_values, mode='markers+lines', name='Loss', line=dict(color='blue', width=2)))

        # Add title and labels
        fig.update_layout(
            title='Loss vs. Optimization Trial',
            xaxis_title='Optimization Step',
            yaxis_title='Loss',
            template='plotly_dark',  # Optional: Use dark theme for the plot
            showlegend=True
        )
        
        # Show the interactive plot
        fig.show()

class Nevergrad_Symbolic_Bode_Fitter:
    def __init__(self, 
                 tf_to_size: sp.Expr,
                 target_tf: sp.Expr,
                 c_range: List[float] = [1e-12, 1e-9], 
                 r_range: List[float] = [1e2, 1e5],
                 frequencies: torch.Tensor = torch.logspace(3, 8, 1000),
                 freq_weights: torch.Tensor = None,
                 max_loss: float = 20,
                 loss_norm_method: str = "min-max",
                 loss_type:     str = "mse",
                 optimizer_name: str = "CMA",
                 rescale_mag: bool = True,
                 random_seed: int = 42,
                 verbose_logging: bool = True
                 ):
        

        self.sizing_assist = Symbolic_Sizing_Assist(tf=tf_to_size)
        self.target_tf = target_tf
        self.c_range = c_range
        self.r_range = r_range
        self.frequencies = frequencies
        self.freq_weights = freq_weights if freq_weights is not None else torch.ones_like(frequencies)
        self.max_mse_loss = max_loss
        self.loss_norm_method  = loss_norm_method
        self.loss_type = loss_type
        self.optimizer_name = optimizer_name
        self.rescale_mag: bool = rescale_mag
        self.random_seed  = random_seed
        self.verbose_logging = verbose_logging
        
        self.parametrization: ng.p.Dict = None
        self.cap_denormailization: float  = None
        self.res_denormailization: float  = None
        self.helper_functions = Transfer_Func_Helper()
        self.optimizer: ng.optimization.base.Optimizer = None
        self.optimizer_trace: List[Tuple[ng.p.Dict, float]] = []
        self.global_min_index: int = 0 # the index of the global min

        self._default_var_bounds = [1, 100]

    def parameterize(self, log_scale: bool = True) -> Tuple[Dict, List, List]:
    
        self.cap_denormailization = min(self.c_range)
        self.res_denormailization = min(self.r_range) 

        c_range_normalized = [1, max(self.c_range)/min(self.c_range)]
        r_range_normalized = [1, max(self.r_range)/min(self.r_range)]

        parameters: Dict[str, ng.p.Log] = {}
        for var in self.sizing_assist.design_variables_dict:
            param = None
            if var.startswith("R"):
                parameters[str(var)] = ng.p.Log(lower=r_range_normalized[0], upper=r_range_normalized[1]) if log_scale else ng.p.Scalar(lower=r_range_normalized[0], upper=r_range_normalized[1]) 
            elif var.startswith("C"):
                parameters[str(var)] = ng.p.Log(lower=c_range_normalized[0], upper=c_range_normalized[1]) if log_scale else ng.p.Scalar(lower=c_range_normalized[0], upper=c_range_normalized[1]) 
            else:
                parameters[str(var)] = ng.p.Log(lower=self._default_var_bounds[0], upper=self._default_var_bounds[1]) if log_scale else ng.p.Scalar(lower=self._default_var_bounds[0], upper=self._default_var_bounds[1])  # Default bounds (fail gracefully)

        self.parametrization = ng.p.Dict(**parameters)
        
        return parameters, self.cap_denormailization, self.res_denormailization

    def denormalize_params(self, parameterization: Dict[str, float]) -> Dict[str, float]:

        for key in parameterization.keys():
            if "R_" in key:
                parameterization[key] = parameterization[key] * self.res_denormailization
            elif "C_" in key:
                parameterization[key] = parameterization[key] * self.cap_denormailization

        return parameterization

    def eval_symbolic_tf_fit(self, parameterization: Dict[str, float], epsilon: float = 1e-10) -> Tuple[Dict, torch.Tensor]:
        
        curr_tf_symbolic = self.sizing_assist.sub_val_design_vars(parameterization)

        current_complex_response = self.helper_functions.eval_tf(tf=curr_tf_symbolic, f_val=self.frequencies)
        target_complex_response  = self.helper_functions.eval_tf(tf=self.target_tf, f_val=self.frequencies)

        fit_summary = get_bode_fitness_loss(
            current_complex_response=current_complex_response, 
            target_complex_response=target_complex_response, 
            freq_weights=self.freq_weights, 
            loss_type=self.loss_type, 
            norm_method=self.loss_norm_method,
            rescale=self.rescale_mag)
        
        mag_loss   = fit_summary['mag_loss']
        phase_loss = fit_summary['phase_loss']

        # Add new data to the summary
        fit_summary["current_complex_response"] = current_complex_response
        fit_summary['target_complex_response']  = target_complex_response
        fit_summary["mag-phase-target"]    = self.helper_functions.get_mag_phase_from_complex_response(complex_response_array=target_complex_response, epsilon=epsilon)
        fit_summary["mag-phase-optimized"] = self.helper_functions.get_mag_phase_from_complex_response(complex_response_array=current_complex_response, epsilon=epsilon)
        fit_summary["frequencies"] = self.frequencies       

        return fit_summary, mag_loss, phase_loss
    
    def evaluate(self, parameterization: Dict[str, float], include_phase_loss: bool = True, include_mag_loss: bool = True, penality_mult: float = 1, epsilon: float = 1e-10) -> float:

        parameterization  = self.denormalize_params(parameterization)
        
        fit_summary, mag_loss, phase_loss = self.eval_symbolic_tf_fit(parameterization, epsilon=epsilon)
        # l1norm = torch.sum(torch.tensor([val for val in parameterization.values()]))

        loss = 0
        loss += mag_loss   if include_mag_loss else 0
        loss += phase_loss if include_phase_loss else 0
        loss  = torch.clip(loss, min=0, max=self.max_mse_loss)

        # Add penalty for violating mag
        loss += penality_mult * max(0, fit_summary['target_max_mag'] - fit_summary['curr_max_mag'])**2

        # Log the summary if an improvement happens

        return float(loss.detach())

    def create_experiment(self, budget: int, overwrite_optimizer:ng.optimization.base.Optimizer | None = None) -> bool:
        if self.parametrization is None:
            print("NEED TO CALL self.parameterize")
            return False

        elif overwrite_optimizer is not None:
            self.optimizer = overwrite_optimizer(parametrization=self.parametrization, budget=budget)
        else:
            self.optimizer = ng.optimizers.registry.get(self.optimizer_name)(parametrization=self.parametrization, budget=budget)
        print(f"Optimizer is set to {self.optimizer.name} with budget = {budget}")
        return True
    
    def optimize(self, include_mag_loss: bool = True, include_phase_loss: bool = True, epsilon: float = 1e-10, render_optimization_trace: bool = True, verbose_logging: bool = True) -> bool:

        if self.optimizer is None:
            return False
        
        
        # Track the loss for plotting
        loss_values = []
        trials = []

        self.optimizer_trace = []  # Store the optimization trace
        
        # Run the optimization process
        for trial in tqdm(range(self.optimizer.budget), desc="Optimizing", unit="trial"):
            candidate = self.optimizer.ask()  # Get a new candidate
            loss = self.evaluate(candidate.value, include_mag_loss=include_mag_loss, include_phase_loss=include_phase_loss, epsilon=epsilon)  # Evaluate function
            self.optimizer.tell(candidate, loss)  # Provide feedback to optimizer
            self.optimizer_trace.append((candidate, loss))  # Log the achieved loss
            
            # Store loss and step number for plotting
            loss_values.append(loss)
            trials.append(trial)

            if loss < self.optimizer_trace[self.global_min_index][1]:
                self.global_min_index = trial
        
        # Plot the loss as a function of optimization step
        self._plot_loss(trials, loss_values)

        return True
    
    def get_best(self) -> Dict[str, float]:

        if self.optimizer is None:
            print("Need to set the optimizer by calling self.create_experiment")
            return
        
        if len(self.optimizer_trace) < 1:
            print("need to run self.optimize")
            return
        
        best_solution, loss = self.optimizer_trace[self.global_min_index]
        best_parameters = best_solution.value

        print("Optimized x - normalized:", best_solution.value)
        print("Optimized x - de-normalized:", self.denormalize_params(best_solution.value))

        print("loss:", loss)

        return self.denormalize_params(best_solution.value), loss
    
    def _plot_loss(self, trials, loss_values):
        """Plot the loss as a function of optimization steps with Plotly."""
        fig = go.Figure()
        
        # Add a line plot with trials on the x-axis and loss_values on the y-axis
        fig.add_trace(go.Scatter(x=trials, y=loss_values, mode='markers+lines', name='Loss', line=dict(color='blue', width=2)))

        # Add title and labels
        fig.update_layout(
            title='Loss vs. Optimization Trial',
            xaxis_title='Optimization Step',
            yaxis_title='Loss',
            template='plotly_dark',  # Optional: Use dark theme for the plot
            showlegend=True
        )
        
        # Show the interactive plot
        fig.show()

    def plot_solution(self, prameterization: Dict[str, float]):

        fit_summary, mag_loss, phase_loss = self.eval_symbolic_tf_fit(prameterization)
        print(f"mag_loss {mag_loss}, phase_loss {phase_loss}, max-mag {fit_summary['curr_max_mag']}")

        target_complex_response  = fit_summary['target_complex_response']
        current_complex_response = fit_summary['current_complex_response']
        frequencies = fit_summary['frequencies']

        plot_complex_response(frequencies=frequencies, complex_response_list=[target_complex_response, current_complex_response], labels=['Target', 'Optimized'])