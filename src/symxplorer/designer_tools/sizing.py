import torch
import numpy  as np
import control as ctrl
import sympy  as sp
from   sympy  import Eq
from   typing import Dict, List, Tuple
from   copy   import deepcopy

import ax
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.notebook.plotting import init_notebook_plotting, render

# Plotting Tools
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from   symxplorer.symbolic_solver.solver  import ExperimentResult
from   symxplorer.symbolic_solver.domains import Filter_Classification
from   symxplorer.symbolic_solver.filter  import Filter_Classifier

from   .visualizer import Visualizer 
from   .utils      import weighted_mse_loss, plot_ac_response, Transfer_Func_Helper

s = sp.symbols("s")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype  = torch.double

torch.set_default_dtype(dtype)
torch.set_default_device(device)
print(f'Using device: {device} and dtype: {dtype}')

class Sizing_Assist:
    def __init__(self, filter_classification: Filter_Classification = None, tf: sp.Expr = None, metrics: Dict[str, sp.Expr] = {}):
        """Either input a filter classification or a transfer function. If a transfer function is input, the filter will be classified automatically against a BiQuad filter."""
        self.filter_classification: Filter_Classification = filter_classification
        self.tf: sp.Expr = deepcopy(filter_classification.transferFunc if filter_classification else tf)

        self.design_variables_dict: Dict[str, sp.Symbol] = {str(sym) : sym 
                                                       for sym in self.tf.free_symbols 
                                                       if str(sym) != 's'} if self.tf else {}
        
        params:            Dict[str, sp.Expr]           = {str(k) : v.simplify() for k, v in filter_classification.parameters.items() if v } if filter_classification else {}
        self.metrics_dict: Dict[str, sp.Expr]           = params if params else metrics if metrics else {}
        self.metrics_symbols_dict: Dict[str, sp.Symbol] = {k : sp.symbols(k) for k in params.keys()} if params else {}
        # To be computed after initialisation
        self.design_var_metrics_dict: Dict[sp.Symbol, sp.Expr] = {} # holds the design variables (not relaxed) in terms of the metrics
        self.relax_design_variables:  Dict[str, sp.Symbol]     = {} # holds the relaxed design variables
        self.equations:               List[sp.Expr]            = []

        self.classifier = Filter_Classifier()
        if filter_classification:
            self.classifier.overwrite_classifications([filter_classification])

        elif tf:
            self.classifier.transferFunctionsList = [tf]
            self.classifier.classifyFilter("BiQuad")

        if filter_classification:
            self.visualizer = Visualizer(filter_classification=filter_classification)
        elif tf:
            self.visualizer = Visualizer(tf=tf)
        else:
            self.visualizer = Visualizer()

    # Problem space
    def get_design_variables(self) -> Dict[str, sp.Symbol]:
        return self.design_variables_dict.keys()
    
    def reset_design_variable(self):
        self.design_variables_dict = deepcopy({str(sym) : sym 
                                                for sym in self.tf.free_symbols 
                                                if str(sym) != 's'} if self.tf else {})
        self.relax_design_variables = {}
        
    def relax_design_variable(self, name: str) -> bool:
        """Relaxes the problem dimension"""
        if self.design_variables_dict.get(name):
            self.relax_design_variables[name] = self.design_variables_dict.pop(name)
            return True
        else:
            print(f'Design variable {name} not found in {self.design_variables_dict.keys()}')
            return False
      
    def get_metrics(self) -> Dict[str, sp.Expr]:
        return self.metrics_dict.keys()
    
    def add_metric(self, metric_expression: sp.Expr, name: str):
        """Add a metric to the sizing problem."""
        self.metrics_dict[name] = metric_expression
        self.metrics_symbols_dict[name] = sp.symbols(name)

    def remove_metric(self, name: str) -> bool:
        """Remove an objective from the sizing problem."""
        if self.metrics_dict.get(name):
            self.metrics_dict.pop(name)
            self.metrics_symbols_dict.pop(name)
            return True
        else:
            print(f'{self.metrics_dict.get(name)}Metric {name} not found in {self.metrics_dict.keys()}')
            return False

    def kill_variable(self, design_variable_to_kill: str, design_variable_to_takeover: str) -> bool:
        """Eliminates a design variable in the objective functions reducing the design space complexity."""
        if self.design_variables_dict.get(design_variable_to_kill) and self.design_variables_dict.get(design_variable_to_takeover):
            sub_dict = {
                self.design_variables_dict[design_variable_to_kill]: self.design_variables_dict[design_variable_to_takeover]
                }
            self.tf = self.tf.subs(sub_dict)
            self.metrics_dict = {k: v.subs(sub_dict) for k, v in self.metrics_dict.items()} 
            self.design_variables_dict.pop(design_variable_to_kill)
            return True

        else:
            print (f'Design variable {design_variable_to_kill} (to kill) or {design_variable_to_takeover} (takeover) not found in {self.design_variables_dict.keys()}')
            return False
    
    def get_problem_space(self) -> Tuple[int, int, int]:
        """Returns the dimension of the problem space."""
        return len(self.relax_design_variables), len(self.design_variables_dict), len(self.metrics_dict)

    def is_overconstrained(self) -> bool:
        """Checks if the problem space is overconstrained. (only light check)"""
        return len(self.design_variables_dict) < len(self.metrics_dict)
    
    def get_jacobian_of_metrics(self) -> sp.Matrix:
        """Returns the Jacobian matrix of the metrics. Can be used before solving the system to check the problem dimensionality."""
        # Form the Jacobian matrix
        equations = [v.simplify() for k, v in self.metrics_dict.items()]
        variables = self.design_variables_dict.values()
        jacobian = sp.Matrix([[eq.diff(var) for var in variables] for eq in equations])

        # Perform row reduction on the Jacobian
        rref_matrix, pivot_columns = jacobian.rref()  # Row reduce the Jacobian
        dependent_rows = [i for i in range(jacobian.rows) if i not in pivot_columns]

        print("Jacobian Matrix:")
        print(jacobian)
        print("Row Reduced Jacobian Matrix:")
        print(rref_matrix)
        print("Dependent Rows (indices):", dependent_rows)

        if dependent_rows:
            print(f"The equations corresponding to rows {dependent_rows} are dependent.")

        return jacobian, rref_matrix, dependent_rows
    
    def is_metrics_independant(self) -> bool:
        """Checks if the metrics are independant of the design variables."""
        if self.design_variables_dict and self.metrics_dict and self.metrics_symbols_dict:
            # Form the Jacobian matrix
            equations = [v.simplify() for k, v in self.metrics_dict.items()]
            variables = self.design_variables_dict.values()
            jacobian = sp.Matrix([[eq.diff(var) for var in variables] for eq in equations])

            # Compute the rank of the Jacobian matrix
            rank = jacobian.rank()

            # Check if the rank equals the number of equations
            print("Jacobian Matrix:")
            print(jacobian)
            print("Rank:", rank)
            print("Number of equations:", len(equations))

            if rank == len(equations):
                print("The system is independent.")
            else:
                print("The system is dependent.")  
            return rank == len(equations)
        print("No metrics or design variables to check.")
        return False  
    
    # Solving tools
    def solve_inverse(self) -> Dict[str, sp.Expr]:
        """Computes the design variables in terms of the objectives."""
        self.equations = [Eq(self.metrics_symbols_dict[k], v.simplify()) for k, v in self.metrics_dict.items()]
        try:
            sol = sp.solve(self.equations, list(self.design_variables_dict.values()), dict=True)
        except KeyboardInterrupt:
            print("Computation interrupted.")
            print(f"Equations ({len(self.equations)}) to solve:")
            for i, eq in enumerate(self.equations):
                print(f"{i} - {eq}")
            return {}

        if len(sol) > 0:
            self.design_var_metrics_dict = sol[0]
            return {str(k): v for k, v in sol[0].items()}
        else:
            print("No solution found for the given metrics.")
            print("Review the metrics to ensure system is not overconstrained.")
            print(f"Returning empty dictionary: {sol}")
            return {}

    def sub_val_metrics(self, sub_dict: Dict[str, float]) -> Dict[str, sp.Basic]:
        """Numeroically finds the design variables given the provided dictionary of the metric values."""
        if self.design_var_metrics_dict:
            sub_dict_validated: Dict[sp.Symbol, float] = {}
            for k, v in sub_dict.items():
                if self.metrics_symbols_dict.get(k):
                    sub_dict_validated[self.metrics_symbols_dict[k]] = v
                else:
                    print(f"Metric {k} not found in {self.metrics_symbols_dict.keys()}")
                    print("Ignoring the metric.")
            if sub_dict_validated:
                return {k: v.subs(sub_dict_validated) for k, v in self.design_var_metrics_dict.items()}
            return {}
        else:
            print("No design variables to substitute.")            
            raise KeyError("No metrics to substitute.")

    def sub_val_design_vars_in_metrics(self, sub_dict: Dict[str, float]) -> Dict[str, sp.Basic]:
        """Numeroically finds the metrics given the provided dictionary of the design variable values."""
        if self.design_variables_dict:
            sub_dict_validated: Dict[sp.Symbol, float] = {}
            for k, v in sub_dict.items():
                if self.design_variables_dict.get(k):
                    sub_dict_validated[self.design_variables_dict[k]] = v
                else:
                    print(f"Design variable {k} not found in {self.design_variables_dict.keys()}")
                    print("Ignoring the variable.")
            if sub_dict_validated:
                return {k: v.subs(sub_dict_validated) for k, v in self.metrics_dict.items()}
            return {}
            
        else:
            raise KeyError("No design variables to substitute.")

    def sub_val_design_vars(self, sub_dict: Dict[str, float]) -> sp.Expr:
        """Substitues the given parameters by their floating point realization and returns a symbolic TF"""
        if self.design_variables_dict:
            sub_dict_validated: Dict[sp.Symbol, float] = {}
            for k, v in sub_dict.items():
                if self.design_variables_dict.get(k):
                    sub_dict_validated[self.design_variables_dict[k]] = v
                else:
                    print(f"Design variable {k} not found in {self.design_variables_dict.keys()}")
                    print("Ignoring the variable.")
            if sub_dict_validated:
                return self.tf.subs(sub_dict_validated)
            return None
            
        else:
            raise KeyError("No design variables to substitute.")

class Ax_Wrapper:
    def __init__(self, 
                 tf_to_size: sp.Expr,
                 target_tf: sp.Expr,
                 c_range: List[float] = [1e-12, 1e-9], 
                 r_range: List[float] = [1e2, 1e5],
                 frequencies: torch.Tensor = torch.logspace(3, 8, 1000),
                 freq_weights: torch.Tensor = None,
                 random_seed: int = 42,
                 ):
        
        init_notebook_plotting()

        self.sizing_assist = Sizing_Assist(tf=tf_to_size)
        self.target_tf = target_tf
        self.c_range = c_range
        self.r_range = r_range
        self.frequencies = frequencies
        self.freq_weights = freq_weights if freq_weights is not None else torch.ones_like(frequencies)
        self.ax_client = AxClient(random_seed=random_seed)
            

        self.parameters = None
        self.cap_denormailization = None
        self.res_denormailization = None
        self.helper_functions = Transfer_Func_Helper()

    def parameterize(self) -> Tuple[Dict, List, List]:
    
        self.cap_denormailization = min(self.c_range)
        self.res_denormailization = min(self.r_range) 

        c_range_normalized = [1, max(self.c_range)/min(self.c_range)]
        r_range_normalized = [1, max(self.r_range)/min(self.r_range)]

        parameters: List[Dict] = []
        for var in self.sizing_assist.design_variables_dict:
            bound = None # Reset range
            if "C_" in var:
                bound = c_range_normalized
            elif "R_" in var:
                bound = r_range_normalized
            parameters.append({
                "name": str(var),
                "type": "range",        # Type of parameter (range, choice, etc.)
                "bounds": bound,        
                "value_type": "float",  
                "log_scale": True       # Indicate that the parameter should be on a log scale
            })

        self.parameters = parameters
        
        return parameters, self.cap_denormailization, self.res_denormailization

    def denormalize_params(self, parameterization: Dict[str, float]) -> Dict[str, float]:

        for key in parameterization.keys():
            if "R_" in key:
                parameterization[key] = parameterization[key] * self.res_denormailization
            elif "C_" in key:
                parameterization[key] = parameterization[key] * self.cap_denormailization

        return parameterization

    def eval_symbolic_tf_fit(self, parameterization: Dict[str, float]) -> Tuple[Dict[str, Tuple[torch.Tensor,torch.Tensor]], torch.Tensor]:
        
        tf_symbolic = self.sizing_assist.sub_val_design_vars(parameterization)
        mag, phase  = self.helper_functions.get_ac_response_from_symbolic(tf_symbolic, self.frequencies)
        mag_target, phase_target = self.helper_functions.get_ac_response_from_symbolic(self.target_tf, self.frequencies)

        fit_summary = {
            "ideal":      (mag_target, phase_target),
            "experiment": (mag, phase),
            "frequencies": (self.frequencies)
        }

        mag_loss,   _ = weighted_mse_loss(mag, mag_target, self.freq_weights)
        phase_loss, _ = weighted_mse_loss(phase, phase_target, self.freq_weights)

        return fit_summary, mag_loss + phase_loss
    
    def evaluate(self, parameterization) -> Dict[str, Tuple[float, float]]:

        parameterization = self.denormalize_params(parameterization)
        fit_summary, loss = self.eval_symbolic_tf_fit(parameterization)

        l1norm = torch.sum(torch.tensor([val for val in parameterization.values()]))

        return {"tf_fitting_loss" : (loss.detach(), 0.0), "l1norm" : (l1norm, 0)}

    def create_experiment(self):
        if self.parameters is None:
            print("Need to parameterize the variables first")
            return
        self.ax_client.create_experiment(
            name="Transfer Function Optimization",  
            parameters=self.parameters,
            objectives={
                "tf_fitting_loss": ObjectiveProperties(minimize=True),
                # "l1norm" : ObjectiveProperties(minimize=True) # to minimize circuit component size
                },
            overwrite_existing_experiment=True,
            # parameter_constraints = ["C_1 + C_2 <= 2.0"],
            # outcome_constraints = ["l1norm <= 1.25"],
        )

    def optimization_loop(self, num_trials: int = 20):
        for _ in range(num_trials):
            parameterization, trial_index = self.ax_client.get_next_trial() 
            self.ax_client.complete_trial(trial_index=trial_index, raw_data=self.evaluate(parameterization)) # Tell Ax the outcome

    # 5. Get the best parameters and the bes1t loss value
    def get_best(self, render_trace: bool = True, denormalize: bool = True):
        best_parameters, values = self.ax_client.get_best_parameters()
        best_trial = self.ax_client.get_best_trial()

        print(f"Best parameters: {best_parameters}")
        print(f"Best loss: {best_trial}")

        if render_trace: 
            render(self.ax_client.get_optimization_trace(objective_optimum = 0))

        if denormalize:
            best_parameters = self.denormalize_params(best_parameters)

        return best_parameters, best_trial
    
    def render_contour_plot(self, param_x: str, param_y: str, metric: str = "tf_fitting_loss"):
        render(self.ax_client.get_contour_plot(param_x=param_x, param_y=param_y, metric_name=metric))

    def get_trials_as_df(self):
        return self.ax_client.generation_strategy.trials_as_df
    
    def plot_solution(self, prameterization = None):
        if prameterization is None:
            prameterization, values = self.ax_client.get_best_parameters()

        fit_summary, loss = self.eval_symbolic_tf_fit(prameterization)

        mag_target, phase_target   = fit_summary["ideal"]
        mag, phase = fit_summary["experiment"]
        frequencies                = fit_summary["frequencies"]
        plot_ac_response(frequencies, [mag, mag_target], [phase, phase_target], ["Optimized", "Target"], "Frequency Response")

