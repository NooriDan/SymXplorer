import torch
import itertools
import numpy  as np
import sympy  as sp
from   sympy  import Eq
from   typing import Dict, List, Tuple
from   copy   import deepcopy
from   tqdm   import tqdm

# Symxplorer Specific Imports
from   symxplorer.symbolic_exploration.domains import Filter_Classification
from   symxplorer.symbolic_exploration.filter  import Filter_Classifier

from   .visualizer import Symbolic_Visualizer, DesignSpaceExploration, ExplorationPoint

s = sp.symbols("s")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype  = torch.double

torch.set_default_dtype(dtype)
torch.set_default_device(device)
# print(f'Using device: {device} and dtype: {dtype}')


class Symbolic_Sizing_Assist:
    def __init__(self, filter_classification: Filter_Classification | None = None, tf: sp.Expr | None = None, metrics: Dict[str, sp.Expr] = {}):
        """Either input a filter classification or a transfer function. If a transfer function is input, the filter will be classified automatically against a BiQuad filter.
        metrics is writtent ot the metrics_dict only when tf is specified."""
        # choose tf: prefer the filter_classification.transferFunc when provided
        self.tf: sp.Expr = deepcopy(filter_classification.transferFunc if filter_classification else tf)

        # design variables: all free symbols except 's'
        self.design_variables_dict: Dict[str, sp.Symbol] = {str(sym) : sym 
                                                       for sym in self.tf.free_symbols 
                                                       if str(sym) != 's'} if self.tf else {}
        
        # get params from filter_classification if present; else use provided metrics dict
        params:            Dict[str, sp.Expr]           = {str(k) : v.simplify() for k, v in filter_classification.parameters.items() if v } if filter_classification else {}

        # select metrics source: params (from classification) takes precedence, else metrics arg
        self.metrics_dict: Dict[str, sp.Expr]           = params if params else metrics if metrics else {}
        # Ensure metrics_symbols_dict exists for whatever metrics we have
        self.metrics_symbols_dict: Dict[str, sp.Symbol] = {k : sp.symbols(k) for k in params.keys()} if params else {}
        
        # To be computed after initialisation
        self.design_var_in_terms_of_metrics_dict: Dict[sp.Symbol, sp.Expr] = {} # holds the design variables (not relaxed) in terms of the metrics
        self.relaxed_design_variables:  Dict[str, sp.Symbol]     = {} # holds the relaxed design variables
        self.metric_equations:          List[sp.Expr]            = []

        self.classifier = Filter_Classifier()
        if filter_classification: self.classifier.overwrite_classifications([filter_classification])
        elif tf:
            self.classifier.transferFunctionsList = [tf]
            self.classifier.classifyFilter("BiQuad")

        if filter_classification:       self.visualizer = Symbolic_Visualizer(filter_classification=filter_classification)
        elif tf:                        self.visualizer = Symbolic_Visualizer(tf=tf)
        else:                           self.visualizer = Symbolic_Visualizer()

    # Problem space
    def get_design_variables(self) -> List[str]:
        return [k for k in self.design_variables_dict.keys()]
    
    def reset_design_variables(self):
        self.design_variables_dict = deepcopy({str(sym) : sym 
                                                for sym in self.tf.free_symbols 
                                                if str(sym) != 's'} if self.tf else {})
        self.relaxed_design_variables = {}
        
    def relax_design_variable(self, name: str) -> bool:
        """Relaxes the problem dimension"""
        if self.design_variables_dict.get(name):
            self.relaxed_design_variables[name] = self.design_variables_dict.pop(name)
            return True
        else:
            print(f'Design variable {name} not found in {self.design_variables_dict.keys()}')
            return False
        
    def get_problem_space(self) -> Tuple[int, int, int]:
        """Returns the dimension of the problem space. <relax-var-dim , design-var-dim, metrics-dim>"""
        return len(self.relaxed_design_variables), len(self.design_variables_dict), len(self.metrics_dict)    
      
    def get_metrics(self) -> List[str]:
        return [k for k in self.metrics_dict.keys()]
    
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
        """Eliminates a design variable in the objective functions by setting it equal to another variable, thereby reducing the design space complexity."""
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
    
    def is_overconstrained(self) -> bool:
        """Checks if the problem space is overconstrained. (only light check)"""
        return len(self.design_variables_dict) < len(self.metrics_dict)
    
    def get_jacobian_of_metrics(self) -> Tuple[sp.Matrix, sp.Matrix, List[int]]:
        """Returns (jacobian, rref_matrix, dependent_rows).
        dependent_rows is a list of indices of rows in the original jacobian that are (likely) dependent.
        """
        equations = [v.simplify() for k, v in self.metrics_dict.items()]
        variables = list(self.design_variables_dict.values())
        jacobian = sp.Matrix([[eq.diff(var) for var in variables] for eq in equations])

        # rref returns (matrix, pivot_columns)
        rref_matrix, pivot_columns = jacobian.rref()
        # pivot_columns is a tuple of pivot column indices; Jacobian rows that become zero rows
        # in rref correspond to dependent equations. Let's detect zero rows in rref:
        dependent_rows = [i for i in range(rref_matrix.rows) if all(rref_matrix[i, j] == 0 for j in range(rref_matrix.cols))]

        print("Jacobian Matrix:")
        print(jacobian)
        print("Row Reduced Jacobian Matrix (RREF):")
        print(rref_matrix)
        print("Pivot columns (from rref):", pivot_columns)
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
        self.metric_equations = [Eq(self.metrics_symbols_dict[k], v.simplify()) for k, v in self.metrics_dict.items()]
        try:
            sol = sp.solve(self.metric_equations, list(self.design_variables_dict.values()), dict=True)
        except KeyboardInterrupt:
            print("Computation interrupted.")
            print(f"Equations ({len(self.metric_equations)}) to solve:")
            for i, eq in enumerate(self.metric_equations):
                print(f"{i} - {eq}")
            return {}

        if len(sol) > 0:
            self.design_var_in_terms_of_metrics_dict = sol[0]
            return {str(k): v for k, v in sol[0].items()}
        else:
            print("No solution found for the given metrics.")
            print("Review the metrics to ensure system is not overconstrained.")
            print(f"Returning empty dictionary: {sol}")
            return {}

    def sub_val_metrics(self, sub_dict: Dict[str, float]) -> Dict[str, sp.Basic]:
        """Numeroically finds the design variables given the provided dictionary of the metric values."""
        if self.design_var_in_terms_of_metrics_dict:
            sub_dict_validated: Dict[sp.Symbol, float] = {}
            for k, v in sub_dict.items():
                if self.metrics_symbols_dict.get(k):
                    sub_dict_validated[self.metrics_symbols_dict[k]] = v
                else:
                    print(f"Metric {k} not found in {self.metrics_symbols_dict.keys()}")
                    print("Ignoring the metric.")
            if sub_dict_validated:
                self.design_var_in_terms_of_metrics_dict =  {k: v.subs(sub_dict_validated) for k, v in self.design_var_in_terms_of_metrics_dict.items()}
                return self.design_var_in_terms_of_metrics_dict
            return {}
        else:
            print("No design variables to substitute.")            
            raise KeyError("No metrics to substitute.")

    def sub_val_design_vars_in_metrics(self, sub_dict: Dict[str, float]) -> Dict[str, sp.Basic]:
        """Numerically finds the metrics given the provided dictionary of the design variable values."""
        if self.design_variables_dict:
            sub_dict_validated: Dict[sp.Symbol, float] = {}
            for k, v in sub_dict.items():
                if self.design_variables_dict.get(k):       sub_dict_validated[self.design_variables_dict[k]] = v
                elif self.relaxed_design_variables.get(k):  sub_dict_validated[self.relaxed_design_variables[k]] = v
                else:
                    print(f"Design variable {k} not found in {self.design_variables_dict.keys()} or {self.relaxed_design_variables.keys()}")
                    print("Ignoring the variable.")
            self.design_var_in_terms_of_metrics_dict = {k: v.subs(sub_dict_validated) for k, v in self.design_var_in_terms_of_metrics_dict.items()}
            return self.design_var_in_terms_of_metrics_dict            
        else: raise KeyError("No design variables to substitute.")

    def sub_val_design_vars_in_tf(self, sub_dict: Dict[str, float]) -> sp.Expr:
        """Substitues the given parameters by their floating point realization and returns a symbolic TF"""
        if self.design_variables_dict:
            sub_dict_validated: Dict[sp.Symbol, float] = {}
            for k, v in sub_dict.items():
                if self.design_variables_dict.get(k): sub_dict_validated[self.design_variables_dict[k]] = v
                elif self.relaxed_design_variables.get(k): sub_dict_validated[self.relaxed_design_variables[k]] = v
                else:
                    print(f"Design variable {k} not found in {self.design_variables_dict.keys()} or {self.relaxed_design_variables.keys()}")
                    print("Ignoring the variable.")
            return self.tf.subs(sub_dict_validated)            
        else: raise KeyError("No design variables to substitute.")


    def explore_systematically(
        self,
        design_var_limits: Dict[str, Tuple[float, float]],
        num_points_per_var: int = 5,
        fixed_vars: Dict[str, float] | None = None,
        show_progress: bool = True,
        use_log_spacing: bool = False
    ) -> DesignSpaceExploration:
        """
        Perform a systematic (grid) exploration of the design space.

        Args:
            design_var_limits: Dict of {var_name: (min_val, max_val)} defining sweep limits.
            num_points_per_var: Number of sweep points per variable (default = 5).
            fixed_vars: Dict of {var_name: value} for variables to keep constant.
            show_progress: Whether to show a tqdm progress bar.

        Returns:
            DesignSpaceExploration: object containing all exploration points.
        """
        exploration_points: List[ExplorationPoint] = []
        metric_dict: Dict[str, sp.Expr] = self.metrics_dict
        fixed_vars = fixed_vars or {}

        # 1️⃣ Generate sweep grid
        if not use_log_spacing:
            grid_values = {
                var: np.linspace(vmin, vmax, num_points_per_var)
                for var, (vmin, vmax) in design_var_limits.items()
            }
        else:
            grid_values = {
                var: np.logspace(np.log10(vmin), np.log10(vmax), num_points_per_var)
                for var, (vmin, vmax) in design_var_limits.items()
            }

        design_var_cross_product = [
            dict(zip(grid_values.keys(), values))
            for values in itertools.product(*grid_values.values())
        ]

        # 2️⃣ Add fixed variables to every design point
        for d in design_var_cross_product:
            d.update(fixed_vars)

        # 3️⃣ Evaluate metrics
        iterator = tqdm(
            design_var_cross_product,
            desc="Exploring design space",
            total=len(design_var_cross_product),
            disable=not show_progress,
            ncols=80
        )

        for sub_dict in iterator:
            performance_metric_dict = {}

            # set circuit parameters in your evaluator
            self.visualizer.set_params(sub_dict)

            # evaluate metrics
            for param_name in self.get_metrics():
                _, metric_val = self.visualizer.eval_filter_parameter(param_name)
                performance_metric_dict[param_name] = float(metric_val)

            exploration_points.append(
                ExplorationPoint(
                    design_var_dict=sub_dict,
                    performance_metric_dict=performance_metric_dict,
                )
            )

        return DesignSpaceExploration(points=exploration_points)
