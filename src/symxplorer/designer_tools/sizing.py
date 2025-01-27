import sympy  as sp
from   sympy  import Eq
from   typing import Dict, List, Tuple
from   copy   import deepcopy

from   symxplorer.symbolic_solver.solver  import ExperimentResult
from   symxplorer.symbolic_solver.domains import Filter_Classification
from   symxplorer.symbolic_solver.filter  import Filter_Classifier

from   .visualizer import Visualizer 


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

    def remove_metric(self, name: str):
        """Remove an objective from the sizing problem."""
        if self.metrics_dict.get(name):
            self.metrics_dict.pop(name)
            self.metrics_symbols_dict.pop(name)
        else:
            raise ValueError(f'{self.metrics_dict.get(name)}Metric {name} not found in {self.metrics_dict.keys()}')

    def kill_variable(self, design_variable_to_kill: str, design_variable_to_takeover: str):
        """Eliminates a design variable in the objective functions reducing the design space complexity."""
        if self.design_variables_dict.get(design_variable_to_kill) and self.design_variables_dict.get(design_variable_to_takeover):
            sub_dict = {
                self.design_variables_dict[design_variable_to_kill]: self.design_variables_dict[design_variable_to_takeover]
                }
            self.tf = self.tf.subs(sub_dict)
            self.metrics_dict = {k: v.subs(sub_dict) for k, v in self.metrics_dict.items()} 
            self.design_variables_dict.pop(design_variable_to_kill)

        else:
            raise ValueError(f'Design variable {design_variable_to_kill} (to kill) or {design_variable_to_takeover} (takeover) not found in {self.design_variables_dict.keys()}')
    
    def get_problem_space(self) -> Tuple[int, int, int]:
        """Returns the dimension of the problem space."""
        return len(self.relax_design_variables), len(self.design_variables_dict), len(self.metrics_dict)
    
    def is_overconstrained(self) -> bool:
        """Checks if the problem space is overconstrained. (only light check)"""
        return len(self.design_variables_dict) < len(self.metrics_dict)
    
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

    def sub_val_design_vars(self, sub_dict: Dict[str, float]) -> Dict[str, sp.Basic]:
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



class Ax_Wrapper:
    def __init__(self):
        pass
