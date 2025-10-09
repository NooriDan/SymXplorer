"""This Module includes base """
import logging
import json
import torch
import numpy        as np
import sympy        as sp
import nevergrad    as ng
import plotly.graph_objects as go

from    typing      import Dict, List, Tuple, Any, Mapping
from    tqdm        import tqdm
from    abc         import ABC, abstractmethod
from    pathlib     import Path
from    spicelib    import RawRead
from    dacite      import from_dict, Config
from    dataclasses import asdict
from    datetime    import datetime


# Symxplorer Specific Imports
from   symxplorer.spice_engine.spicelib     import Spicelib_Wrapper
from   symxplorer.designer_tools.domains    import Project_Setup, ListTargetSpec, TargetSpec
from   symxplorer.designer_tools.domains    import OptimizationGoalType, OptimizationPoint, OptimizationLogEntry, Error_Types, OptimizationLog
from   symxplorer.designer_tools.utils      import compute_error, compute_reward, convert_linear_to_log, log_denormalize, linear_denormalize
from   symxplorer.designer_tools.utils      import plot_complex_response, get_bode_fitness_loss, Transfer_Func_Helper, Frequency_Weight, UNIT_DICT

from   symxplorer.optimization.base         import Spice_Constraint_Satisfaction, Spice_Single_Objective, Spice_Bode_Optimizer, Base_Optimizer
from   symxplorer.optimization.base         import MAX_PENALTY, MAX_REWARD

logger = logging.getLogger("SymXplorer.Nevergrad")

s = sp.symbols("s")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype  = torch.double

torch.set_default_dtype(dtype)
torch.set_default_device(device)
logger.info(f'Using device: {device} and dtype: {dtype}')


# ----------------------------
# --- Global Constants ---
# ----------------------------


# ----------------------------
# --- Class Definitions ---
# ----------------------------

# ------------------------------------------------
# A [ABSTRACT] Nevergrad-based Optimizers
# ------------------------------------------------
class NevergradMixin(Base_Optimizer):
    """Reusable mixin for all Nevergrad-based optimizers."""
    # --- Overwriting Some Abstract Methods ---
    def parameterize(self) -> ng.p.Dict:        
        parameters: Dict[str, ng.p.Scalar] = {}
        for param in self.setup_obj.dut_params:
            if param.log_scale:
                parameters[param.name] = ng.p.Log(
                    lower=self.optimizer_config.log_variable_bounds.min, 
                    upper=self.optimizer_config.log_variable_bounds.max)
            else:
                parameters[param.name] = ng.p.Scalar(
                    lower=self.optimizer_config.lin_variable_bounds.min, 
                    upper=self.optimizer_config.lin_variable_bounds.max)
                
        self.parametrization = ng.p.Dict(**parameters)
        return self.parametrization
    
    def _create_optimizer_obj(self) -> bool:
        if self.parametrization is None:
            logger.critical("NEED TO CALL self.parameterize")
            return False
        
        if self.optimizer_config.random_seed is not None:
            self.parametrization.random_state = np.random.RandomState(self.optimizer_config.random_seed)

        registry = ng.optimizers.registry.get(self.optimizer_config.name)
        if registry is not None:
            self.optimizer = registry(parametrization=self.parametrization, budget=self.optimizer_config.budget)
            logger.info(f"Optimizer is set to {self.optimizer.name} with budget = {self.optimizer_config.budget}")
            return True
        return False

    def optimization_step(self) -> Tuple[Dict[str, np.floating] , np.floating , Dict[str, Any]]:
        # Get a new candidate
        candidate : ng.p.Parameter = self.optimizer.ask()
        # Evaluate function
        denorm_params: Dict[str, float] = self.denormalize_params(parameterization=candidate.value)
        curr_score, metadata = self.evaluate(parameterization=denorm_params)
        # Provide feedback to optimizer (The negative of the fitness score is used because the optimizer is set to minimize this value... this way the optimizer will maximize the fitness score.
        self.optimizer.tell(candidate, -1 * curr_score)
        return candidate.value, curr_score, metadata
    

# ------------------------------------------------
# B [CONCRETE] Nevergrad-based Bode Fitter
# ------------------------------------------------
class Nevergrad_Spice_Bode_Optimizer(NevergradMixin, Spice_Bode_Optimizer):
    pass

# ------------------------------------------------
# B [CONCRETE] Nevergrad-based Constraint Satisfaction
# ------------------------------------------------
class Nevergrad_Spice_Constraint_Satisfaction(NevergradMixin, Spice_Constraint_Satisfaction):
    def __init__(self,
                 setup_obj: Project_Setup,
                 spicelib_wrapper : Spicelib_Wrapper):
        super().__init__(setup_obj = setup_obj, spicelib_wrapper = spicelib_wrapper)
        self.parametrization: ng.p.Dict | None = None
    
# ------------------------------------------------
# B [CONCRETE] Nevergrad-based Single Objective Optimizer
# ------------------------------------------------
class Nevergrad_Spice_Single_Objective(NevergradMixin, Spice_Single_Objective):
    def __init__(self,
                 setup_obj: Project_Setup,
                 spicelib_wrapper : Spicelib_Wrapper):
        super().__init__(setup_obj = setup_obj, spicelib_wrapper = spicelib_wrapper)
        self.parametrization: ng.p.Dict | None = None
