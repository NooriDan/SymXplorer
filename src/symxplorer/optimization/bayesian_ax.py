"""This Module implements the ax-based (Bayesian Optimization) optimizers """
import logging
import numpy        as np

from    typing      import Dict, Tuple, Any, List

# Ax Imports
from ax.api.client  import Client
from ax.api.configs import RangeParameterConfig
from ax.api.types   import TParameterization
from ax.api.protocols.metric import IMetric
# Symxplorer Specific Imports
from   symxplorer.spice_engine.spicelib     import Spicelib_Wrapper
from   symxplorer.designer_tools.domains    import Project_Setup
from   symxplorer.optimization.base         import Spice_Constraint_Satisfaction, Spice_Single_Objective, Spice_Bode_Optimizer, Base_Optimizer


logger = logging.getLogger("SymXplorer.Ax")
logger.debug(f'imported {__name__}')


# ----------------------------
# --- Global Constants ---
# ----------------------------
SCORE_METRIC_NAME = "score"

# ----------------------------
# --- Class Definitions ---
# ----------------------------

# ------------------------------------------------
# A [ABSTRACT] Ax-client-based Optimizers
# ------------------------------------------------
class Ax_Client_Mixin(Base_Optimizer):
    """Reusable mixin for all Ax-based optimizers."""
    # --- Overwriting Some Abstract Methods ---
    def parameterize(self) -> List[RangeParameterConfig]:        
        ax_parameters: List[RangeParameterConfig] = []
        for param in self.setup_obj.dut_params:
            ax_parameters.append(RangeParameterConfig(
                name=param.name,
                parameter_type="float",
                bounds= self.optimizer_config.get_lin_min_max()
            ))
        self.parametrization : List[RangeParameterConfig] = ax_parameters
        return ax_parameters
    
    def _create_optimizer_obj(self) -> bool:
        if self.parametrization is None:
            logger.critical("NEED TO CALL self.parameterize")
            return False
        # (1) Ax - create the client
        client = Client(random_seed=self.optimizer_config.random_seed, storage_config=None)
        # (2) Ax - add the parameterization
        client.configure_experiment(parameters=self.parametrization, name="SymXplorer-Experiment")
        # (3) Ax - Configure the objective
        client.configure_optimization(objective=SCORE_METRIC_NAME) # maxmimize the "score"
        # (4) Ax - Add tracking metrics
        _tracking_metrics = self.get_tracking_metrics_from_config()
        client.configure_metrics(metrics=_tracking_metrics)
        # Set the optimizer object
        self.optimizer : Client = client
        return True

    def optimization_step(self) -> Tuple[Dict[str, np.floating] , np.floating , Dict[str, Any]]:
        # Get a new candidate
        trials = self.optimizer.get_next_trials(max_trials=1)

        parameters : TParameterization = {}
        curr_score : Any = None
        metadata   : Dict[str, Any] = {}

        for trial_index, parameters in trials.items():
            # Evaluate function
            denorm_params = self.denormalize_params(parameterization=parameters)
            curr_score, metadata = self.evaluate(parameterization=denorm_params)
            # Provide feedback to optimizer: Complete the trial with the result
            raw_data = {SCORE_METRIC_NAME : curr_score}
            raw_data = self.extract_tracking_metrics_from_metadata(metadata=metadata, save_in_dict=raw_data)
            self.optimizer.complete_trial(trial_index=trial_index, raw_data=raw_data)

        return parameters, curr_score, metadata

    # Helper methods
    def get_tracking_metrics_from_config(self) -> List[IMetric]:
        list_of_metrics: List[IMetric] = []
        for spec_name in self.optimizer_config.target_specs.list_target_names():
            list_of_metrics.append(IMetric(spec_name))
        return list_of_metrics
    
    def extract_tracking_metrics_from_metadata(self, metadata, save_in_dict: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug("Extracing the tracking metrics from the metadata.")
        for metric_name, content in metadata.items():
            if metric_name in self.optimizer_config.target_specs.list_target_names() and np.isfinite(content['curr_val']): # The AX client complains even if a tracking metric is set to NaN
                save_in_dict[metric_name] = content['curr_val']
                logger.debug(f"\tadded {metric_name} = {content['curr_val']} to tracking metric dict - curr size {len(save_in_dict)}")
            else: logger.debug(f"skipping {metric_name}")
        logger.debug("Completed extracting the tracking metrics.")
        return save_in_dict

# ------------------------------------------------
# B [ABSTRACT] Optimizers with with Ax-client + custom-BoTorch Model 
# ------------------------------------------------
class Ax_Custom_BoTorch_Mixin(Ax_Client_Mixin):
    """TODO Reusable mixin for ax-based optimizers that use custom GenerationStrategy with BoTorch models."""
    # --- Overwriting Some Abstract Methods ---
    def _create_optimizer_obj(self) -> bool:
        pass

# ------------------------------------------------
# B [USER-ENDPOINT] Ax-based Bode Fitter
# ------------------------------------------------
class Nevergrad_Spice_Bode_Optimizer(Ax_Client_Mixin, Spice_Bode_Optimizer):
    pass

# ------------------------------------------------
# B [USER-ENDPOINT] Ax-based Constraint Satisfaction
# ------------------------------------------------
class Ax_Spice_Constraint_Satisfaction(Ax_Client_Mixin, Spice_Constraint_Satisfaction):
    def __init__(self,
                 setup_obj: Project_Setup,
                 spicelib_wrapper : Spicelib_Wrapper):
        super().__init__(setup_obj = setup_obj, spicelib_wrapper = spicelib_wrapper)
        logger.info(f"started the {__class__} optimizer class")
# ------------------------------------------------
# B [USER-ENDPOINT] Ax-based Single Objective Optimizer
# ------------------------------------------------
class Ax_Spice_Single_Objective(Ax_Client_Mixin, Spice_Single_Objective):
    def __init__(self,
                 setup_obj: Project_Setup,
                 spicelib_wrapper : Spicelib_Wrapper):
        super().__init__(setup_obj = setup_obj, spicelib_wrapper = spicelib_wrapper)
        logger.info(f"started the {__class__} optimizer class")