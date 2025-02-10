import logging
import os
import torch
import sympy  as sp
from   typing import Dict, List, Tuple, Callable
from   tqdm   import tqdm
from   datetime import datetime


# Ax Imports
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry    import Models
from ax.service.ax_client       import AxClient, ObjectiveProperties
from ax.utils.notebook.plotting import init_notebook_plotting, render

# Symxplorer Specific Imports
from   symxplorer.spice_engine.spicelib   import LTspice_Wrapper

from   .symbolic_sizing import Symbolic_Sizing_Assist
from   .utils           import weighted_mse_loss, plot_ac_response, plot_complex_response, get_bode_fitness_loss, Transfer_Func_Helper, Frequency_Weight, UNIT_DICT


s = sp.symbols("s")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype  = torch.double

torch.set_default_dtype(dtype)
torch.set_default_device(device)
print(f'Using device: {device} and dtype: {dtype}')

class Ax_Symbolic_Bode_Fitter:
    def __init__(self, 
                 tf_to_size: sp.Expr,
                 target_tf: sp.Expr,
                 c_range: List[float] = [1e-12, 1e-9], 
                 r_range: List[float] = [1e2, 1e5],
                 frequencies: torch.Tensor = torch.logspace(3, 8, 1000),
                 freq_weights: torch.Tensor = None,
                 max_mse_loss: float = 20,
                 mse_norm_method: str = "min-max",
                 random_seed: int = 42,
                 verbose_logging: bool = True
                 ):
        
        init_notebook_plotting()

        self.sizing_assist = Symbolic_Sizing_Assist(tf=tf_to_size)
        self.target_tf = target_tf
        self.c_range = c_range
        self.r_range = r_range
        self.frequencies = frequencies
        self.freq_weights = freq_weights if freq_weights is not None else torch.ones_like(frequencies)
        self.max_mse_loss = max_mse_loss
        self.mse_norm_method  = mse_norm_method
        self.random_seed  = random_seed
        self.verbose_logging = verbose_logging
        
        self.ax_client  = None
        self.gs         = None # Generation Strategy
        self.parameters = None
        self.cap_denormailization = None
        self.res_denormailization = None
        self.helper_functions = Transfer_Func_Helper()

    def parameterize(self, log_scale: bool = True) -> Tuple[Dict, List, List]:
    
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
                "log_scale": log_scale       # Indicate that the parameter should be on a log scale
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

    def eval_symbolic_tf_fit(self, parameterization: Dict[str, float], epsilon: float = 1e-10) -> Tuple[Dict[str, Tuple[torch.Tensor,torch.Tensor]], torch.Tensor]:
        
        tf_symbolic = self.sizing_assist.sub_val_design_vars(parameterization)
        mag, phase  = self.helper_functions.get_ac_response_from_symbolic(tf_symbolic, self.frequencies,  epsilon=epsilon)
        mag_target, phase_target = self.helper_functions.get_ac_response_from_symbolic(self.target_tf, self.frequencies,  epsilon=epsilon)

        fit_summary = {
            "ideal":      (mag_target, phase_target),
            "experiment": (mag, phase),
            "frequencies": (self.frequencies)
        }

        mag_loss,   _ = weighted_mse_loss(mag,   mag_target,   self.freq_weights, normalize_method=self.mse_norm_method)
        phase_loss, _ = weighted_mse_loss(phase, phase_target, self.freq_weights, normalize_method=self.mse_norm_method)

        return fit_summary, mag_loss, phase_loss
    
    def evaluate(self, parameterization, include_phase_loss: bool = True, include_mag_loss: bool = True, epsilon: float = 1e-10) -> Dict[str, Tuple[float, float]]:

        parameterization  = self.denormalize_params(parameterization)
        fit_summary, mag_loss, phase_loss = self.eval_symbolic_tf_fit(parameterization, epsilon=epsilon)
        l1norm = torch.sum(torch.tensor([val for val in parameterization.values()]))

        loss = 0
        loss += mag_loss if include_mag_loss else 0
        loss += phase_loss if include_phase_loss else 0

        loss = torch.clip(loss, min=0, max=self.max_mse_loss)
        return {"tf_fitting_loss" : (loss.detach(), 0.0), "l1norm" : (l1norm, 0)}

    def create_experiment(self, num_sobol_trials: int = 5, max_parallelism_sobol: int = 5, max_parallelism_bo: int = 3) -> None:

        self.gs = GenerationStrategy(
                        steps=[
                            # 1. Initialization step (does not require pre-existing data and is well-suited for
                            # initial sampling of the search space)
                            GenerationStep(
                                model_name="Sobol Generator",
                                model=Models.SOBOL,
                                num_trials=num_sobol_trials,                # How many trials should be produced from this generation step
                                min_trials_observed= num_sobol_trials-2,    # How many trials need to be completed to move to next model
                                max_parallelism=max_parallelism_sobol,      # Max parallelism for this step
                                model_kwargs={"seed": self.random_seed},    # Any kwargs you want passed into the model
                                model_gen_kwargs={"torch_device": device},  # Any kwargs you want passed to `modelbridge.gen`
                            ),
                            # 2. Bayesian optimization step (requires data obtained from previous phase and learns
                            # from all data available at the time of each new candidate generation call)
                            GenerationStep(
                                model=Models.BOTORCH_MODULAR,
                                # model_kwargs = {"botorch_model_class" : SaasFullyBayesianSingleTaskGP},
                                num_trials=-1,                       # No limitation (-1) on how many trials should be produced from this step
                                max_parallelism=max_parallelism_bo,  # Parallelism limit for this step, often lower than for Sobol
                                # More on parallelism vs. required samples in BayesOpt:
                                # https://ax.dev/docs/bayesopt.html#tradeoff-between-parallelism-and-total-number-of-trials
                                model_gen_kwargs={"torch_device": device},  # Any kwargs you want passed to `modelbridge.gen`
                            ),

                        ]
                    )

        self.ax_client = AxClient(
            generation_strategy=self.gs,
            random_seed=self.random_seed,
            verbose_logging=self.verbose_logging)

        if self.parameters is None:
            print("Need to parameterize the variables first")
            return
        
        self.ax_client.create_experiment(
            name="Transfer Function Optimization",  
            parameters=self.parameters,
            objectives={
                "tf_fitting_loss": ObjectiveProperties(minimize=True, threshold=0),
                },
            overwrite_existing_experiment=True,
            # parameter_constraints = ["C_1 + C_2 <= 2.0"],
            # outcome_constraints = ["tf_fitting_loss <= 10"],
            tracking_metric_names= ["l1norm"]
        )

    def optimization_loop(self, num_trials: int = 20, include_mag_loss: bool = True, include_phase_loss: bool = True, epsilon: float = 1e-10, verbose_logging: bool = True):
        # Save current logging level
        previous_logging_level = logging.getLogger().level
        
        if not verbose_logging:
            logging.getLogger().setLevel(logging.CRITICAL)
        
        try:
            for _ in tqdm(range(num_trials), desc="Optimizing", unit="trial"):
                parameterization, trial_index = self.ax_client.get_next_trial()
                self.ax_client.complete_trial(trial_index=trial_index, raw_data=self.evaluate(parameterization, include_phase_loss=include_phase_loss, include_mag_loss=include_mag_loss, epsilon=epsilon))  # Tell Ax the outcome
        finally:
            # Restore previous logging level
            logging.getLogger().setLevel(previous_logging_level)

    # 5. Get the best parameters and the bes1t loss value
    def get_best(self, render_trace: bool = True, denormalize: bool = True, use_model_predictions: bool = False):
        best_parameters, values = self.ax_client.get_best_parameters(use_model_predictions=use_model_predictions)
        # best_trial = self.ax_client.get_best_trial()


        if render_trace: 
            render(self.ax_client.get_optimization_trace(objective_optimum = 0))

        if denormalize:
            best_parameters = self.denormalize_params(best_parameters)

        return best_parameters, values[0]
    
    def render_contour_plot(self, param_x: str, param_y: str, metric: str = "tf_fitting_loss"):
        render(self.ax_client.get_contour_plot(param_x=param_x, param_y=param_y, metric_name=metric))

    def get_trials_as_df(self):
        return self.ax_client.generation_strategy.trials_as_df
    
    def plot_solution(self, prameterization = None):
        if prameterization is None:
            prameterization, values = self.ax_client.get_best_parameters()
            prameterization = self.denormalize_params(parameterization=prameterization)

        fit_summary, mag_loss, phase_loss = self.eval_symbolic_tf_fit(prameterization)
        print(f"mag_loss {mag_loss}, phase_loss {phase_loss}")
        mag_target, phase_target   = fit_summary["ideal"]
        mag, phase                 = fit_summary["experiment"]
        frequencies                = fit_summary["frequencies"]
        plot_ac_response(frequencies, [mag, mag_target], [phase, phase_target], ["Optimized", "Target"], "Frequency Response")
    
    # Utility
    def save_ax(self, name: str) -> str:
        # Generate a timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Create a file path with the timestamp
        dir = "./checkpoints"
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

        save_path = f"{dir}/{name}_{timestamp}.json"

        # Save the AxClient data to the JSON file with the timestamped path
        self.ax_client.save_to_json_file(save_path)
        return save_path
    
    @classmethod
    def load_ax(save_path: str) -> AxClient:
        return AxClient.load_from_json_file(save_path)

class Ax_LTspice_Bode_Fitter:
    def __init__(self, 
                 ltspice_wrapper: LTspice_Wrapper,
                 target_tf: sp.Expr,
                 design_params: Dict[str, List[float]],  # optional bound as List[float] --> [lower, upper]
                 c_range: List[float] = [1e0, 1e3],  # default in pico Farad 
                 r_range: List[float] = [1e1, 1e6],  # default in Ohm
                 output_node: str = "Vout",
                 frequency_weight: Frequency_Weight = None,
                 max_loss: float = 20,
                 loss_fn:  str   = 'mse',
                 norm_method: str = "min-max",
                 random_seed: int = 42,):
        
        self.ltspice_wrapper = ltspice_wrapper
        self.target_tf = target_tf
        self.design_params = design_params
        self.c_range = c_range
        self.r_range = r_range
        self.output_node = output_node

        self.frequency_weight = frequency_weight
        self.random_seed = random_seed
        self.norm_method = norm_method
        self.max_loss = max_loss
        self.loss_fn = loss_fn

        init_notebook_plotting()
        self.ax_client  = None
        self.gs         = None # Generation Strategy
        self.parameters = None
        self.helper_functions = Transfer_Func_Helper()
        self.target_complex_response: torch.Tensor = None
        self.frequency_array: torch.Tensor = None # is resolved the first time the LTspice is run

        self.optimization_log = []

    def parameterize(self, log_scale: bool = True) -> Tuple[Dict, List, List]:

        parameters: List[Dict] = []

        for var, given_bounds in self.design_params.items():
            bound = None # Reset range

            if isinstance(given_bounds, list) and len(given_bounds) == 2:
                bound = given_bounds
            elif var.startswith("C"):
                bound = self.c_range
            elif var.startswith("R"):
                bound = self.r_range
            else:
                bound = [1, 1e2] # default bound for undetermined variables!
                print(f"{var} does not have a valid defined range using the default bound {bound}")
            parameters.append({
                "name": str(var),
                "type": "range",        # Type of parameter (range, choice, etc.)
                "bounds": bound,        
                "value_type": "float",  
                "log_scale": log_scale       # Indicate that the parameter should be on a log scale
            })

        self.parameters = parameters
        
        return parameters
    
    def parameterize_update(self, variable: str, bounds: List[float]):
        """
        Updates the bound of the specified variable in design_params.
        
        Args:
            variable (str): The variable whose bounds are to be updated.
            bounds (List[float]): The new bounds for the variable.
        """
        # Iterate through design_params to find the variable and update its bounds
        for i, (var, _) in enumerate(self.design_params):
            if var == variable:
                self.design_params[i] = (var, bounds)
                print(f"Updated bounds for {variable}: {bounds}")
                break
        else:
            print(f"{variable} not found in design parameters.")
        
        # Reparameterize after updating the bounds
        return self.parameterize()

    def evaluate(self, params, include_phase_loss: bool = True, include_mag_loss: bool = True):

        # Spice simulation        
        self.ltspice_wrapper.update_params(parameterization=params)

        self.ltspice_wrapper.run_and_wait()

        # Only for the first run
        if self.frequency_array is None:
            self.frequency_array = self.ltspice_wrapper.extract_wave("frequency", is_real=True)
            self.target_complex_response = self.helper_functions.eval_tf(tf=self.target_tf, f_val=self.frequency_array)

        if self.frequency_weight is None or self.frequency_weight.weights is None:
            self.frequency_weight.parent_frequency_array = self.frequency_array
            self.frequency_weight.compute_weights()
        
        current_complex_response = self.ltspice_wrapper.extract_wave(self.output_node)

        fit_summary = get_bode_fitness_loss(current_complex_response=current_complex_response, 
                                                     target_complex_response=self.target_complex_response, 
                                                     freq_weights=self.frequency_weight.weights, 
                                                     norm_method='min-max',
                                                     loss_type=self.loss_fn)
        
        mag_loss     = fit_summary['mag_loss']
        phase_loss   = fit_summary['phase_loss']
        curr_max_mag = fit_summary['curr_max_mag']

        metric_value  = 0
        metric_value += mag_loss if include_mag_loss else 0
        metric_value += phase_loss if include_phase_loss else 0

        l1norm = torch.sum(torch.tensor([val for val in params.values()]))

        self.optimization_log.append({
            "complex_response" : current_complex_response,
            "mag_loss" : mag_loss.detach(),
            "phase_loss" : phase_loss.detach(),
            "max_mag": curr_max_mag.detach(),
            "bode_fitting_loss" : metric_value.detach(),
            "l1norm" : l1norm.detach(),
            "params" : params
        })

        return {
            "bode_fitting_loss" : (metric_value.detach(), 0.0), 
            "max_mag" : (curr_max_mag.detach(), 0), 
            "l1norm" : (l1norm.detach(), 0)
            }

    def create_experiment(self, num_sobol_trials: int = 5, verbose_logging: bool = False) -> None:

        self.gs = GenerationStrategy(
                        steps=[
                            # 1. Initialization step (does not require pre-existing data and is well-suited for
                            # initial sampling of the search space)
                            GenerationStep(
                                model_name="Sobol Generator",
                                model=Models.SOBOL,
                                num_trials=num_sobol_trials,                # How many trials should be produced from this generation step
                                model_kwargs={"seed": self.random_seed},    # Any kwargs you want passed into the model
                                model_gen_kwargs={"torch_device": device},  # Any kwargs you want passed to `modelbridge.gen`
                            ),
                            # 2. Bayesian optimization step (requires data obtained from previous phase and learns
                            # from all data available at the time of each new candidate generation call)
                            GenerationStep(
                                model=Models.BOTORCH_MODULAR,
                                num_trials=-1,                       # No limitation (-1) on how many trials should be produced from this step
                                model_gen_kwargs={"torch_device": device},  # Any kwargs you want passed to `modelbridge.gen`
                            ),
                        ]
                    )

        self.ax_client = AxClient(
            generation_strategy=self.gs,
            random_seed=self.random_seed,
            verbose_logging=verbose_logging)

        if self.parameters is None:
            print("Need to parameterize the variables first")
            return
        
        self.ax_client.create_experiment(
            name="Transfer Function Optimization",  
            parameters=self.parameters,
            objectives={
                "bode_fitting_loss": ObjectiveProperties(minimize=True, threshold=0),
                },
            overwrite_existing_experiment=True,
            # parameter_constraints = ["C_1 + C_2 <= 2.0"],
            # outcome_constraints = ["tf_fitting_loss <= 10"],
            tracking_metric_names= ["l1norm"]
        )

    def optimization_loop(self, num_trials: int = 20, include_mag_loss: bool = True, include_phase_loss: bool = True, verbose_logging: bool = True):
        # Save current logging level
        previous_logging_level = logging.getLogger().level
        
        if not verbose_logging:
            logging.getLogger().setLevel(logging.CRITICAL)
        
        try:
            for _ in tqdm(range(num_trials), desc="Optimizing", unit="trial"):
                parameterization, trial_index = self.ax_client.get_next_trial()
                self.ax_client.complete_trial(trial_index=trial_index, raw_data=self.evaluate(parameterization, include_phase_loss=include_phase_loss, include_mag_loss=include_mag_loss))  # Tell Ax the outcome
        finally:
            # Restore previous logging level
            logging.getLogger().setLevel(previous_logging_level)
    
    # Utility
    def spicelib_cleanup(self):
        self.ltspice_wrapper.runner.cleanup_files()

    def save_ax(self, name: str) -> str:
        # Generate a timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Create a file path with the timestamp
        dir = "./checkpoints"
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

        save_path = f"{dir}/{name}_{timestamp}.json"

        # Save the AxClient data to the JSON file with the timestamped path
        self.ax_client.save_to_json_file(save_path)
        return save_path
    
    @classmethod
    def load_ax(save_path: str) -> AxClient:
        return AxClient.load_from_json_file(save_path)
    
