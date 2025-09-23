import logging
import torch
import numpy        as np
import sympy        as sp
import nevergrad    as ng
import plotly.graph_objects as go
from   typing import Dict, List, Tuple, Any
from   tqdm   import tqdm
from abc import ABC, abstractmethod
from pathlib import Path

# Symxplorer Specific Imports
from   symxplorer.spice_engine.spicelib   import Spicelib_Wrapper, Sim_Execution_Type
from   symxplorer.designer_tools.domains  import Project_Setup, OptimizerConfig, Param
from   .symbolic_sizing import Symbolic_Sizing_Assist
from   .utils           import plot_complex_response, get_bode_fitness_loss, Transfer_Func_Helper, Frequency_Weight, UNIT_DICT

logger = logging.getLogger("SymXplorer.optimizer")

s = sp.symbols("s")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype  = torch.double

torch.set_default_dtype(dtype)
torch.set_default_device(device)
logger.info(f'Using device: {device} and dtype: {dtype}')


def log_normalize(p, pmin, pmax) -> float:
    """
    Log-normalize parameter p to [0, 1] range.
    p, pmin, pmax must be > 0.
    Always returns a float.
    """
    p = np.asarray(p, dtype=np.float64)
    pmin = np.asarray(pmin, dtype=np.float64)
    pmax = np.asarray(pmax, dtype=np.float64)

    log_p = np.log10(p)
    log_min = np.log10(pmin)
    log_max = np.log10(pmax)
    result = (log_p - log_min) / (log_max - log_min)
    return float(np.asarray(result, dtype=np.float64))


def log_denormalize(x, pmin, pmax) -> float:
    """
    Map normalized x in [0, 1] back to physical parameter using log scaling.
    Always returns a float.
    """
    x = np.asarray(x, dtype=np.float64)
    pmin = np.asarray(pmin, dtype=np.float64)
    pmax = np.asarray(pmax, dtype=np.float64)

    log_min = np.log10(pmin)
    log_max = np.log10(pmax)
    log_p = x * (log_max - log_min) + log_min
    result = 10.0 ** log_p
    return float(np.asarray(result, dtype=np.float64))


def linear_normalize(p, pmin, pmax) -> float:
    """
    Linearly normalize parameter p to [0, 1] range.
    Always returns a float.
    """
    p = np.asarray(p, dtype=np.float64)
    pmin = np.asarray(pmin, dtype=np.float64)
    pmax = np.asarray(pmax, dtype=np.float64)

    result = (p - pmin) / (pmax - pmin)
    return float(np.asarray(result, dtype=np.float64))


def linear_denormalize(x, pmin, pmax) -> float:
    """
    Map normalized x in [0, 1] back to physical parameter linearly.
    Always returns a float.
    """
    x = np.asarray(x, dtype=np.float64)
    pmin = np.asarray(pmin, dtype=np.float64)
    pmax = np.asarray(pmax, dtype=np.float64)

    result = pmin + x * (pmax - pmin)
    return float(np.asarray(result, dtype=np.float64))

# ----------------------------
# --- Class Definitions ---
# ----------------------------
class Nevergrad_Base_Optimizer(ABC):
    def __init__(self, optimizer_config : OptimizerConfig):
        self.optimizer_config = optimizer_config
        # The following Properties are instantiated by the class
        self.optimizer: ng.optimization.base.Optimizer | None = None
        self.optimizer_trace: List[Dict[str, Any]] = []
        self.loss_values : List[float] = []
        self.global_best_index: int = 0 # the index of the global best solution
        self.parametrization: ng.p.Dict | None = None

        self.logger = logger
    
    def _create_experiment(self) -> bool:
        if self.parametrization is None:
            logger.critical("NEED TO CALL self.parameterize")
            return False
        
        registry = ng.optimizers.registry.get(self.optimizer_config.name)
        if registry is not None:
            self.optimizer = registry(parametrization=self.parametrization, budget=self.optimizer_config.budget)
            logger.info(f"Optimizer is set to {self.optimizer.name} with budget = {self.optimizer_config.budget}")
            return True
        return False
    
    @abstractmethod
    def parameterize(self) ->  ng.p.Dict:
        """Returns the parametrization dictionary for nevergrad and any denormalization factors needed"""
        pass

    @abstractmethod
    def denormalize_params(self, parameterization: Dict[str, float]) -> Dict[str, float]:
        """Convert the normalized parameters back to the original scale"""
        pass

    @abstractmethod
    def evaluate(self, parameterization: Dict[str, float]) -> np.float64:
        """Evaluate the objective function for the given parameterization"""
        pass
        
    def optimize(self, render_optimization_trace: bool = False) -> List[Dict[str, Any]] | None:
        """Run the optimization process for a given budget and returns the optimization trace as 
        a list of (parameterization, loss) tuples"""
        
        self._create_experiment()
        
        if self.optimizer is None:
            logger.critical("Oops... The optimizer object was not created!")
            return None
        
        # Track the loss for plotting
        self.loss_values : List[float] = []
        self.optimizer_trace = []  # Store the optimization trace
        
        # Run the optimization process
        for trial in tqdm(range(self.optimizer_config.budget), desc="Optimizing", unit="trial"):
            # Get a new candidate
            candidate : ng.p.Parameter = self.optimizer.ask()
            # Evaluate function
            denorm_params: Dict[str, float] = self.denormalize_params(parameterization=candidate.value)
            curr_loss : np.float64   = self.evaluate(parameterization=denorm_params)
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
                logger.info(f"a New fit was found... trial {trial} loss {curr_loss:.2f}")
        
        # Plot the loss as a function of optimization step
        if render_optimization_trace:
            self.plot_loss()

        return self.optimizer_trace
    
    def get_best_params(self) -> Tuple[Dict[str, float], float] | None:

        if self.optimizer is None:
            logger.info("Need to set the optimizer by calling self.create_experiment")
            return
        if len(self.optimizer_trace) < 1:
            logger.info("need to run self.optimize")
            return
        
        point = self.optimizer_trace[self.global_best_index]
        best_solution : ng.p.Parameter = point['params']
        loss : float = point['loss']

        # logger.info("Optimized x - normalized:", best_solution)
        # logger.info("Optimized x - de-normalized:", self.denormalize_params(best_solution))
        logger.info(f"best loss: {float(loss)}")

        return self.denormalize_params(best_solution), loss
    
    def plot_loss(self, save_path: Path | None = None, show: bool = False):
        """Plot the loss as a function of optimization steps with Plotly."""

        if len(self.optimizer_trace) < 1:
            logger.warning("No optimization trace to plot")
            return

        loss_values = [entry["loss"] for entry in self.optimizer_trace]
        x_values = list(range(len(loss_values)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_values,
            y=loss_values,
            mode="markers+lines",
            name="Loss",
            line=dict(color="blue", width=2)
        ))

        fig.update_layout(
            title="Loss vs. Optimization Trial",
            xaxis_title="Optimization Step",
            yaxis_title="Loss",
            template="plotly_dark",
            showlegend=True
        )

        # Save to file if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_path))
            logger.info(f"📊 Plot saved to {save_path}")

        # Optionally show interactively (browser popup)
        if show:
            logger.info("Opening interactive plot in browser...")
            fig.show()

class Nevergrad_Spice_Bode_Optimizer(Nevergrad_Base_Optimizer):
    def __init__(self,
                 setup_obj: Project_Setup,
                 spicelib_wrapper : Spicelib_Wrapper,
                 target_tf: sp.Expr,
                 output_node: str = "Vout", # FIXME this needs to go into the spicelib_wrapper
                 frequency_weight: Frequency_Weight | None = None
                 ):
        
        if setup_obj.optimizer_config is None:
            raise ValueError("cannot use a Null optimizer_config instance")
        
        super().__init__(optimizer_config = setup_obj.optimizer_config)
        self.setup_obj = setup_obj
        self.spicelib_wrapper = spicelib_wrapper
        self.target_tf = target_tf
        self.output_node = output_node
        self.frequency_weight  = frequency_weight

        self.helper_functions = Transfer_Func_Helper()
        # To be calculated during the program runtime
        self.target_complex_response: torch.Tensor  | None = None
        self.frequency_array: torch.Tensor | None = None # is resolved the first time the LTspice is run
        self.optimization_log : List[Dict[str, Any]] = []

    # --- Overwriting the Abstract Methods ---
    def parameterize(self) -> ng.p.Dict:
        super().parameterize()
        
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
    
    def denormalize_params(self, parameterization: Dict[str, float]) -> Dict[str, float]:
        denorm_params: Dict[str, float] = {}        
        for param_name in parameterization:
            val = parameterization[param_name]
            param_obj = self.setup_obj.get_param_by_name(name=param_name)

            if param_obj is None:
                raise KeyError(f"Could not find param name {param_name} in {self.setup_obj.list_params()}")
            
            if param_obj.log_scale:
                denorm_params[param_name] = log_denormalize(x=val, pmin=param_obj.min_val, pmax=param_obj.max_val)
            else:
                denorm_params[param_name] = linear_denormalize(x=val, pmin=param_obj.min_val, pmax=param_obj.max_val)

        return denorm_params

    def evaluate(self, parameterization: Dict[str, float]) -> np.float64:
        """
        Evaluate the given parameterization by running a SPICE simulation,
        computing the fitness loss, and returning it as np.float64.
        """

        # 1 - Run a SPICE simulation
        # ---------------------------------------------------------------
        self.spicelib_wrapper.update_params(parameterization=parameterization)
        curr_raw, curr_log, task_name = self.spicelib_wrapper.run_and_wait(exe_log=True)
        
        # 2 - Extract frequency array (first run only)
        # ---------------------------------------------------------------
        if self.frequency_array is None:
            try:
                self.frequency_array = self.spicelib_wrapper.extract_wave("frequency", is_real=True)
            except IndexError:
                logger.critical("Attempted to look up the 'frequency' trace but it doesnt exist in the RAW file")
                raise RuntimeError("Attempted to look up the 'frequency' trace but it doesnt exist in the RAW file")
            self.examine_target(f_array=self.frequency_array)

        if self.frequency_weight is None:
            raise RuntimeError("frequency_weight must be specified.")
        if self.frequency_weight.weights is None:
            self.frequency_weight.parent_frequency_array = self.frequency_array
            self.frequency_weight.compute_weights()

        # 3 - Extract circuit response
        # ---------------------------------------------------------------
        current_complex_response = self.spicelib_wrapper.extract_wave(self.output_node)

        if self.setup_obj.optimizer_config is None:
            raise RuntimeError("Optimizer config cannot be None.")
        if self.target_complex_response is None:
            raise RuntimeError("Reached the comparison between target and simulated performance but the target was not computed... make sure self.examine_target works correctly.")
        
        loss_fn_config = self.setup_obj.optimizer_config.loss_function
        fit_summary = get_bode_fitness_loss(
            current_complex_response=current_complex_response,
            target_complex_response=self.target_complex_response,
            freq_weights=self.frequency_weight.weights,
            norm_method=loss_fn_config.loss_norm_method,
            loss_type=loss_fn_config.loss_type,
            rescale=loss_fn_config.rescale_mag
        )

        mag_loss   = fit_summary['mag_loss']
        phase_loss = fit_summary['phase_loss']

        mag, _ = self.helper_functions.get_mag_phase_from_complex_response(
            complex_response_array=current_complex_response
        )

        # --- Compute final metric (NumPy only) ---
        metric_value = np.float64(0.0)
        metric_value += np.float64(mag_loss if loss_fn_config.include_mag_loss else 0.0)
        metric_value += np.float64(phase_loss if loss_fn_config.include_phase_loss else 0.0)
        metric_value += np.float64(
            max(0.0, fit_summary['target_max_mag'] - fit_summary['curr_max_mag']) ** 2
        )

        # --- Log results ---
        self.optimization_log.append({
            "complex_response": current_complex_response,
            "mag_loss": np.float64(mag_loss),
            "phase_loss": np.float64(phase_loss),
            "max_mag": np.float64(fit_summary['curr_max_mag']),
            "bode_fitting_loss": np.float64(metric_value),
            "params": parameterization
        })

        logger.debug(f"finished the trial evaluation.... summary")
        logger.debug(f"\tmetric_value = {metric_value}")
        logger.debug(f"\t\t- mag_loss : {mag_loss}")
        logger.debug(f"\t\t- phase_loss : {phase_loss}")

        return np.float64(metric_value)

    # --- Helper Methods (only in child class) ---
    def examine_target(self, f_array: torch.Tensor):
        logger.info(f"computing the target complex response for {self.target_tf}")
        self.target_complex_response = self.helper_functions.eval_tf(tf=self.target_tf, f_val=f_array)
        # mag, _ = self.helper_functions.get_mag_phase_from_complex_response(self.target_complex_response)
  

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
        
        self.parametrization: ng.p.Dict | None = None
        self.cap_denormailization: float| None = None
        self.res_denormailization: float| None  = None
        self.helper_functions = Transfer_Func_Helper()
        self.optimizer: ng.optimization.base.Optimizer| None = None
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

