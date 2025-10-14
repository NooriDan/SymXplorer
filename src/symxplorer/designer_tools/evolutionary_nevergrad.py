import logging
import json
import torch
import numpy        as np
import sympy        as sp
import nevergrad    as ng
import plotly.graph_objects as go

from    typing      import Dict, List, Tuple, Any, Optional
from    tqdm        import tqdm
from    abc         import ABC, abstractmethod
from    pathlib     import Path
from    spicelib    import RawRead
from    dacite      import from_dict, Config
<<<<<<< HEAD
from    dataclasses import asdict
=======
from    dataclasses import asdict, is_dataclass
>>>>>>> origin/main
from    datetime    import datetime


# Symxplorer Specific Imports
from   symxplorer.spice_engine.spicelib     import Spicelib_Wrapper
from   symxplorer.designer_tools.domains    import Project_Setup, OptimizerConfig, ListTargetSpec, TargetSpec
from   symxplorer.designer_tools.domains    import OptimizationGoalType, OptimizationPoint, OptimizationLogEntry, Error_Types
<<<<<<< HEAD
from   symxplorer.designer_tools.utils      import compute_error, compute_reward, convert_linear_to_log, log_denormalize, linear_denormalize
=======
from   symxplorer.designer_tools.utils      import compute_error, convert_linear_to_log, log_denormalize, linear_denormalize
from   .symbolic_sizing                     import Symbolic_Sizing_Assist
>>>>>>> origin/main
from   .utils                               import plot_complex_response, get_bode_fitness_loss, Transfer_Func_Helper, Frequency_Weight, UNIT_DICT

logger = logging.getLogger("SymXplorer.optimizer")

s = sp.symbols("s")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype  = torch.double

torch.set_default_dtype(dtype)
torch.set_default_device(device)
logger.info(f'Using device: {device} and dtype: {dtype}')

# ----------------------------
# --- Global Constants ---
# ----------------------------
<<<<<<< HEAD
MAX_PENALTY = np.float64(1e6) # The maximum score used when a trial does not have a performance metric in it.
MAX_REWARD  = np.float64(1e6) # The maximum reward score a spec can achieve.
CHECKPOINT_SCHEMA_VERSION = "1.0.0"
EPSILON = np.float64(1e-12)
=======
MAX_LOSS = np.float64(1e12) # The maximum loss used when a trial does not have a performance metric in it.
CHECKPOINT_SCHEMA_VERSION = "1.0.0"

>>>>>>> origin/main

# ----------------------------
# --- Class Definitions ---
# ----------------------------

# ------------------------------------------------
# [ABSTRACT] Nevergrad Optimizer Class
# ------------------------------------------------
class Nevergrad_Base_Optimizer(ABC):
    def __init__(self, optimizer_config : OptimizerConfig):
        self.optimizer_config = optimizer_config
        # The following Properties are instantiated by the class
        self.optimizer: ng.optimization.base.Optimizer | None = None
        self.optimizer_trace: List[OptimizationPoint] = []
<<<<<<< HEAD
        self.score_values : List[float] = []
=======
        self.loss_values : List[float] = []
>>>>>>> origin/main
        self.global_best_index: int = 0 # the index of the global best solution
        self.parametrization: ng.p.Dict | None = None

        self.logger = logger
        self.verbose: bool = True
    
    def _create_experiment(self) -> bool:
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
    
    @abstractmethod
    def parameterize(self) ->  ng.p.Dict:
        """Returns the parametrization dictionary for nevergrad and any denormalization factors needed"""
        pass

    @abstractmethod
    def denormalize_params(self, parameterization: Dict[str, float]) -> Dict[str, float]:
        """Convert the normalized parameters back to the original scale"""
        pass

    @abstractmethod
    def evaluate(self, parameterization: Dict[str, float]) -> Tuple[np.float64, Dict[str, Any]]:
        """Evaluate the objective function for the given parameterization (de-normalized)"""
        pass

    @abstractmethod
    def compute_fitness(self, performance_array: Dict[str, np.float64 | torch.Tensor]) -> Tuple[np.float64, Dict[str, Any]]:
        pass

    def optimize(self, render_optimization_trace: bool = False) -> List[OptimizationPoint] | None:
        """Run the optimization process for a given budget and returns the optimization trace as 
<<<<<<< HEAD
        a list of (parameterization, score) tuples"""
=======
        a list of (parameterization, loss) tuples"""
>>>>>>> origin/main

        logger.info("Optimization process started.")

        self._create_experiment()
        
        if self.optimizer is None:
            logger.critical("Oops... The optimizer object was not created!")
            return None
        
        # Track the score for plotting
        self.score_values : List[float] = []
        self.optimizer_trace = []  # Store the optimization trace
        
        # Run the optimization process
        for trial in tqdm(range(self.optimizer_config.budget), desc="Optimizing", unit="trial"):
            logger.debug(f"STARTING trial {trial+1}/{self.optimizer_config.budget}...")
            # Get a new candidate
            candidate : ng.p.Parameter = self.optimizer.ask()
            # Evaluate function
            denorm_params: Dict[str, float] = self.denormalize_params(parameterization=candidate.value)
            curr_score, metadata = self.evaluate(parameterization=denorm_params)
            # Provide feedback to optimizer (The negative of the fitness score is used because the optimizer is set to minimize this value... this way the optimizer will maximize the fitness score.
            self.optimizer.tell(candidate, -1 * curr_score)
            
<<<<<<< HEAD
            # Log the achieved score
            self.optimizer_trace.append(OptimizationPoint(params=candidate.value, score=curr_score, metadata=metadata))

            logger.debug(f"Trial {trial+1}/{self.optimizer_config.budget} COMPLETED with score: {curr_score:.4f}")

            # Update the index of the global best solution (lowest score)
            if curr_score > self.optimizer_trace[self.global_best_index].score:
=======
            # Log the achieved loss
            self.optimizer_trace.append(OptimizationPoint(params=candidate.value, loss=curr_loss, metadata=metadata))

            logger.debug(f"Trial {trial+1}/{self.optimizer_config.budget} COMPLETED with loss: {curr_loss:.4f}")

            # Update the index of the global best solution (lowest loss)
            if curr_loss < self.optimizer_trace[self.global_best_index].loss:
>>>>>>> origin/main
                self.global_best_index = trial
                logger.info(f"a New fit was found... trial {trial} score {curr_score:.2f}")
        
        # Plot the score as a function of optimization step
        if render_optimization_trace:
<<<<<<< HEAD
            self.plot_score()
=======
            self.plot_loss()
>>>>>>> origin/main
        logger.info("Optimization process completed.")
        return self.optimizer_trace
    
    def get_best_params(self, verbose: bool = False) -> Tuple[Dict[str, float], float, Dict[str, Any]] | None:
<<<<<<< HEAD
        """Retrieve the best parameters and corresponding score from the optimization trace."""
=======
        """Retrieve the best parameters and corresponding loss from the optimization trace."""
>>>>>>> origin/main
        
        if self.optimizer is None:
            logger.info("Need to set the optimizer by calling self.create_experiment")
            return
        if len(self.optimizer_trace) < 1:
            logger.info("need to run self.optimize")
            return
        
        point = self.optimizer_trace[self.global_best_index]
        best_solution : ng.p.Parameter = point.params
<<<<<<< HEAD
        score : float = point.score
=======
        loss : float = point.loss
>>>>>>> origin/main

        if verbose:
            logger.info("Optimized x - normalized:", best_solution)
            logger.info("Optimized x - de-normalized:", self.denormalize_params(best_solution))
<<<<<<< HEAD
        logger.info(f"best score: {float(score)}")

        return self.denormalize_params(best_solution), score, point.metadata
    
    def plot_score(self, save_path: Path | None = None, show: bool = False):
        """Plot the score as a function of optimization steps with Plotly."""
=======
        logger.info(f"best loss: {float(loss)}")

        return self.denormalize_params(best_solution), loss, point.metadata
    
    def plot_loss(self, save_path: Path | None = None, show: bool = False):
        """Plot the loss as a function of optimization steps with Plotly."""
>>>>>>> origin/main
        logger = logging.getLogger("SymXplorer.plotter")

        if len(self.optimizer_trace) < 1:
            logger.warning("No optimization trace to plot")
            return

<<<<<<< HEAD
        score_values = [entry.score for entry in self.optimizer_trace]
        x_values = list(range(len(score_values)))

        # Compute running best (cumulative maximum)
        best_scores = np.maximum.accumulate(np.array(score_values))
=======
        loss_values = [entry.loss for entry in self.optimizer_trace]
        x_values = list(range(len(loss_values)))
>>>>>>> origin/main

        # Compute running best (cumulative minimum)
        best_losses = []
        current_best = float("inf")
        for val in loss_values:
            current_best = min(current_best, val)
            best_losses.append(current_best)

        fig = go.Figure()

<<<<<<< HEAD
        # Plot raw score values
=======
        # Plot raw loss values
>>>>>>> origin/main
        fig.add_trace(go.Scatter(
            x=x_values,
            y=score_values,
            mode="markers+lines",
<<<<<<< HEAD
            name="Score",
=======
            name="Loss",
>>>>>>> origin/main
            line=dict(color="blue", width=2),
            opacity=0.6
        ))

        # Plot best-so-far curve
        fig.add_trace(go.Scatter(
            x=x_values,
<<<<<<< HEAD
            y=best_scores,
            mode="lines",
            name="Best Score So Far",
=======
            y=best_losses,
            mode="lines",
            name="Best Loss So Far",
>>>>>>> origin/main
            line=dict(color="red", width=2)
        ))

        fig.update_layout(
            title="Score vs. Optimization Trial",
            xaxis_title="Optimization Step",
            yaxis_title="Score",
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

    def plot_design_space_exploration(self, param_x: str, param_y: str, save_path: Path | None = None, show: bool = False, denorm: bool = True) -> Tuple[torch.Tensor, torch.Tensor] | None:
        """Plot the exploration of the design space in terms of two parameters with Plotly."""
        logger = logging.getLogger("SymXplorer.plotter")

        if len(self.optimizer_trace) < 1:
            logger.warning("No optimization trace to plot")
            return None

        if param_x not in self.optimizer_trace[0].params:
            logger.warning(f"param_x '{param_x}' not found in optimization trace")
            return None

        if param_y not in self.optimizer_trace[0].params:
            logger.warning(f"param_y '{param_y}' not found in optimization trace")
            return None

        # De-normalize
        if denorm:
            denormalized_params = [self.denormalize_params(entry.params) for entry in self.optimizer_trace]
            x_values = torch.tensor([entry[param_x] for entry in denormalized_params], device=device)
            y_values = torch.tensor([entry[param_y] for entry in denormalized_params], device=device)
        else:
            x_values = torch.tensor([entry.params[param_x] for entry in self.optimizer_trace], device=device)
            y_values = torch.tensor([entry.params[param_y] for entry in self.optimizer_trace], device=device)
        
<<<<<<< HEAD
        loss      = torch.tensor([entry.score for entry in self.optimizer_trace], device=device)
=======
        loss      = torch.tensor([entry.loss for entry in self.optimizer_trace], device=device)
>>>>>>> origin/main

        
        fig = go.Figure()

        # Scatter with heatmap coloring by FOM
        fig.add_trace(go.Scatter(
            x=x_values.cpu().numpy(),
            y=y_values.cpu().numpy(),
            mode="markers",
            marker=dict(
                size=10,
                color=loss.cpu().numpy(),   # heatmap coloring
                colorscale="Viridis",      # you can change to "Plasma", "Cividis", etc.
<<<<<<< HEAD
                colorbar=dict(title="Score"),
=======
                colorbar=dict(title="Loss"),
>>>>>>> origin/main
                showscale=True
            ),
            name="Design Space Exploration"
        ))

        fig.update_layout(
            title=f"Design Space Exploration: {param_y} vs. {param_x}",
            xaxis_title=param_x,
            yaxis_title=param_y,
            template="plotly_dark",
            showlegend=False
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

        return x_values, y_values
    
    @abstractmethod
    def plot_solution(self, parameterization: Dict[str, float], **kwargs):
        pass

# ------------------------------------------------
# A [ABSTRACT] SPICE-based Optimizers
# ------------------------------------------------
class Nevergrad_Spice_Base_Optimizer(Nevergrad_Base_Optimizer):
    """ Base class for Nevergrad optimizers that use SPICE simulations. """
    def __init__(self,  
                setup_obj: Project_Setup,
                spicelib_wrapper : Spicelib_Wrapper):
        
        if setup_obj.optimizer_config is None:
            raise ValueError("cannot use a Null optimizer_config instance")
        super().__init__(optimizer_config = setup_obj.optimizer_config)

        self.setup_obj = setup_obj
        self.spicelib_wrapper = spicelib_wrapper

        self.optimization_log : List[OptimizationLogEntry] = []
    
    # --- Overwriting Some Abstract Methods ---
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

        log_range = self.setup_obj.optimizer_config.get_log_variable_range()
        lin_range = self.setup_obj.optimizer_config.get_lin_variable_range()

        for param_name in parameterization:
            val = parameterization[param_name]
            param_obj = self.setup_obj.get_param_by_name(name=param_name)

            if param_obj is None:
                raise KeyError(f"Could not find param name {param_name} in {self.setup_obj.list_params()}")
            
            if param_obj.log_scale:
                denorm_params[param_name] = log_denormalize(x=val/log_range, pmin=param_obj.min_val, pmax=param_obj.max_val)
            else:
                denorm_params[param_name] = linear_denormalize(x=val/lin_range, pmin=param_obj.min_val, pmax=param_obj.max_val)

        return denorm_params

    def save_checkpoint(self, name: str | Path) -> None:
        """Save optimizer state to JSON with schema versioning."""
        
        # Clean up optimization log
        cleaned_optimization_log = []
        for e in self.optimization_log:
<<<<<<< HEAD
            if not isinstance(e.log_file, str):
                e.log_file = str(e.log_file)
            cleaned_optimization_log.append(asdict(e))

        timestamp = datetime.now().isoformat()
        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "timestamp": timestamp,
            "optimization_log": cleaned_optimization_log,
        }
        
        p = Path(name).with_suffix(".json")
        path = p.with_name(f"{p.stem}_{timestamp}{p.suffix}")
=======
            if not isinstance(e.log, str):
                e.log = str(e.log)
            cleaned_optimization_log.append(asdict(e))

        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "timestamp": datetime.now().isoformat(),
            "optimization_log": cleaned_optimization_log,
        }

        path = Path(name).with_suffix(".json")
>>>>>>> origin/main
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"✅ Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, setup_obj: Project_Setup, spicelib_wrapper: Spicelib_Wrapper, path_to_checkpoint: str | Path) -> "Nevergrad_Spice_Base_Optimizer":
        """Load optimizer and project setup from JSON checkpoint with version validation."""
        path = Path(path_to_checkpoint)
        with open(path, "r") as f:
            data = json.load(f)

        # Validate schema version
        version = data.get("schema_version")
        if version != CHECKPOINT_SCHEMA_VERSION:
            logger.warning(f"⚠️ Checkpoint version mismatch: {version} != {CHECKPOINT_SCHEMA_VERSION}")

        # Recreate optimizer instance
        obj = cls(setup_obj=setup_obj, spicelib_wrapper=spicelib_wrapper)

        # Rebuild optimization log
        obj.optimization_log = [
            from_dict(OptimizationLogEntry, entry, Config(strict=False))
            for entry in data.get("optimization_log", [])
        ]

        logger.info(f"✅ Checkpoint loaded successfully from {path}")
        return obj
    
    # --- Helper Methods (only in child class) ---
    def simulate_circuit(self, parameterization: Dict[str, float], save_sim_override: bool = False) -> RawRead:
        logger.debug("Simulating the circuit with the given parameterization")
        self.spicelib_wrapper.update_params(parameterization=parameterization)
        curr_raw, curr_log, task_name = self.spicelib_wrapper.run_and_wait(exe_log=True)
        if curr_raw is None:
            logger.critical("Something went wrong during simulation as no RAW file was generated")
            raise RuntimeError("Something went wrong during simulation as no RAW file was generated")
        
        if not self.setup_obj.save_sim and not save_sim_override:
            self.spicelib_wrapper.clean_up()
        return curr_raw
    
    def plot_optimization_trace(self, metric_x: str, metric_y: str, save_path: Path | None = None, show: bool = False) -> Tuple[torch.Tensor, torch.Tensor] | None:
        logger = logging.getLogger("SymXplorer.plotter")
        if len(self.optimization_log) < 1:
            logger.warning("No optimization log to plot")
            return None

        if metric_x not in self.optimization_log[0].fit_summary:
            logger.warning(f"metric_x '{metric_x}' not found in optimization log")
            return None

        if metric_y not in self.optimization_log[0].fit_summary:
            logger.warning(f"metric_y '{metric_y}' not found in optimization log")
            return None

        x_values = torch.tensor([entry.fit_summary[metric_x]['curr_val'] for entry in self.optimization_log], device=device)
        y_values = torch.tensor([entry.fit_summary[metric_y]['curr_val'] for entry in self.optimization_log], device=device)
<<<<<<< HEAD
        fom      = torch.tensor([entry.score for entry in self.optimization_log], device=device)
=======
        fom      = torch.tensor([entry.metric_value for entry in self.optimization_log], device=device)
>>>>>>> origin/main

        fig = go.Figure()

        # Scatter with heatmap coloring by FOM
        fig.add_trace(go.Scatter(
            x=x_values.cpu().numpy(),
            y=y_values.cpu().numpy(),
            mode="markers",
            marker=dict(
                size=10,
                color=fom.cpu().numpy(),   # heatmap coloring
                colorscale="Viridis",      # you can change to "Plasma", "Cividis", etc.
                colorbar=dict(title="FOM"),
                showscale=True
            ),
            name="Optimization Trace"
        ))

        fig.update_layout(
            title=f"Optimization Trace: {metric_y} vs. {metric_x}",
            xaxis_title=metric_x,
            yaxis_title=metric_y,
            template="plotly_dark",
            showlegend=False
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

        return x_values, y_values

<<<<<<< HEAD
    def plot_score_value_by_spec(self, spec_name: str, save_path: Path | None = None, show: bool = False):
        """
        Plot the score value for a specific target spec over the optimization trials.
=======
    def plot_loss_value_by_spec(self, spec_name: str, save_path: Path | None = None, show: bool = False):
        """
        Plot the loss value for a specific target spec over the optimization trials.
>>>>>>> origin/main
        Includes target spec value, tolerance band, and error type information.
        """
        logger = logging.getLogger("SymXplorer.plotter")
        if len(self.optimization_log) < 1:
            logger.warning("No optimization log to plot")
            return

        if spec_name not in self.optimization_log[0].fit_summary:
            logger.warning(f"spec_name '{spec_name}' not found in optimization log")
            return

        # Extract values from optimization log
        spec_values = [entry.fit_summary[spec_name]['curr_val'] for entry in self.optimization_log]
<<<<<<< HEAD
        score_values = [entry.fit_summary[spec_name]['score'] for entry in self.optimization_log]
        
        logger.info(f"\tmin score {min(score_values)}; max score {max(score_values)}")
=======
        loss_values = [entry.fit_summary[spec_name]['loss'] for entry in self.optimization_log]
        
        logger.info(f"\tmin loss {min(loss_values)}; max loss {max(loss_values)}")
>>>>>>> origin/main
        
        # Get TargetSpec definition
        target_spec = self.setup_obj.optimizer_config.target_specs.get_target_by_name(spec_name)
        if target_spec is None:
            logger.warning(f"No TargetSpec found for '{spec_name}'")
            return

        target_val = float(target_spec.target)
        tolerance  = float(target_spec.tolerance if target_spec.tolerance is not None else 0.05*target_val)
        error_type = target_spec.error_type

        fig = go.Figure()

        # Scatter plot
        fig.add_trace(go.Scatter(
            x=spec_values,
<<<<<<< HEAD
            y=score_values,
            mode="markers",
            name=f"Score: {spec_name}",
=======
            y=loss_values,
            mode="markers",
            name=f"Loss: {spec_name}",
>>>>>>> origin/main
            marker=dict(color="blue", size=8, opacity=0.7, symbol="circle"),
        ))

        # Add vertical line at target value
        fig.add_vline(
            x=target_val,
            line=dict(color="red", width=2, dash="dash"),
            annotation_text=f"Target = {target_val:.2e}",
            annotation_position="top right",
            annotation_font=dict(color="red")
        )

        # Add tolerance bounds if available
        if tolerance is not None:
            if target_spec.goal != OptimizationGoalType.MINIMIZE:
                fig.add_vline(
                    x=target_val - tolerance,
                    line=dict(color="green", width=1, dash="dot"),
                    annotation_text=f"-tol ({target_val - tolerance:.2e})",
                    annotation_position="bottom left",
                    annotation_font=dict(color="green")
                )
            
            if target_spec.goal != OptimizationGoalType.EXCEED:
                fig.add_vline(
                    x=target_val + tolerance,
                    line=dict(color="green", width=1, dash="dot"),
                    annotation_text=f"+tol ({target_val + tolerance:.2e})",
                    annotation_position="bottom right",
                    annotation_font=dict(color="green")
                )

        # Dynamic title with error type info
        error_type_str = error_type.value if error_type else "unknown"
        goal_type_str = target_spec.goal.value if target_spec.goal else "unknown"

        fig.update_layout(
<<<<<<< HEAD
            title=f"Score for Spec '{spec_name}' (Error: {error_type_str}, Goal {goal_type_str})",
            xaxis_title=f"{spec_name} Value",
            yaxis_title="Score",
=======
            title=f"Loss for Spec '{spec_name}' (Error: {error_type_str}, Goal {goal_type_str})",
            xaxis_title=f"{spec_name} Value",
            yaxis_title="Loss",
>>>>>>> origin/main
            template="plotly_dark",
            showlegend=True
        )

        # Save to file if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_path))
            logger.info(f"📊 Plot saved to {save_path}")

        # Optionally show interactively
        if show:
            logger.info("Opening interactive plot in browser...")
            fig.show()

# ------------------------------------------------
<<<<<<< HEAD
# A.1 [CONCRETE] Bode Fitter
=======
# A.2 [CONCRETE] Bode Fitter
>>>>>>> origin/main
# ------------------------------------------------
class Nevergrad_Spice_Bode_Optimizer(Nevergrad_Spice_Base_Optimizer):
    """ Nevergrad optimizer that fits a SPICE-simulated transfer function to a target transfer function. """
    def __init__(self,
                 setup_obj: Project_Setup,
                 spicelib_wrapper : Spicelib_Wrapper,
                 target_tf: sp.Expr,
                 output_node: str = "Vout", # FIXME this needs to go into the spicelib_wrapper
                 frequency_weight: Frequency_Weight | None = None
                 ):
        
        super().__init__(setup_obj = setup_obj, spicelib_wrapper = spicelib_wrapper)

        self.target_tf = target_tf
        self.output_node = output_node
        self.frequency_weight  = frequency_weight

        self.helper_functions = Transfer_Func_Helper()
        # To be calculated during the program runtime
        self.target_complex_response: torch.Tensor  | None = None
        self.frequency_array: torch.Tensor | None = None # is resolved the first time the LTspice is run

    # --- Overwriting the Abstract Methods ---
    def evaluate(self, parameterization: Dict[str, float]) -> Tuple[np.float64, Dict[str, Any]]:
        """
        Evaluate the given parameterization by running a SPICE simulation,
        computing the fitness score, and returning it as np.float64.
        """
        # 1 - Run a SPICE simulation
        # ---------------------------------------------------------------
        raw = self.simulate_circuit(parameterization=parameterization)

        # 2 - Extract frequency array (first run only)
        # ---------------------------------------------------------------
        self.prepare_frequency_array()

        # 3 - Extract circuit response
        # ---------------------------------------------------------------
        current_complex_response = self.extract_circuit_response_from_latest_run()

        # 4 - Compute the fitness
        # ---------------------------------------------------------------
        fitness_score, fit_summary = self.compute_fitness({"current_complex_response" : current_complex_response})

        # --- Log results ---
        mag_loss   = fit_summary['mag_loss']
        phase_loss = fit_summary['phase_loss']
        
        self.optimization_log.append(
            OptimizationLogEntry(
                params=parameterization, 
<<<<<<< HEAD
                score=np.float64(fitness_score),
=======
                metric_value=np.float64(metric_value),
>>>>>>> origin/main
                fit_summary={
                    "mag_loss": np.float64(mag_loss),
                    "phase_loss": np.float64(phase_loss),
                    "max_mag": np.float64(fit_summary['curr_max_mag'])
                }, 
                metadata={"complex_response": current_complex_response},
                log=None
                )
            )


        logger.debug(f"finished the trial evaluation.... summary")
        logger.debug(f"\tmetric_value = {fitness_score}")
        logger.debug(f"\t\t- mag_loss : {mag_loss}")
        logger.debug(f"\t\t- phase_loss : {phase_loss}")

        return np.float64(fitness_score), fit_summary

    # --- Helper Methods (only in child class) ---
    def extract_circuit_response_from_latest_run(self) -> torch.Tensor:
        logger.debug("Extracting the circuit response from the latest RAW file")
        current_complex_response = self.spicelib_wrapper.extract_wave(self.output_node)
        return current_complex_response

    def examine_target(self, f_array: torch.Tensor):
        logger.info(f"computing the target complex response for {self.target_tf}")
        self.target_complex_response = self.helper_functions.eval_tf(tf=self.target_tf, f_val=f_array)
        # mag, _ = self.helper_functions.get_mag_phase_from_complex_response(self.target_complex_response)
    
    def compute_fitness(self, performance_array: Dict[str, np.float64 | torch.Tensor]) -> Tuple[np.float64, Dict[str, Any]]:
        
        current_complex_response: torch.Tensor = performance_array["current_complex_response"]
        
        if self.setup_obj.optimizer_config is None:
            raise RuntimeError("Optimizer config cannot be None.")
        if self.target_complex_response is None:
            raise RuntimeError("Reached the comparison between target and simulated performance but the target was not computed... make sure self.examine_target works correctly.")
        
        loss_fn_config = self.setup_obj.optimizer_config.loss_function_config
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

        return metric_value, fit_summary

    def prepare_frequency_array(self):
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
    
    # --- Visualization Methods ---
    def plot_solution(self, parameterization: Dict[str, float], **kwargs):

        raw = self.simulate_circuit(parameterization)
        current_complex_response = self.extract_circuit_response_from_latest_run()
        self.prepare_frequency_array()
        
        loss, fit_summary = self.compute_fitness({"current_complex_response" : current_complex_response})

        logger.info(f"total loss: {loss}")
        logger.info(f"mag_loss {fit_summary['mag_loss']}, phase_loss {fit_summary['phase_loss']}")

        plot_complex_response(
            frequencies=self.frequency_array if self.frequency_array is not None else torch.tensor([]), 
            complex_response_list=[self.target_complex_response, current_complex_response], 
            labels=['Target', 'Optimized']
            )
        
# ------------------------------------------------
<<<<<<< HEAD
# A.2 [CONCRETE] Constraint Satisfaction
=======
# A.3 [CONCRETE] Constraint Satisfaction
>>>>>>> origin/main
# ------------------------------------------------
class Nevergrad_Spice_Multi_Spec_Constraint_Satisfaction(Nevergrad_Spice_Base_Optimizer):
    """ Nevergrad Optimizer that uses the perfomance metrics computed in SPICE simulations to size a circuit. """
    def __init__(self,
                 setup_obj: Project_Setup,
                 spicelib_wrapper : Spicelib_Wrapper):
        super().__init__(setup_obj = setup_obj, spicelib_wrapper = spicelib_wrapper)
        self.target_specs: ListTargetSpec = setup_obj.optimizer_config.target_specs
        logger.info(f"Initialized the Nevergrad_Spice_Multi_Spec_Optimizer with {len(self.target_specs.targets)} target specs")
<<<<<<< HEAD
=======
    
    # --- Overwriting the Abstract Methods ---
    def evaluate(self, parameterization: Dict[str, float]) -> Tuple[np.float64, Dict[str, Any]]:
        """
        Evaluate the given parameterization by running a SPICE simulation,
        computing the fitness loss, and returning it as np.float64 plus a metadata dictionary.
        """
        # 1 - Run a SPICE simulation
        # ---------------------------------------------------------------
        raw = self.simulate_circuit(parameterization=parameterization)

        # 2 - Extract performance metrics
        # ---------------------------------------------------------------
        self.spicelib_wrapper.load_raw(raw) # [Redundant but safe]
        # have to make sure to use the correct plot type
        performance_array = self.spicelib_wrapper.extract_scalar_variable_from_raw(self.target_specs.list_target_names())

        # 3 - Compute the fitness of the performance metrics
        # ---------------------------------------------------------------
        metric_value, fit_summary = self.compute_fitness(performance_array=performance_array)

        # --- Log results ---
        
        self.optimization_log.append(OptimizationLogEntry(
            metric_value=metric_value, 
            fit_summary=fit_summary, 
            params=parameterization, 
            log=self.spicelib_wrapper.curr_log
            ))

        logger.debug(f"finished the trial evaluation.... summary")
        logger.debug(f"\tmetric_value = {metric_value}")

        return metric_value, fit_summary
    
    def plot_solution(self, parameterization: Dict[str, float], **kwargs):

        raw = self.simulate_circuit(parameterization, save_sim_override=True)
        self.spicelib_wrapper.load_raw(raw) # [Redundant but safe]
        
        performance_array = self.spicelib_wrapper.extract_scalar_variable_from_raw(self.target_specs.list_target_names())
        loss, fit_summary = self.compute_fitness(performance_array=performance_array)

        logger.info(f"total loss: {loss}")
        for spec_name, spec_info in fit_summary.items():
            logger.info(f"\tSpec '{spec_name}': curr_val={spec_info['curr_val']}, loss={spec_info['loss']}")

        if kwargs.get("show_plot", False):
            trace_name = kwargs.get("trace_name", None)
            if trace_name is None:
                logger.error("To plot the solution, trace_name must be provided in kwargs")
                raise RuntimeError("To plot the solution, trace_name must be provided in kwargs")
            
            trace = self.spicelib_wrapper.extract_wave(trace_name, is_real=False)
            
            plot_complex_response(
                frequencies=self.spicelib_wrapper.extract_wave("frequency", is_real=True),
                complex_response_list=[trace],
                labels = kwargs.get("labels", [trace_name]), 
                title  = kwargs.get("title", f"Response: {trace_name}")
                )

    def compute_fitness(self, performance_array: Dict[str, np.float64 | torch.Tensor]) -> Tuple[np.float64, Dict[str, Any]]:
        """ Compute the fitness based on the performance metrics extracted from SPICE simulations and the target specs. """
        # Initialize variables
        fitness : np.float64 = np.float64(0.0)
        fit_summary : Dict[str, Any] = {}

        # Iterate over each target specification
        for spec in self.target_specs.targets:
            spec_loss: np.float64 = np.float64(0.0)
            # a - Compute the spec loss
            if spec.name in performance_array and performance_array[spec.name] is not None and not np.isnan(performance_array[spec.name]):
                loss = self.compute_spec_loss(curr_val=performance_array[spec.name], target_spec=spec)
                spec_loss = np.clip(loss, None, MAX_LOSS) # cap the loss to avoid overflow
            else:
                if self.verbose:
                    logger.debug(f"Target spec name '{spec.name}' not found in performance array keys: {list(performance_array.keys())}")
                    logger.debug(f"assigning large loss to the {spec.name} spec")
                spec_loss = np.float64(MAX_LOSS) # assign a large loss if the spec is not found
            # b - Log the spec loss
            fit_summary[spec.name] = {
                "curr_val": performance_array.get(spec.name, np.nan) ,
                "loss": spec_loss
            }
            # c - Update the overall fitness (if enabled)
            if spec.enable:
                fitness += spec_loss
            
        logger.debug(f"Computed fitness: {fitness} for performance array: {performance_array}")
        return fitness, fit_summary

    # --- Helper Methods (only in child class) ---
    def compute_spec_loss(self, curr_val: np.float64 | float, target_spec: TargetSpec) -> np.float64:
        """ Compute the loss for a single performance specification. """
        spec_loss:           np.float64 = np.float64(0.0)
        spec_loss_weighted:  np.float64 = np.float64(0.0)

        spec_curr_val: np.float64 = np.float64(curr_val)
        target_val: np.float64 = np.float64(target_spec.target)
        tolerance:  np.float64 = np.float64(target_spec.tolerance)

        if target_spec.log_scale:
            spec_curr_val = np.float64(convert_linear_to_log(curr_val))
            target_val    = np.float64(convert_linear_to_log(target_val))
            tolerance     = np.float64(convert_linear_to_log(tolerance))

        normalizing_coeff = np.float64(target_spec.range)
        # --------------------------
        # Case 1: Exact Match
        # --------------------------
        adjusted_target = target_val - tolerance if spec_curr_val < target_val else target_val + tolerance
        if target_spec.goal == OptimizationGoalType.EXACT:
            if abs(spec_curr_val - target_val) > tolerance:
                spec_loss = compute_error(curr_val=spec_curr_val, target_val=adjusted_target, error_type=target_spec.error_type, normalizing_coeff=normalizing_coeff)
            else:
                spec_loss = np.float64(0.0)
        # --------------------------
        # Case 2: Exceed the Target
        # --------------------------
        elif target_spec.goal == OptimizationGoalType.EXCEED:
            if spec_curr_val < target_val - tolerance:
                spec_loss = compute_error(curr_val=spec_curr_val, target_val=adjusted_target, error_type=target_spec.error_type, normalizing_coeff=normalizing_coeff)
            elif spec_curr_val > target_val + tolerance:
                spec_loss = np.float64(0.0) # optional... could award negative loss for exceeding the target
        # --------------------------
        # Case 3: Minimize the Target
        # --------------------------
        elif target_spec.goal == OptimizationGoalType.MINIMIZE:
            if spec_curr_val > target_val + tolerance:
                spec_loss = compute_error(curr_val=spec_curr_val, target_val=adjusted_target, error_type=target_spec.error_type, normalizing_coeff=normalizing_coeff)
            else:
                spec_loss = np.float64(0.0) 

        # --------------------------
        # Case 4: Invalid Goal Type 
        # --------------------------
        else:
            logger.error(f"Unknown optimization goal type: {target_spec.goal}")
            raise ValueError(f"Unknown optimization goal type: {target_spec.goal}")
        # --------------------------
        
        spec_loss_weighted = spec_loss * np.float64(target_spec.weight)
        logger.debug(f"Spec '{target_spec.name}': curr_val={curr_val}, target={target_spec.target}, loss={spec_loss}, weighted_loss={spec_loss_weighted} - (goal={target_spec.goal})")
        return spec_loss_weighted

# ------------------------------------------------
# A.4 [CONCRETE] Single-objective
# ------------------------------------------------
class Nevergrad_Spice_Single_Objective(Nevergrad_Spice_Multi_Spec_Constraint_Satisfaction):
    def __init__(self,
                setup_obj: Project_Setup,
                spicelib_wrapper : Spicelib_Wrapper):
        super().__init__(setup_obj = setup_obj, spicelib_wrapper = spicelib_wrapper)

# ------------------------------------------------

# ------------------------------------------------
# B [CONCRETE] Symbolic Optimizers (Legacy)
# ------------------------------------------------
class Nevergrad_Symbolic_Bode_Fitter:
    def __init__(self, 
                 tf_to_size: sp.Expr,
                 target_tf: sp.Expr,
                 c_range: List[float] = [1e-12, 1e-9], 
                 r_range: List[float] = [1e2, 1e5],
                 frequencies: torch.Tensor = torch.logspace(3, 8, 1000),
                 freq_weights: Optional[torch.Tensor] = None,
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
>>>>>>> origin/main
    
    # --- Overwriting the Abstract Methods ---
    def evaluate(self, parameterization: Dict[str, float]) -> Tuple[np.float64, Dict[str, Any]]:
        """
        Evaluate the given parameterization by running a SPICE simulation,
        computing the fitness score, and returning it as np.float64 plus a metadata dictionary.
        """
        # 1 - Run a SPICE simulation
        # ---------------------------------------------------------------
        raw = self.simulate_circuit(parameterization=parameterization)

        # 2 - Extract performance metrics
        # ---------------------------------------------------------------
        self.spicelib_wrapper.load_raw(raw) # [Redundant but safe]
        # have to make sure to use the correct plot type
        performance_array = self.spicelib_wrapper.extract_scalar_variable_from_raw(self.target_specs.list_target_names())

        # 3 - Compute the fitness of the performance metrics
        # ---------------------------------------------------------------
        fitness_score, fit_summary = self.compute_fitness(performance_array=performance_array)

        # --- Log results ---
        
        self.optimization_log.append(OptimizationLogEntry(
            score=fitness_score, 
            fit_summary=fit_summary, 
            params=parameterization, 
            log=self.spicelib_wrapper.curr_log
            ))

        logger.debug(f"finished the trial evaluation.... summary")
        logger.debug(f"\tmetric_value = {fitness_score}")

        return fitness_score, fit_summary
    
    def plot_solution(self, parameterization: Dict[str, float], **kwargs):

        raw = self.simulate_circuit(parameterization, save_sim_override=True)
        self.spicelib_wrapper.load_raw(raw) # [Redundant but safe]
        
        performance_array = self.spicelib_wrapper.extract_scalar_variable_from_raw(self.target_specs.list_target_names())
        score, fit_summary = self.compute_fitness(performance_array=performance_array)

        logger.info(f"total score: {score}")
        for spec_name, spec_info in fit_summary.items():
            logger.info(f"\tSpec '{spec_name}': curr_val={spec_info['curr_val']}, score={spec_info['score']}")

        if kwargs.get("show_plot", False):
            trace_name = kwargs.get("trace_name", None)
            if trace_name is None:
                logger.error("To plot the solution, trace_name must be provided in kwargs")
                raise RuntimeError("To plot the solution, trace_name must be provided in kwargs")
            
            trace = self.spicelib_wrapper.extract_wave(trace_name, is_real=False)
            
            plot_complex_response(
                frequencies=self.spicelib_wrapper.extract_wave("frequency", is_real=True),
                complex_response_list=[trace],
                labels = kwargs.get("labels", [trace_name]), 
                title  = kwargs.get("title", f"Response: {trace_name}")
                )

    def compute_fitness(self, performance_array: Dict[str, np.float64 | torch.Tensor]) -> Tuple[np.float64, Dict[str, Any]]:
        """ Compute the fitness based on the performance metrics extracted from SPICE simulations and the target specs. """
        # Initialize variables
        reward      : np.float64 = np.float64(0.0)
        penalty     : np.float64 = np.float64(0.0)
        total_score : np.float64 = np.float64(0.0)
        fit_summary : Dict[str, Any] = {}

        # Iterate over each target specification
        # ------------------------------------------------------------------------------
        for spec in self.target_specs.targets:
            spec_fitness: np.float64 = np.float64(0.0)
            # a - Compute the spec score
            if spec.name in performance_array and performance_array[spec.name] is not None and not np.isnan(performance_array[spec.name]):
                spec_fitness = self.compute_fitness_for_spec(curr_val=performance_array[spec.name], target_spec=spec)
                spec_fitness = np.clip(spec_fitness, -1 * MAX_PENALTY, MAX_REWARD) # cap the score to avoid overflow
            else:
                if self.verbose:
                    logger.debug(f"Target spec name '{spec.name}' not found in performance array keys: {list(performance_array.keys())}")
                    logger.debug(f"assigning large penalty to the {spec.name} spec")
                spec_fitness = -1 * np.float64(MAX_PENALTY) # assign a large score if the spec is not found
            # b - Log the spec score
            fit_summary[spec.name] = {
                "curr_val": performance_array.get(spec.name, np.nan) ,
                "score": spec_fitness
            }
            # c - Update the overall fitness (if enabled)
            if spec.enable:
                if spec_fitness > 0:    reward  += spec_fitness
                else:                   penalty += spec_fitness
        # ------------------------------------------------------------------------------
        
        total_score = reward if penalty > -1*EPSILON else penalty

        logger.debug(f"Computed fitness: {total_score} for performance array: {performance_array}")
        logger.debug(f"\tReward: {reward}")
        logger.debug(f"\tPenalty: {penalty}")
        return total_score, fit_summary

    def compute_fitness_for_spec(self, curr_val: np.float64 | float, target_spec: TargetSpec) -> np.float64:
        """Computes the fitness score for current achieved metric given the target spec. Negative values """
        score = np.float64(0.0)
        # (1) Only return the constraint satisfaction score.
        score += -1 * self.compute_constraint_violation_penalty_for_spec(curr_val=curr_val, target_spec=target_spec)
        return score
    
    # --- Helper Methods (only in this child class) ---
    def compute_constraint_violation_penalty_for_spec(self, curr_val: np.float64 | float, target_spec: TargetSpec) -> np.float64:
        """ Compute a non-negative value representing the penalty for constraint violation. If zero is returned, the constraint is satisfied."""
        spec_penalty:           np.float64 = np.float64(0.0)
        spec_penalty_weighted:  np.float64 = np.float64(0.0)

        spec_curr_val: np.float64 = np.float64(curr_val)
        target_val: np.float64 = np.float64(target_spec.target)
        tolerance:  np.float64 = np.float64(target_spec.tolerance)

        if target_spec.log_scale:
            spec_curr_val = np.float64(convert_linear_to_log(curr_val))
            target_val    = np.float64(convert_linear_to_log(target_val))
            tolerance     = np.float64(convert_linear_to_log(tolerance))

        normalizing_coeff = np.float64(target_spec.range)
        # --------------------------
        # Case 1: Exact Match
        # --------------------------
        adjusted_target = target_val - tolerance if spec_curr_val < target_val else target_val + tolerance
        if target_spec.goal == OptimizationGoalType.EXACT:
            if abs(spec_curr_val - target_val) > tolerance:
                spec_penalty = compute_error(curr_val=spec_curr_val, target_val=adjusted_target, error_type=target_spec.error_type, normalizing_coeff=normalizing_coeff)
            else:
                spec_penalty = np.float64(0.0)
        # --------------------------
        # Case 2: Exceed the Target
        # --------------------------
        elif target_spec.goal == OptimizationGoalType.EXCEED:
            if spec_curr_val < target_val - tolerance:
                spec_penalty = compute_error(curr_val=spec_curr_val, target_val=adjusted_target, error_type=target_spec.error_type, normalizing_coeff=normalizing_coeff)
            elif spec_curr_val > target_val + tolerance:
                spec_penalty = np.float64(0.0)
        # --------------------------
        # Case 3: Minimize the Target
        # --------------------------
        elif target_spec.goal == OptimizationGoalType.MINIMIZE:
            if spec_curr_val > target_val + tolerance:
                spec_penalty = compute_error(curr_val=spec_curr_val, target_val=adjusted_target, error_type=target_spec.error_type, normalizing_coeff=normalizing_coeff)
            else:
                spec_penalty = np.float64(0.0) 

        # --------------------------
        # Case 4: Invalid Goal Type 
        # --------------------------
        else:
            logger.error(f"Unknown optimization goal type: {target_spec.goal}")
            raise ValueError(f"Unknown optimization goal type: {target_spec.goal}")
        # --------------------------
        
        spec_penalty_weighted = spec_penalty * np.float64(target_spec.weight)
        logger.debug(f"Computed Penalty - Spec '{target_spec.name}': curr_val={curr_val}, target={target_spec.target}, penalty={spec_penalty}, weighted_penalty={spec_penalty_weighted} - (goal={target_spec.goal})")
        return spec_penalty_weighted

# ------------------------------------------------
# A.3 [CONCRETE] Single-objective
# ------------------------------------------------
class Nevergrad_Spice_Single_Objective(Nevergrad_Spice_Multi_Spec_Constraint_Satisfaction):
    def __init__(self,
                setup_obj: Project_Setup,
                spicelib_wrapper : Spicelib_Wrapper):
        super().__init__(setup_obj = setup_obj, spicelib_wrapper = spicelib_wrapper)
    
    def compute_fitness_for_spec(self, curr_val: np.float64 | float, target_spec: TargetSpec) -> np.float64:
        """Computes the fitness score for current achieved metric given the target spec. Negative values """
        score = np.float64(0.0)
        # (1) Only return the constraint satisfaction score.
        score += -1 * self.compute_constraint_violation_penalty_for_spec(curr_val=curr_val, target_spec=target_spec)
        score +=      self.compute_reward_for_spec(curr_val=curr_val, target_spec=target_spec)
        return score

    # --- Helper Methods (only in this child class) ---
    def compute_reward_for_spec(self, curr_val: np.float64 | float, target_spec: TargetSpec) -> np.float64:
        """ Compute a non-negative value representing the reward."""
        spec_reward:           np.float64 = np.float64(0.0)
        spec_reward_weighted:  np.float64 = np.float64(0.0)

        spec_curr_val: np.float64 = np.float64(curr_val)
        target_val: np.float64 = np.float64(target_spec.target)
        tolerance:  np.float64 = np.float64(target_spec.tolerance)

        if target_spec.log_scale:
            spec_curr_val = np.float64(convert_linear_to_log(curr_val))
            target_val    = np.float64(convert_linear_to_log(target_val))
            tolerance     = np.float64(convert_linear_to_log(tolerance))

        normalizing_coeff = np.float64(target_spec.range)
        adjusted_target = target_val - tolerance if spec_curr_val < target_val else target_val + tolerance
        # --------------------------
        # Case 1: Exceed the Target
        # --------------------------
        if target_spec.goal == OptimizationGoalType.EXCEED:
            if spec_curr_val < target_val - tolerance:
                spec_reward = compute_reward(curr_val=spec_curr_val, target_val=adjusted_target, error_type=target_spec.error_type, normalizing_coeff=normalizing_coeff)
            elif spec_curr_val > target_val + tolerance:
                spec_reward = np.float64(0.0)
        # --------------------------
        # Case 2: Minimize the Target
        # --------------------------
        elif target_spec.goal == OptimizationGoalType.MINIMIZE:
            if spec_curr_val > target_val + tolerance:
                spec_reward = compute_reward(curr_val=spec_curr_val, target_val=adjusted_target, error_type=target_spec.error_type, normalizing_coeff=normalizing_coeff)
            else:
                spec_reward = np.float64(0.0) 

        # --------------------------
        # Case 3: Invalid Goal Type 
        # --------------------------
        else:
            logger.error(f"Unknown optimization goal type: {target_spec.goal}")
            raise ValueError(f"Unknown optimization goal type: {target_spec.goal}")
        # --------------------------
        
<<<<<<< HEAD
        spec_reward_weighted = spec_reward * np.float64(target_spec.weight)
        logger.debug(f"Computed Penalty - Spec '{target_spec.name}': curr_val={curr_val}, target={target_spec.target}, penalty={spec_reward}, weighted_penalty={spec_reward_weighted} - (goal={target_spec.goal})")
        return spec_reward_weighted
=======
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
>>>>>>> origin/main
# ------------------------------------------------
