import yaml
import numpy as np
import logging
from datetime import datetime

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any, Iterator, Tuple
from pathlib import Path
from enum import Enum

from dacite import from_dict, Config
from dacite.exceptions import DaciteError, MissingValueError, WrongTypeError, UnexpectedDataError

# ------------------ Module Logger ------------------

logger = logging.getLogger("SymXplorer.domains")

# ------------------ Enums ------------------

class SimType(str, Enum):
    DC = "dc"
    AC = "ac"
    TRAN = "tran"
    NOISE = "noise"

class OptimizationGoalType(str, Enum):
    EXACT    = "exact"
    EXCEED   = "exceed"
    MINIMIZE = "minimize"

class OptimizerType(str, Enum):
    NEVERGRAD = "nevergrad"

class SpicePlotType(str, Enum):
    OP = "Operating Point"
    TRANSIENT = "tran"
    AC_MAG = "ac_mag"
    AC_PHASE = "ac_phase"
    DC = "dc"
    NOISE = "noise"

class Error_Types(str, Enum):
    ABSOLUTE = "absolute"
    SQUARED  = "squared"
    EXPONENTIAL = "exponential"
    RELATIVE_ABSOLUTE = "relative-absolute"
    RELATIVE_SQUARED  = "relative-squared"
    RELATIVE_EXPONENTIAL = "relative-exponential"
    RELATIVE_SIGMOID = "relative-sigmoid"

    def is_relative(self) -> bool:
        return "relative" in self.value

class Reward_Types(str, Enum):
    ABSOLUTE = "absolute"
    RELATIVE_ABSOLUTE = "relative-absolute"
    RELATIVE_SIGMOID = "relative-sigmoid"

    def is_relative(self) -> bool:
        return "relative" in self.value

# ------------------ Constants ------------------

MULTIPLIERS = {
        "p": 1e-12,
        "n": 1e-9,
        "u": 1e-6,
        "m": 1e-3,
        "k": 1e3,
        "M": 1e6,
        "G": 1e9,
    }
# ------------------ Helpers ------------------

def parse_value(val: Union[str, float, int]) -> np.float64:
    """
    Parse numeric values with optional suffix multipliers (u, n, p, k, M, G).
    Parses a string like '0.18u', '10u', '1.8', or a number into np.float64.
    Supports 'u' (micro), 'n' (nano), 'p' (pico), 'k', 'M' suffixes.
    """
    if isinstance(val, (float, int)):
        return np.float64(val)
    if val.lower() == "inf":
        return np.float64(np.inf)
    
    val = val.strip()

    for suffix, factor in MULTIPLIERS.items():
        if val.endswith(suffix):
            return np.float64(float(val[:-1]) * factor)
    return np.float64(float(val))

def resolve_reference(value: Union[str, float, int], constraints: Dict[str, np.float64 | float]) -> np.float64:
    """If value is a reference to a constraint key, resolve it, else parse normally."""
    if isinstance(value, str) and value in constraints:
        return np.float64(constraints[value])
    return parse_value(value)

def safe_from_dict(cls, data: dict, logger: logging.Logger, config: Config = Config(cast=[Enum])):
    try:
        return from_dict(data_class=cls, data=data, config=config)
    except MissingValueError as e:
        logger.critical(f"❌ Missing required field in {cls.__name__}: {e}")
        raise
    except WrongTypeError as e:
        logger.critical(f"❌ Wrong type while parsing {cls.__name__}: {e}")
        raise
    except UnexpectedDataError as e:
        logger.critical(f"❌ Unexpected field while parsing {cls.__name__}: {e}")
        raise

def list_target_spec_hook(data: list) -> 'ListTargetSpec':
    return ListTargetSpec([TargetSpec(**item) for item in data])

# ---------- Core Dataclasses ----------

@dataclass
class TechSpec:
    """Process technology (PDK) specification with constraints on device parameters."""
    name: str
    constraints: Dict[str, np.float64 | float | str] = field(default_factory=dict)

    def __post_init__(self):
        for key, val in self.constraints.items():
            if isinstance(val, str):
                self.constraints[key] = parse_value(val)
                logger.debug(f"Parsed constraint '{key}': '{val}' to {self.constraints[key]}")

@dataclass
class PVT:
    temp:   float
    corner: str
    supply: float

@dataclass
class Param:
    name: str
    min_val: Optional[Union[float, np.float64, str]]
    max_val: Optional[Union[float, np.float64, str]]
    val: Optional[Union[float, np.float64, str]]
    description: Optional[str]
    log_scale: bool = False

    def needs_resolution(self) -> bool:
        return isinstance(self.min_val, str) or isinstance(self.max_val, str) or (self.val is not None and isinstance(self.val, str))

    def resolve_min_max(self, constraints: Dict[str, np.float64]) -> None:
        if self.min_val is None or self.max_val is None:
            raise ValueError(f"Param {self.name} missing min or max value for resolution")
        self.min_val = resolve_reference(self.min_val, constraints)
        self.max_val = resolve_reference(self.max_val, constraints)
        if self.val is not None:
            self.val = resolve_reference(self.val, constraints)
        if self.min_val >= self.max_val:
            raise ValueError(f"Param {self.name} has min_val >= max_val ({self.min_val} >= {self.max_val})")

    def compute_lin_normalization(self, denorm_val: np.float64) -> np.float64:
        if self.needs_resolution():
            raise ValueError(f"Param {self.name} min/max not resolved before normalization")
        if self.max_val is None or self.min_val is None:
            raise ValueError(f"No min/max defined for parameter {self.name}")
        return denorm_val * (self.max_val - self.min_val) + self.min_val

    def compute_log_normalization(self, denorm_val: np.float64) -> np.float64:
        if self.needs_resolution():
            raise ValueError(f"Param {self.name} min/max not resolved before normalization")
        if self.max_val is None or self.min_val is None:
            raise ValueError(f"No min/max defined for log-normalization of {self.name}")
        log_min, log_max = np.log(self.min_val), np.log(self.max_val)
        return np.exp(denorm_val * (log_max - log_min) + log_min)

@dataclass
class DutParams:
    params: List[Param]

@dataclass
class TestbenchParams:
    name: str
    params: List[Param]

@dataclass
class TargetSpec:
    name: str
    target: float | np.float64
    goal: Union[OptimizationGoalType, str]
    sim_type: Union[SimType, str]
    log_scale: bool = False
    enable: bool = True
    range: Union[np.float64, float, str | None] = None
    error_type: Union[Error_Types, str] = Error_Types.RELATIVE_ABSOLUTE
    weight: Optional[float | np.float64] = 1.0
    tolerance: Optional[float | np.float64] = None  # if not given use 5% of target
    description: Optional[str] = None

    def __post_init__(self):
        # Prepare human-friendly lists for error messages
        valid_goals = [g.value for g in OptimizationGoalType]
        valid_sim_types = [s.value for s in SimType]

        # --- Validate / convert goal ---
        if isinstance(self.goal, str):
            try:
                self.goal = OptimizationGoalType(self.goal.lower())
            except ValueError:
                logger.critical(
                    f"Invalid goal '{self.goal}' for target '{self.name}'. "
                    f"Must be one of {valid_goals}."
                )
                raise ValueError(f"Invalid goal '{self.goal}'. Must be one of {valid_goals}.")
        elif not isinstance(self.goal, OptimizationGoalType):
            logger.critical(
                f"Invalid goal type '{type(self.goal)}' for target '{self.name}'. "
                f"Must be one of {valid_goals}."
            )
            raise ValueError(f"Invalid goal '{self.goal}'. Must be one of {valid_goals}.")

        # --- Validate / convert sim_type ---
        if isinstance(self.sim_type, str):
            try:
                self.sim_type = SimType(self.sim_type.lower())
            except ValueError:
                logger.critical(
                    f"Invalid sim_type '{self.sim_type}' for target '{self.name}'. "
                    f"Must be one of {valid_sim_types}."
                )
                raise ValueError(f"Invalid sim_type '{self.sim_type}'. Must be one of {valid_sim_types}.")
        elif not isinstance(self.sim_type, SimType):
            logger.critical(
                f"Invalid sim_type type '{type(self.sim_type)}' for target '{self.name}'. "
                f"Must be one of {valid_sim_types}."
            )
            raise ValueError(f"Invalid sim_type '{self.sim_type}'. Must be one of {valid_sim_types}.")

        # --- Validate / convert error_type ---
        if isinstance(self.error_type, str):
            try:
                self.error_type = Error_Types(self.error_type.lower())
            except ValueError:
                valid_errors = [e.value for e in Error_Types]
                logger.critical(
                    f"Invalid error_type '{self.error_type}' for target '{self.name}'. "
                    f"Must be one of {valid_errors}."
                )
                raise ValueError(f"Invalid error_type '{self.error_type}'. Must be one of {valid_errors}.")
        
        # --- Validate / convert range ---
        self.range = np.float64(self.range)

        # --- Tolerance fallback ---
        if isinstance(self.tolerance, str):
            self.tolerance = parse_value(self.tolerance)

        if self.tolerance is None or not(self.tolerance > 0):
            self.tolerance = abs(0.05 * self.target)
            logger.warning(
                f"No valid tolerance specified for target '{self.name}'. "
                f"Using default tolerance of 5%: {self.tolerance}"
            )

        # --- Initialization log ---
        logger.debug(
            f"Initialized TargetSpec: {self.name}, target={self.target}, "
            f"tolerance={self.tolerance}, goal={self.goal}, sim_type={self.sim_type}, enable={self.enable}"
        )

    def get_simple_penalty(self, value: np.float64) -> np.float64:
        """Compute a simple penalty based on the goal and tolerance. Will allow reward in the form of negative penalty."""
        if not self.enable:
            return np.float64(0.0)
        
        if self.tolerance is None:
            logger.error(f"Something went wrong and tolerance is None for target '{self.name}'")
            raise RuntimeError("Tolerance should never be None here... check the log.")

        if self.goal == OptimizationGoalType.EXACT:
            if np.abs(value - self.target) <= self.tolerance:
                return np.float64(0.0)
            else:
                return np.abs(value - self.target) - self.tolerance
        
        elif self.goal == OptimizationGoalType.EXCEED:
            if value >= self.target - self.tolerance:
                return np.float64(0.0)
            else:
                return np.float64(self.target - self.tolerance - value)
        
        elif self.goal == OptimizationGoalType.MINIMIZE:
            if value <= self.target + self.tolerance:
                return np.float64(0.0)
            else:
                return np.float64(value - (self.target + self.tolerance))
        
        else:
            logger.error(f"Unknown goal type '{self.goal}' for target '{self.name}'")
            raise ValueError(f"Unknown goal type '{self.goal}'")
        
    def meets_spec(self, value: np.float64) -> bool:
        """Check if the given value meets the specification."""
        penalty = self.get_simple_penalty(value)
        return not (penalty > np.float64(0.0))
    
    def __str__(self) -> str:
        return (
            f"TargetSpec(name={self.name}, target={self.target}, range={self.range:.2e} "
            f"tolerance={self.tolerance}, goal={self.goal.value}, sim_type={self.sim_type.value}, enable={self.enable}, "
            f"error_type={self.error_type.value}, weight={self.weight}, enable={self.enable}, description={self.description})"
        )

@dataclass
class ListTargetSpec:
    targets: List[TargetSpec] = field(default_factory=list)

    def add_target(self, target: TargetSpec) -> None:
        logger.info(f"Adding target '{target.name}' to ListTargetSpec")
        self.targets.append(target)

    def get_target_by_name(self, name: str) -> Optional[TargetSpec]:
        for t in self.targets:
            if t.name == name:
                return t
        return None
    
    def list_target_names(self) -> List[str]:
        return [t.name for t in self.targets]
    
    def enabled_targets(self) -> List[TargetSpec]:
        return [t for t in self.targets if t.enable]

@dataclass
class LossFunctionConfig:
    max_loss: Union[np.float64, str]
    loss_norm_method: Optional[str]
    loss_type: Optional[str]
    rescale_mag: Optional[bool] = False
    include_phase_loss : Optional[bool] = False
    include_mag_loss : Optional[bool] = False

@dataclass
class VariableBoundConfig:
    min: float
    max: float

    def get_range(self) -> float:
        return self.max - self.min

    def get_min_max(self) -> Tuple[float, float]:
        return (self.min, self.max)

@dataclass
class OptimizerConfig:
    name: str # Optimization algorithm name
    type: str # Optimizer family type
    budget: int
    target_specs: ListTargetSpec
    lin_variable_bounds: Optional[VariableBoundConfig]
    log_variable_bounds: Optional[VariableBoundConfig]
    loss_function_config: Optional[LossFunctionConfig]
    random_seed: Optional[int]

    def __post_init__(self):
        # Mandatory checks
        if not self.name or not self.type or self.budget is None:
            logger.critical("OptimizerConfig is missing a mandatory field (name, type, or budget).")
            raise ValueError("OptimizerConfig requires name, type, and budget.")

        # -------------------------
        # General
        # -------------------------
        logger.info(
            f"Initialized OptimizerConfig: {self.name}, "
            f"type={self.type}, budget={self.budget}, random_seed={self.random_seed}"
        )

        # -------------------------
        # Bounds
        # -------------------------
        if self.lin_variable_bounds is None:
            logger.warning("No lin_variable_bounds provided; using default [0.0, 1.0].")
            self.lin_variable_bounds = VariableBoundConfig(min=0.0, max=1.0)
        else: 
            logger.info(
                f"\tLinear bounds: min={self.lin_variable_bounds.min}, max={self.lin_variable_bounds.max}"
            )
        if self.log_variable_bounds is None:
            logger.warning("No log_variable_bounds provided; using default [0.0, 1.0].")
            self.log_variable_bounds = VariableBoundConfig(min=1, max=100.0)
        else:
            logger.info(
                f"\tLog bounds: min={self.log_variable_bounds.min}, max={self.log_variable_bounds.max}"
            )

        # -------------------------
        # Loss function config
        # -------------------------
        if self.loss_function_config is None:
            logger.warning("No loss_function_config provided; using default values.")
        else: 
            logger.info(
                f"\tLoss function: max_loss={self.loss_function_config.max_loss}, "
                f"norm_method={self.loss_function_config.loss_norm_method}, "
                f"type={self.loss_function_config.loss_type}, rescale_mag={self.loss_function_config.rescale_mag}, "
                f"include_phase_loss={self.loss_function_config.include_phase_loss}, "
                f"include_mag_loss={self.loss_function_config.include_mag_loss}"
            )

        # -------------------------
        # Target Specs
        # -------------------------
        logger.info(f"\tNumber of target specs: {len(self.target_specs.targets)}")
        for t in self.target_specs.targets:
            logger.info(f"\t\t- {t}")
    
    def get_lin_variable_range(self) -> np.float64:
        if self.lin_variable_bounds is None:
            raise ValueError("Linear variable bounds are not set")
        return np.float64(self.lin_variable_bounds.get_range())

    def get_log_variable_range(self) -> np.float64:
        if self.log_variable_bounds is None:
            raise ValueError("Log variable bounds are not set")
        return np.float64(self.log_variable_bounds.get_range())
    
    def get_lin_min_max(self) -> Tuple[float, float]:
        if self.lin_variable_bounds is None:
            raise ValueError("Linear variable bounds are not set")
        return self.lin_variable_bounds.get_min_max()

    def get_log_min_max(self) -> Tuple[float, float]:
        if self.log_variable_bounds is None:
            raise ValueError("Linear variable bounds are not set")
        return self.log_variable_bounds.get_min_max()
        
# ---------- Interface Dataclass ----------

@dataclass
class Project_Setup:
    # General Info
    name: str
    description: str
    simulator:  str
    ws_root :   Path | str
    netlist:    Path | str
    outdir :    Path | str
    
    # Custom Data types
    tech_spec: TechSpec
    pvt: PVT
    dut_params: List[Param]
    testbench_params: TestbenchParams
    optimizer_config: OptimizerConfig
    
    save_sim:  bool = False

    def __post_init__(self):
        # correct path types
        if isinstance(self.ws_root, str):
            self.ws_root = Path(self.ws_root)
        if isinstance(self.netlist, str):
            self.netlist = Path(self.netlist)
        if isinstance(self.outdir, str):
            self.outdir = Path(self.outdir)
        # Log basic info
        logger.info(f"Project '{self.name}' initialized with simulator '{self.simulator}'")
        logger.info(f"\tWorkspace root: {self.ws_root}")
        logger.info(f"\tNetlist path: {self.netlist}")
        logger.info(f"\tOutput directory: {self.outdir}")

    # ------------------ Class Methods ------------------

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "Project_Setup":
        """Load a Project object from a YAML file with variable resolution."""
        
        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            logger.debug(f"YAML content successfully loaded: {list(data.keys())}")

            project = safe_from_dict(cls, data['project'], logger, config=DECITE_CONFIG)
            
            # Resolve constraints in tech_spec
            project.resolve_all_parameter_ranges()

            logger.info("✅ Project setup successfully created")
            return project

        except FileNotFoundError:
            logger.critical(f"YAML file not found: {yaml_path}")
            raise
        except yaml.YAMLError as e:
            logger.critical(f"Failed to parse YAML {yaml_path}: {e}")
            raise
        except DaciteError as e:
            logger.critical(f"Failed to map YAML → Project_Setup: {e}")
            raise
        except Exception as e:
            logger.critical(f"Unexpected error while loading {yaml_path}: {e}")
            raise

    # ------------------ Getters & Helpers ------------------
    def resolve_all_parameter_ranges(self) -> None:
        """Resolve all parameter min/max/default values based on tech_spec constraints."""
        for param in self.dut_params:
            if param.needs_resolution():
                logger.info(f"Resolving ranges for param '{param.name}'")
                param.resolve_min_max(self.tech_spec.constraints)
                logger.debug(f"Resolved param '{param.name}': min={param.min_val}, max={param.max_val}, default={param.val}")
            
    def get_constraint_by_name(self, name: str) -> Optional[np.float64]:
        value = self.tech_spec.constraints.get(name)
        logger.debug(f"Constraint '{name}': {value}")
        return value

    def list_constraints(self) -> Dict[str, np.float64]:
        logger.debug(f"Listing all constraints: {self.tech_spec.constraints}")
        return self.tech_spec.constraints

    def get_param_by_name(self, name: str) -> Optional[Param]:
        for p in self.dut_params:
            if p.name == name:
                # logger.debug(f"Found DUT param: {p}")
                return p
        logger.warning(f"DUT param '{name}' not found")
        return None

    def list_params(self) -> List[str]:
        param_names = [p.name for p in self.dut_params]
        logger.debug(f"DUT param names: {param_names}")
        return param_names

    def get_log_scaled_params(self) -> List[Param]:
        log_params = [p for p in self.dut_params if p.log_scale]
        logger.debug(f"Log-scaled params: {[p.name for p in log_params]}")
        return log_params

    def filter_params_by_range(self, min_value: float, max_value: float) -> List[Param]:
        filtered = [p for p in self.dut_params if p.val is not None and min_value <= p.val <= max_value]
        logger.debug(f"Params in range {min_value}-{max_value}: {[p.name for p in filtered]}")
        return filtered

    def summary(self) -> None:
        logger.info("========== Project Setup Summary ==========")
        logger.info(f"📂 Project: {self.name}")
        logger.info(f"📝 Description: {self.description}")
        logger.info(f"🧠 Simulator: {self.simulator}")
        logger.info(f"📜 Netlist: {self.netlist}")
        logger.info(f"⚙️  PVT: temp={self.pvt.temp}, corner={self.pvt.corner}, supply={self.pvt.supply}")
        logger.info(f"🔧 Tech Spec: {len(self.tech_spec.constraints)} constraints")
        for k, v in self.tech_spec.constraints.items():
            logger.info(f"   • {k}: {v:.2e}")
        logger.info(f"🎛 DUT Params: {len(self.dut_params)} params -> {[p.name for p in self.dut_params]}")
        logger.info(f"🔍 target specs: {[p.name for p in self.optimizer_config.target_specs.targets]}")
        logger.info("===========================================")

# ------------------ Dacite Config ------------------
DECITE_CONFIG = Config(
    type_hooks={
        ListTargetSpec: list_target_spec_hook
    }
)

# ------------------ Optimizer Objects ------------------
@dataclass
class OptimizationPoint:
    """Represents the simplest point in the optimization trace."""
    params: Dict[str, float | np.float64]
    score: float | np.float64
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict) # to add any other information

@dataclass
class OptimizationLogEntry:
    """Represents a single entry in the optimization log."""
    point: 'OptimizationPoint'           
    fit_summary: Optional[Dict[str, Any]] = field(default_factory=dict)   # Depends on your optimizer output (could refine type)
    log_file: Optional[str | Path] = None                                  # Any log/debug info

    def get_score(self) -> float | np.floating:
        return self.point.score
    
    def get_params(self) -> Dict[str, float | np.floating]:
        return self.point.params
    
    def get_metadata(self) -> Dict[str, Any] | None:
        return self.point.metadata
    
    def get_param_val(self, param_name: str) -> float | np.floating | None:
        if not param_name in self.point.params.keys():
            logger.debug(f"{param_name} was not found in the OptimizationLogEntry object - should be one of {self.point.params.keys()}")
            return None
        return self.point.params[param_name]
    
    def get_fit_summary(self) -> Dict[str, Any]:
        if self.fit_summary is None:
            logger.error(f"tried accessing the fit_summary but this was never created")
            raise ValueError(f"tried accessing the fit_summary but this was never created")
        return self.fit_summary


class OptimizationLog:
    """Acts like a list of OptimizationLogEntry objects."""
    def __init__(self, initial_logs: List[OptimizationLogEntry] = []):
        self.log: List[OptimizationLogEntry] = initial_logs

    def __iter__(self) -> Iterator[OptimizationLogEntry]:
        """Allow iteration over log entries."""
        return iter(self.log)

    def __len__(self) -> int:
        """Return the number of log entries."""
        return len(self.log)

    def __getitem__(self, index: int) -> OptimizationLogEntry:
        """Support indexing like a list."""
        return self.log[index]

    def __setitem__(self, index: int, value: OptimizationLogEntry) -> None:
        """Support assignment by index."""
        self.log[index] = value

    def __delitem__(self, index: int) -> None:
        """Support deletion by index."""
        del self.log[index]

    def append(self, entry: OptimizationLogEntry) -> None:
        """Append a new entry to the log."""
        self.log.append(entry)

    def extend(self, entries: List[OptimizationLogEntry]) -> None:
        """Extend log with multiple entries."""
        self.log.extend(entries)

    def __repr__(self) -> str:
        """Readable representation."""
        return f"OptimizationLog({self.log!r})"
    
    def get_score(self, index: int) -> float | np.floating:
        return self.log[index].get_score()

    def get_params(self, index: int) -> Dict[str, float | np.floating]:
        return self.log[index].get_params()

    def get_metadata(self, index: int) -> Dict[str, Any]:
        return self.log[index].get_metadata()
    
    def has_param(self, param_name: str) -> bool:
        if len(self.log) == 0:
            logger.debug("no log file in the object")
            return False
        if not param_name in self.log[0].get_params():
            logger.debug(f"param '{param_name}' not found in optimization trace")
            return False
        return True
        
        