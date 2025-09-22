import yaml
import numpy as np
import logging

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger("SymXplorer.domains")

# ------------------ Helpers ------------------

def parse_value(val: Union[str, float, int]) -> np.float64:
    """
    Parse a string like '0.18u', '10u', '1.8', or a number into np.float64.
    Supports 'u' (micro), 'n' (nano), 'p' (pico), 'k', 'M' suffixes.
    """
    if isinstance(val, (float, int)):
        return np.float64(val)
    if val.lower() == "inf":
        return np.float64(np.inf)
    
    val = val.strip()
    multipliers = {
        "p": 1e-12,
        "n": 1e-9,
        "u": 1e-6,
        "m": 1e-3,
        "k": 1e3,
        "M": 1e6,
        "G": 1e9,
    }
    for suffix, factor in multipliers.items():
        if val.endswith(suffix):
            return np.float64(float(val[:-1]) * factor)
    return np.float64(float(val))


def resolve_reference(value: Union[str, float, int], constraints: Dict[str, np.float64]) -> np.float64:
    """If value is a reference (string key), replace it with its constraint value."""
    if isinstance(value, str) and value in constraints:
        return constraints[value]
    return parse_value(value)


# ---------- Core Dataclasses ----------

@dataclass
class TechSpec:
    name: str
    constraints: Dict[str, np.float64] = field(default_factory=dict)


@dataclass
class PVT:
    temp:   np.float64
    corner: str
    supply: np.float64


@dataclass
class Param:
    name: str
    min_val: Optional[np.float64] = None
    max_val: Optional[np.float64] = None
    log_scale: bool = False
    default: Optional[np.float64] = None

    def compute_lin_normalization(self, denorm_val: np.float64) -> np.float64:
        if self.max_val is None or self.min_val is None:
            raise ValueError("there is either no min or max value defined for this parameter")
        return denorm_val * (self.max_val - self.min_val) + self.min_val
    
    def compute_log_normalization(self, denorm_val: np.float64) -> np.float64:
        if self.max_val is None or self.min_val is None:
            raise ValueError("there is either no min or max value defined for this parameter")
        return denorm_val * (self.max_val - self.min_val) + self.min_val


@dataclass
class DutParams:
    params: List[Param]


@dataclass
class TestbenchParams:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Probes:
    name: str
    node: str
    type: str

@dataclass
class LossFunctionConfig:
    max_loss: np.float64
    loss_norm_method: str
    loss_type: str
    rescale_mag: bool
    include_phase_loss : bool
    include_mag_loss : bool

@dataclass
class VariableBoundConfig:
    min: float
    max: float


@dataclass
class OptimizerConfig:
    name: str
    type: str
    budget: int
    loss_function: LossFunctionConfig
    lin_variable_bounds: VariableBoundConfig
    log_variable_bounds: VariableBoundConfig
    random_seed: Optional[int] = None
# ---------- Interface Dataclass ----------

@dataclass
class Project_Setup:
    project_name: str
    description: str
    simulator:  str
    ws_root :   Path
    netlist:    Path
    outdir :    Path
    # Custom Data types
    tech_spec: TechSpec
    pvt: PVT
    dut_params: List[Param]
    testbench: TestbenchParams
    probes: List[Probes]
    optimizer_config: Optional[OptimizerConfig] = None
    logger: logging.Logger = logger

    # ------------------ Class Methods ------------------

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "Project_Setup":
        """Load a Project object from a YAML file with variable resolution."""
        yaml_path = Path(yaml_path)
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        project_data = data["project"]
        logger.debug("loaded project yaml... creating the project setup hierarchy")

        # --- Parse tech_spec constraints ---
        logger.debug("\tLoading the tech constraints...")
        raw_constraints = project_data["tech_spec"]["constraints"]
        parsed_constraints = {k: parse_value(v) for k, v in raw_constraints.items()}
        tech_spec = TechSpec(name=project_data["tech_spec"]["name"], constraints=parsed_constraints)
        logger.debug(f"\tThe tech constraints are loaded (count {len(tech_spec.constraints)}) for {tech_spec.name}...")

        # --- Parse PVT ---
        logger.debug("\tLoading the PVT Data...")
        pvt_data = project_data["pvt"]
        pvt = PVT(
            temp=parse_value(pvt_data["temp"]),
            corner=pvt_data["corner"],
            supply=parse_value(pvt_data["supply"]),
        )
        logger.debug("\tPVT Data is loaded...")

        # --- Build DUT params with reference resolution ---
        dut_params = []
        logger.debug("\tLoading the DUT params...")
        for p in project_data["params"]["dut"]:
            dut_params.append(
                Param(
                    name=p["name"],
                    min_val=resolve_reference(p.get("min_val"), parsed_constraints) if "min_val" in p else None,
                    max_val=resolve_reference(p.get("max_val"), parsed_constraints) if "max_val" in p else None,
                    log_scale=p.get("log_scale", False),
                    default=resolve_reference(p.get("default"), parsed_constraints) if "default" in p else None,
                )
            )
        logger.debug("\tThe DUT params are loaded...")

        # --- Build Testbench ---
        tb_data = project_data["params"]["testbench"]
        testbench = TestbenchParams(name=tb_data["name"], params=tb_data.get("params", {}))

        # --- Build Probes ---
        probes = [Probes(**p) for p in project_data.get("probes", [])]
        
        # --- Parse Optimizer Config (optional) ---
        optimizer_config = None
        if "optimizer_config" in project_data:
            opt_data = project_data["optimizer_config"]
            loss_fn = LossFunctionConfig(
                max_loss=parse_value(opt_data["loss_function"]["max_loss"]),
                loss_norm_method=opt_data["loss_function"]["loss_norm_method"],
                loss_type=opt_data["loss_function"]["loss_type"],
                rescale_mag=bool(opt_data["loss_function"]["rescale_mag"]),
                include_mag_loss=opt_data["loss_function"]["include_mag_loss"],
                include_phase_loss=opt_data["loss_function"]["include_phase_loss"],
            )
            optimizer_config = OptimizerConfig(
                name=opt_data["name"],
                type=opt_data["type"],
                budget=int(opt_data["budget"]),
                random_seed=opt_data["random_seed"],
                lin_variable_bounds=VariableBoundConfig(**opt_data['lin_variable_bounds']),
                log_variable_bounds=VariableBoundConfig(**opt_data['log_variable_bounds']),
                loss_function=loss_fn
            )

        proj = cls(
            project_name=project_data["name"],
            ws_root = Path(project_data['ws_root']),
            netlist = Path(project_data['netlist']),
            outdir  = Path(project_data['outdir']),
            description = project_data["description"],
            simulator   = project_data["simulator"],
            tech_spec   = tech_spec,
            pvt         = pvt,
            dut_params  = dut_params,
            testbench   = testbench,
            probes      = probes,
            optimizer_config=optimizer_config,
            logger      = logger
        )

        proj.logger.info(f"Loaded project '{proj.project_name}' with {len(proj.dut_params)} DUT params and {len(proj.probes)} probes.")
        return proj

    # ------------------ Getters & Helpers ------------------

    def get_constraint_by_name(self, name: str) -> Optional[np.float64]:
        value = self.tech_spec.constraints.get(name)
        self.logger.debug(f"Constraint '{name}': {value}")
        return value

    def list_constraints(self) -> Dict[str, np.float64]:
        self.logger.debug(f"Listing all constraints: {self.tech_spec.constraints}")
        return self.tech_spec.constraints

    def get_param_by_name(self, name: str) -> Optional[Param]:
        for p in self.dut_params:
            if p.name == name:
                self.logger.debug(f"Found DUT param: {p}")
                return p
        self.logger.warning(f"DUT param '{name}' not found")
        return None

    def list_params(self) -> List[str]:
        param_names = [p.name for p in self.dut_params]
        self.logger.debug(f"DUT param names: {param_names}")
        return param_names

    def get_log_scaled_params(self) -> List[Param]:
        log_params = [p for p in self.dut_params if p.log_scale]
        self.logger.debug(f"Log-scaled params: {[p.name for p in log_params]}")
        return log_params

    def filter_params_by_range(self, min_value: float, max_value: float) -> List[Param]:
        filtered = [p for p in self.dut_params if p.default is not None and min_value <= p.default <= max_value]
        self.logger.debug(f"Params in range {min_value}-{max_value}: {[p.name for p in filtered]}")
        return filtered

    def get_probe_by_name(self, name: str) -> Optional[Probes]:
        for pr in self.probes:
            if pr.name == name:
                self.logger.debug(f"Found probe: {pr}")
                return pr
        self.logger.warning(f"Probe '{name}' not found")
        return None

    def get_probes_by_type(self, probe_type: str) -> List[Probes]:
        probes_of_type = [p for p in self.probes if p.type == probe_type]
        self.logger.debug(f"Probes of type '{probe_type}': {[p.name for p in probes_of_type]}")
        return probes_of_type

    def summary(self) -> None:
        self.logger.info("========== Project Setup Summary ==========")
        self.logger.info(f"📂 Project: {self.project_name}")
        self.logger.info(f"📝 Description: {self.description}")
        self.logger.info(f"🧠 Simulator: {self.simulator}")
        self.logger.info(f"📜 Netlist: {self.netlist}")
        self.logger.info(f"⚙️  PVT: temp={self.pvt.temp}, corner={self.pvt.corner}, supply={self.pvt.supply}")
        self.logger.info(f"🔧 Tech Spec: {len(self.tech_spec.constraints)} constraints")
        for k, v in self.tech_spec.constraints.items():
            self.logger.info(f"   • {k}: {v:.2e}")
        self.logger.info(f"🎛 DUT Params: {len(self.dut_params)} params -> {[p.name for p in self.dut_params]}")
        self.logger.info(f"🔍 Probes: {[p.name for p in self.probes]}")
        self.logger.info("===========================================")
