"""This Module implements the user endpoint for selecting optimizers types based on an input project_setup yaml file and optimization engine type"""
import logging
import numpy        as np

# Third-party imports
from    enum        import Enum
from    typing      import Dict, Tuple, Any, List, Type
from    pathlib     import Path
from    abc         import ABC, abstractmethod

# Symxplorer Specific Imports
from    symxplorer.spice_engine              import Spicelib_Wrapper, Sim_Execution_Type
from    symxplorer.designer_tools.domains    import Project_Setup

from    .base           import Spice_Base_Optimizer, Base_Optimizer
from    .nevergrad      import Nevergrad_Spice_Bode_Optimizer, Nevergrad_Spice_Constraint_Satisfaction,  Nevergrad_Spice_Single_Objective
from    .bayesian_ax    import Ax_Spice_Constraint_Satisfaction, Ax_Spice_Single_Objective

# ------------------ Module Logger ------------------

logger = logging.getLogger("SymXplorer.orchestrator")


# ------------------ Enums ------------------
class Optimizer_Type_Enum(Enum):
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    NEVERGRAD_BODE = "nevergrad_bode"
    NEVERGRAD_CONSTRAINT = "nevergrad_constraint"
    NEVERGRAD_SINGLE = "nevergrad_single"
    AX_CONSTRAINT = "ax_constraint"
    AX_SINGLE = "ax_single"

SPICE_OPTIMIZER_CLASSES : Dict[Optimizer_Type_Enum, Type[Spice_Base_Optimizer]] = {
    Optimizer_Type_Enum.NEVERGRAD_BODE: Nevergrad_Spice_Bode_Optimizer,
    Optimizer_Type_Enum.NEVERGRAD_CONSTRAINT: Nevergrad_Spice_Constraint_Satisfaction,
    Optimizer_Type_Enum.NEVERGRAD_SINGLE: Nevergrad_Spice_Single_Objective,
    Optimizer_Type_Enum.AX_CONSTRAINT: Ax_Spice_Constraint_Satisfaction,
    Optimizer_Type_Enum.AX_SINGLE: Ax_Spice_Single_Objective,
}

# ------------------ Classes ------------------

class Circuit_Optimizer_Orchestrator_Base(ABC):
    def __init__(self, project_setup_path: str | Path, optimizer_type: Optimizer_Type_Enum, verbose: bool = False):
        self.project_setup_path = project_setup_path
        self.optimizer_type = optimizer_type
        self.verbose = verbose

        self.__post_init__()

    def __post_init__(self):
        # Order matters here
        self.project_setup:     Project_Setup   = self.read_project_setup()
        logger.debug(f"created the project setup for {self.project_setup.name}")

        self.spicelib_wrapper:  Spicelib_Wrapper = self.create_spicelib_wrapper()
        logger.debug(f"created the spicelib_wrapper.")


    def read_project_setup(self) -> Project_Setup:
        # Load the project setup information
        try:
            PROJECT_SETUP = Project_Setup.from_yaml(yaml_path=self.project_setup_path)
        except Exception as e:
            logger.critical(
                f"Failed to load project setup from '{self.project_setup_path}': {e.__class__.__name__}: {e}"
            )
            raise RuntimeError(
                f"Error loading project setup from '{self.project_setup_path}'. "
                f"Check if the file exists, is valid YAML, and has correct structure."
            ) from e
        return PROJECT_SETUP
    
    def create_spicelib_wrapper(self) -> Spicelib_Wrapper:
        PROJECT_SETUP = self.project_setup
        # (2) Create the Spice Simulator Wrapper
        netlist_filename = Path(PROJECT_SETUP.ws_root) / Path(PROJECT_SETUP.netlist)
        output_folder    = Path(PROJECT_SETUP.ws_root) / Path(PROJECT_SETUP.outdir)

        logger.debug(f"spicelib_wrappper will use the following configs:")
        logger.debug(f"\t- project_name {PROJECT_SETUP.name}")
        logger.debug(f"\t- output_folder {output_folder}")
        logger.debug(f"\t- netlist_filename {netlist_filename}")
        logger.debug(f"\t- path_to_simulator {PROJECT_SETUP.simulator}")

        wrapper = Spicelib_Wrapper(
            project_name=PROJECT_SETUP.name,
            netlist_filename=netlist_filename,
            output_folder=output_folder,
            sim_execution_t=Sim_Execution_Type.RUN_AND_WAIT,  # only RUN_AND_WAIT is supported as of now...,
            path_to_simulator=Path(PROJECT_SETUP.simulator),
            verbose=self.verbose
            )

        return wrapper
    
    def get_project_setup(self) -> Project_Setup:
        return self.project_setup
    
    def get_spicelib_wrapepr(self) -> Spicelib_Wrapper:
        return self.spicelib_wrapper

    @abstractmethod
    def get_optimizer(self) -> Base_Optimizer:
        pass  


class Circuit_Optimizer_Orchestrator_with_SPICE(Circuit_Optimizer_Orchestrator_Base):
    def get_optimizer(self) -> Spice_Base_Optimizer:
        logger.info(f"creating the circuit_optimizer of type {self.optimizer_type.value}")
        circuit_optimizer = SPICE_OPTIMIZER_CLASSES[self.optimizer_type](
            spicelib_wrapper=self.spicelib_wrapper,
            setup_obj=self.project_setup
        )
        logger.info(f"created the circuit_optimizer; type {type(circuit_optimizer)}")
        return circuit_optimizer
    
    def run_sanity_on_spicelib_wrapper(self, use_editor: bool = True)-> bool:
        return self.spicelib_wrapper.run_sanity_check(use_editor=use_editor, sim_execution_t=Sim_Execution_Type.RUN_NOW)