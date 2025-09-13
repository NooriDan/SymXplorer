import os

import torch
import numpy as np
import sympy 
import tqdm
import shutil

from pathlib import Path
from time    import sleep

from spicelib import SimRunner, RawRead, SpiceEditor, AscEditor
# Import simulation runners
from spicelib.simulators.ltspice_simulator import LTspice
from spicelib.simulators.ngspice_simulator import NGspiceSimulator
from spicelib.simulators.xyce_simulator    import XyceSimulator

# For typing
from typing import List, Dict, Tuple, Any
from spicelib.sim.simulator  import Simulator as SpicelibSimulatorClass
from spicelib.sim.run_task   import RunTask   as SpicelibRunTaskClass
from spicelib.editor.base_editor import ParameterNotFoundError, ComponentNotFoundError

from .utils import setup_loggers

SIM_ENGINES = {
    "ltspice" : LTspice,
    "ngspice" : NGspiceSimulator,
    "xyce"    : XyceSimulator
}


class LTspice_Wrapper:
    def __init__(self, asc_filename: str, traces_of_interest: List[str] = [], dump_parent_folder: str = "runner", verbose: bool = False):
        """Reads and simulates the circuit defined in the given .asc file"""
        self.asc_filename: str = asc_filename
        self.netlist: AscEditor = AscEditor(asc_file=asc_filename)
        self.simengine:  SpicelibSimulatorClass  = SIM_ENGINES["ltspice"]

        output_folder = f"{dump_parent_folder}/{self.simengine.__name__}"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        self.runner: SimRunner = SimRunner(simulator=self.simengine, verbose=verbose, output_folder=output_folder)
        self.output_folder = output_folder
        self.verbose = verbose

        if not self.validate_runner():
            raise RuntimeError("Runner Cannot be validated --- check LTspice simulator is available to spicelib")


        # Storing Simulation Runs
        self.traces:     List[str]    = traces_of_interest
        self.curr_raw: RawRead  = None
        self.tasks: Dict[SpicelibRunTaskClass] = {}
        self.cap_unit: str = 'p' # default size to pico
        self.res_unit: str = 'k' # default size to kilo

    def validate_runner(self) -> bool:
        """Validation logic to check SPICE simulator is loaded correctly"""

        if len(self.runner.simulator.get_default_library_paths()) < 1:
            print(f"* default libs for {self.runner.simulator.__name__} cannot be ressolved")
            return False
        
        if len(self.runner.simulator.spice_exe) < 1:
            print(f"* spice_exe for {self.runner.simulator.__name__} cannot be ressolved")
            return False
        
        return True
    
    def update_params(self, parameterization: Dict[str, float]) -> bool:

        for key, value in parameterization.items():

            try: # Validate parameter already exists
                self.netlist.get_parameter(key)
            except ParameterNotFoundError:
                return False

            if key.startswith("C"):
                self.netlist.set_parameter(key, f"{value}{self.cap_unit}")
            elif key.startswith("R"):
                self.netlist.set_parameter(key, f"{value}{self.res_unit}")
            else:
                self.netlist.set_parameter(key, f"{value}")
        
        return True

    def update_component_values(self, parameterization: Dict[str, float]) -> bool:
        for key, value in parameterization:

            try: # Validate parameter already exists
                self.netlist.get_parameter(key)
            except ParameterNotFoundError:
                return False

            if key.startswith("C"):
                self.netlist.set_component_value(key, f"{value}{self.cap_unit}")
            elif key.startswith("R"):
                self.netlist.set_component_value(key, f"{value}{self.res_unit}")
            else:
                self.netlist.set_component_value(key, f"{value}")
        
        return True
    
    def update_component_parameters(self, parameterization: Dict[str, Dict[str, float]]) -> bool:
        for component_name, component_parameters in parameterization:
            try:
                self.netlist.set_component_parameters(component_name, **component_parameters)
            except ComponentNotFoundError:
                return False
            
        return True

    @classmethod
    def callback(raw_file: str, log_file: str, traces_to_read: str):
        raw_read = RawRead(raw_filename=raw_file, traces_to_read=traces_to_read)
        return raw_read        
    
    def run_and_wait(self, exe_log: bool = True) -> Tuple[RawRead, str]:

        task = self.runner.run(self.netlist, exe_log=exe_log)

        while task.is_alive():
            pass # wait so its done

        raw_file, log_file = task.get_results()
        self.tasks[task.name] = (raw_file, log_file)

        self.curr_raw = RawRead(raw_filename=raw_file)

        return self.curr_raw, task.name
    
    # def run_with_callback(self):
    #     pass

    def extract_wave(self, wave_name: str, is_real: bool = False) -> torch.Tensor:
        
        if self.curr_raw is None:
            raise RuntimeError("Need to run the simulation at least once")
        
        wave = self.curr_raw.get_wave(wave_name)
        dtype = torch.float64 if is_real else torch.complex128

        if is_real:
            return torch.from_numpy(wave).real.to(dtype=torch.float64)
        
        return torch.from_numpy(wave)


class Spicelib_Wrapper:
    def __init__(self,  
                 netlist_filename:      Path, 
                 traces_of_interest:    List[str] = [], 
                 project_name:          str = "default_project", 
                 output_folder:         Path = Path("./spicelib_runs"),
                 path_to_simulator:     None | Path = None,
                 use_callback:          bool = False,
                 verbose:               bool = False,
                 ):
        """Reads and simulates the circuit defined in the given .spice file"""
        self.logger = setup_loggers()

        self.netlist_filename   = netlist_filename
        self.traces_of_interest = traces_of_interest
        self.project_name       = project_name
        self.output_folder      = output_folder
        self.path_to_simulator  = path_to_simulator
        self.use_callback       = use_callback
        self.verbose            = verbose

        self._default_compatibility_mode = "a" # ngspice compatibility mode (refer to spicelib and ngspice docs for details)
        self.runner: SimRunner | None = None
        self.editor: None | SpiceEditor = None
        self.tasks_outputs: Dict[str, Any] = {} # task name -> (raw, log) Tuple[Path, Path]
        self.curr_raw: RawRead | None = None

        self.__post_init__()
        
    def __post_init__(self):
        # (1) Validate the settings
        if not self._validate():
            raise RuntimeError("Spicelib wrapper validation failed")

        # (2) Create the simulator
        simulator : type[NGspiceSimulator] = self._create_simulator()

        # (3) Create the runner
        self.runner = SimRunner(
            simulator=simulator, 
            output_folder=self.output_folder,
            verbose=self.verbose
            )

        # (4) Create a SpiceEditor Instance
        self.editor = SpiceEditor(netlist_file=self.netlist_filename)

        # (5) print circuit info
        if self.verbose:
            self.print_circuit_info()
        
    def _validate(self) -> bool:
        if os.path.exists(self.output_folder):
            self.logger.warning(f"Output directory already exists, re-creating: {self.output_folder}")
            shutil.rmtree(self.output_folder)
        else:
            self.logger.info(f"Creating output directory for the first time: {self.output_folder}")
        
        os.makedirs(self.output_folder, exist_ok=False)

        if not self.netlist_filename.exists():
            raise FileNotFoundError(f"Initial netlist not found: {self.netlist_filename}")
        
        self.logger.info(f"project: {self.project_name}, schematic: {self.netlist_filename.stem}")

        return True

    def _create_simulator(self) -> type[NGspiceSimulator]:
        if self.path_to_simulator is not None:
            simulator = NGspiceSimulator.create_from(path_to_exe=self.path_to_simulator)
        else:
            simulator = NGspiceSimulator
        simulator.set_compatibility_mode(self._default_compatibility_mode)

        self.logger.info(f"Using ngspice from {simulator.spice_exe}")
        return simulator

    def print_circuit_info(self) -> None:
        if self.logger is None or self.editor is None:
            raise RuntimeError("Logger or Editor not initialized")
        logger = self.logger
        editor = self.editor
        # Nodes
        nodes = editor.get_all_nodes()
        logger.info(f"Nodes in the netlist: {nodes}")

        # Parameters
        tb_params  = self.get_tb_params()
        dut_params = self.get_dut_params()
        
        logger.info(f"Testbench parameters: {tb_params}")
        logger.info(f"DUT parameters: {dut_params}")

    def run_sanity_check(self, use_editor: bool = True) -> bool:
        # (1) Pre-body
        logger = self.logger
        if self.runner is None or self.editor is None:
            raise RuntimeError("Runner or Editor not initialized")
        # (2) Run the simulation with the parameters already in the netlist
        raw, log = self.runner.run_now(
            netlist= self.editor if use_editor else self.netlist_filename,
            exe_log=True,
            run_filename=f"{self.project_name}_sanity" )
        # (3) Check the simulation ran successfully
        if log is None or log.suffix == ".fail": 
            logger.error("Sanity check failed: log is .fail")
            return False
        if raw is None: 
            logger.error("Sanity check failed: RAW is None")
            return False
        if not raw.exists(): 
            logger.error("Sanity check failed: RAW returned but generation failed")
            return False
        if not log.exists():
            logger.error("Sanity check failed: log returned but generation failed")
            return False
        logger.info("Sanity check passed")
        return True

    def update_params(self, parameterization: Dict[str, float]) -> bool:
        RES_UNIT = 'k' # kilo
        CAP_UNIT = 'p' # pico
        if self.editor is None:
            raise RuntimeError("Editor not initialized")

        for key, value in parameterization.items():
            try: # Validate parameter already exists
                self.editor.get_parameter(key)
            except ParameterNotFoundError:
                return False

            if key.startswith("C"):
                self.editor.set_parameter(key, f"{value}{CAP_UNIT}")
            elif key.startswith("R"):
                self.editor.set_parameter(key, f"{value}{RES_UNIT}")
            else:
                self.editor.set_parameter(key, f"{value}")
        
        return True
    
    def get_dut_params(self) -> List[Tuple[str, Any]]:
        if self.editor is None:
            raise RuntimeError("Editor not initialized")
        editor = self.editor
        params = editor.get_all_parameter_names()
        dut_params = [(param, editor.get_parameter(param)) for param in params if "X_DUT" in param]
        return dut_params

    def get_tb_params(self) -> List[Tuple[str, Any]]:
        if self.editor is None:
            raise RuntimeError("Editor not initialized")
        editor = self.editor
        params = editor.get_all_parameter_names()
        tb_params  = [(param, editor.get_parameter(param)) for param in params if not "X_DUT" in param]
        return tb_params

    def simulate(self) -> None:
        pass

    def extract_wave(self, wave_name: str, is_real: bool = False) -> torch.Tensor:
        
        if self.curr_raw is None:
            raise RuntimeError("Need to run the simulation at least once")
        
        wave = self.curr_raw.get_wave(wave_name)
        dtype = torch.float64 if is_real else torch.complex128

        if is_real:
            return torch.from_numpy(wave).real.to(dtype=torch.float64)
        
        return torch.from_numpy(wave)
    
    def run_and_wait(self, exe_log: bool = True) -> Tuple[RawRead | None, str]:
        # (1) Pre-body
        logger = self.logger
        if self.runner is None or self.editor is None:
            raise RuntimeError("Runner or Editor not initialized")
        # (2) Run the simulation with the parameters already in the editor instance
        task = self.runner.run(self.editor, exe_log=exe_log)
        
        if task is None:
            raise RuntimeError("Failed to create a RunTask --- cannot proceed")
        
        # (3) Wait for the task to complete
        while task.is_alive():
            sleep(0.05)
            pass # wait so its done

        # (4) Get the results
        out = task.get_results()
        self.tasks_outputs[task.name] = out

        if isinstance(out, tuple) and len(out) == 2:
            raw_file, log_file = out
            self.curr_raw = RawRead(raw_filename=raw_file)
        else: self.curr_raw = None

        return self.curr_raw, task.name
    
    @classmethod
    def callback(cls, raw_file: str, log_file: str, traces_to_read: str):
        raw_read = RawRead(raw_filename=raw_file, traces_to_read=traces_to_read)
        return raw_read



if __name__ == "__main__":
    Spicelib_Wrapper(Path("/foss/designs/eda/SymXplorer/examples/tunable-tia/tia-bpf-1/netlist/tb_ac.spice"))