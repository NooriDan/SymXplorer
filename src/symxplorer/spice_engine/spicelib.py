from __future__ import annotations

import os

import torch
import numpy as np
import logging
import shutil

from pathlib import Path
from time    import sleep
from enum    import Enum

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

from symxplorer.logging import setup_loggers

logger = logging.getLogger("SymXplorer.spicelib")

# ---------------------------------
# Enums Definition
# ---------------------------------
class Sim_Engines_Type(Enum):
    LTSPICE = "ltspice"
    NGSPICE = "ngspice"
    XYCE    = "xyce"


class Sim_Execution_Type(Enum):
    RUN_AND_WAIT        = "RUN_AND_WAIT"
    RUN_NOW             = "RUN_NOW"
    RUN_WITH_CALLBACK   = "RUN_WITH_CALLBACK"

class Ngspice_Plot_Type(Enum):
    AC = "AC Analysis"
    OP = "OP"
# ---------------------------------
# Class Definitions
# ---------------------------------
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
                 sim_execution_t:       Sim_Execution_Type = Sim_Execution_Type.RUN_AND_WAIT,
                 path_to_simulator:     None | Path = None,
                 verbose:               bool = False,
                 ):
        """Reads, modifies, and simulates the circuit defined in the given netlist_filename .spice file"""
        # self.logger = setup_loggers(parent_folder=output_folder.parent, out_logname=project_name)
        self.logger = logger

        self.netlist_filename   = netlist_filename
        self.traces_of_interest = traces_of_interest
        self.project_name       = project_name
        self.output_folder      = output_folder
        self.path_to_simulator  = path_to_simulator
        self.sim_execution_t    = sim_execution_t
        self.verbose            = verbose

        self._default_compatibility_mode: str   = "a"   # ngspice compatibility mode (refer to spicelib and ngspice docs for details)
        self._dut_parameter_prefix: str         = "X_DUT"

        self.runner: SimRunner | None       = None
        self.editor: None | SpiceEditor     = None
        self.tasks_outputs: Dict[str, Any]  = {}    # task name -> (raw, log) Tuple[Path, Path]
        self.curr_raw: RawRead | None       = None
        self.curr_log: str | None           = None

        self.counter: int = 1

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
        self.print_circuit_info()

    def _validate(self) -> bool:
        # Check if output folder exists
        if os.path.exists(self.output_folder):
            self.logger.warning(f"⚠️ Output directory already exists, re-creating: {self.output_folder}")
            shutil.rmtree(self.output_folder)
        else:
            self.logger.info(f"📂 Creating output directory for the first time: {self.output_folder}")
        
        os.makedirs(self.output_folder, exist_ok=False)

        # Check for netlist existence
        if not self.netlist_filename.exists():
            self.logger.critical(f"❌ Initial netlist not found: {self.netlist_filename}")
            raise FileNotFoundError(f"Initial netlist not found: {self.netlist_filename}")

        # Log project info
        self.logger.info("--------------------------------------------------")
        self.logger.info("🚀 Spicelib_Wrapper initialized successfully!")
        self.logger.info(f"\t📝 Project: {self.project_name}")
        self.logger.info(f"\t📜 Schematic: {self.netlist_filename.stem}")
        self.logger.info(f"\t📂 Output Folder: {self.output_folder}")
        self.logger.info("--------------------------------------------------")
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

        logger.info("📊 --- Circuit Information ---")

        # Nodes
        nodes = editor.get_all_nodes()
        if nodes:
            logger.info(f"🔗 Nodes in the netlist: {nodes}")
        else:
            logger.warning("⚠️ No nodes found in the netlist!")

        # Parameters
        tb_params = self.get_tb_params()
        dut_params = self.get_dut_params()

        if tb_params:
            logger.info(f"Testbench parameters: {tb_params}")
        else:
            logger.warning("⚠️ No testbench parameters found!")

        if dut_params:
            logger.info(f"DUT parameters: {dut_params}")
        else:
            logger.warning("⚠️ No DUT parameters found!")

        logger.info("✅ --- Circuit info printed successfully --- 🎉 ")

    def run_sanity_check(
        self,
        use_editor: bool = True,
        sim_execution_t: Sim_Execution_Type = Sim_Execution_Type.RUN_NOW
    ) -> bool:
        logger = self.logger

        # (1) Pre-body
        if self.runner is None or self.editor is None:
            logger.critical("💥 Runner or Editor not initialized!")
            raise RuntimeError("Runner or Editor not initialized")

        # (1.1) Create a dedicated folder for sanity check
        if isinstance(self.runner.output_folder, Path):
            logger.info("📂 Creating dedicated sanity check folder...")
            self.runner.output_folder = self.runner.output_folder / "sanity_check"
            self.runner.output_folder.mkdir(parents=True, exist_ok=True)
            self.counter += 1
        else:
            logger.critical("❌ Runner output folder is not a Path instance!")
            raise RuntimeError("Runner output folder is not a Path instance")
        
        # (2) Run the simulation with the parameters already in the netlist
        logger.info("🧪 Running sanity check simulation...")
        
        netlist_used = self.editor if use_editor else self.netlist_filename
        run_filename = f"{self.project_name}_sanity.spice"
        raw, log = None, None

        # Allow running sanity check with different execution types
        if sim_execution_t == Sim_Execution_Type.RUN_NOW:
            logger.debug("⚡ Executing simulation immediately (RUN_NOW)")
            raw, log = self.runner.run_now(
                netlist=netlist_used,
                exe_log=True,
                run_filename=run_filename
            )
            logger.info(f"simulator log: {log}")
            logger.info(f"simulator RAW: {raw}")
        elif sim_execution_t == Sim_Execution_Type.RUN_AND_WAIT:
            logger.debug("⏳ Running simulation and waiting for completion...")
            self.run_and_wait(exe_log=True)
        elif sim_execution_t == Sim_Execution_Type.RUN_WITH_CALLBACK:
            logger.warning("🛑 RUN_WITH_CALLBACK not implemented yet 🚧")
            raise NotImplementedError("RUN_WITH_CALLBACK simulation execution type is not implemented yet :(")
        else:
            logger.critical("🚨 Invalid sim_execution_t provided!")
            raise RuntimeError("Invalid sim_execution_t")
        
        # (3) Check the simulation ran successfully
        logger.info("🔎 Verifying simulation results...")
        if log is None or log.suffix == ".fail":
            logger.error("❌ Sanity check failed: log is .fail")
            return False
        if raw is None:
            logger.error("❌ Sanity check failed: RAW is None")
            return False
        if not raw.exists():
            logger.error("❌ Sanity check failed: RAW returned but generation failed")
            return False
        if not log.exists():
            logger.error("❌ Sanity check failed: log returned but generation failed")
            return False

        logger.info("✅ Sanity check passed 🎉")

        # (4) Move out of the sanity check folder
        if isinstance(self.runner.output_folder, Path):
            logger.debug("📦 Restoring output folder to parent directory")
            self.runner.output_folder = self.runner.output_folder.parent
        
        return True

    def update_params(self, parameterization: Dict[str, float]) -> bool:
        logger = self.logger
        logger.debug(f"Updating parameters...")
        RES_UNIT = 'k' # kilo
        CAP_UNIT = 'p' # pico
        if self.editor is None:
            raise RuntimeError("Editor not initialized")

        for key, value in parameterization.items():
            try: # Validate parameter already exists
                self.editor.get_parameter(key)
            except ParameterNotFoundError:
                logger.error(f"❌ Parameter {key} not found in the netlist... exiting")
                return False

            if key.startswith("C"):
                self.editor.set_parameter(key, f"{value}{CAP_UNIT}")
            elif key.startswith("R"):
                self.editor.set_parameter(key, f"{value}{RES_UNIT}")
            else:
                self.editor.set_parameter(key, f"{value}")
                logger.debug(f"... Parameter {key} set to {value:.3e}")
        logger.debug(f"✅  All parameters updated successfully")
        return True
    
    def get_dut_params(self) -> List[Tuple[str, Any]]:
        self.logger.debug("Getting DUT parameters")
        if self.editor is None:
            raise RuntimeError("Editor not initialized")
        editor = self.editor
        params = editor.get_all_parameter_names()
        dut_params = [(param, editor.get_parameter(param)) for param in params if "X_DUT" in param]
        return dut_params

    def get_tb_params(self) -> List[Tuple[str, Any]]:
        self.logger.debug("Getting TB parameters")
        if self.editor is None:
            raise RuntimeError("Editor not initialized")
        editor = self.editor
        params = editor.get_all_parameter_names()
        tb_params  = [(param, editor.get_parameter(param)) for param in params if not "X_DUT" in param]
        return tb_params

    def extract_wave(self, wave_name: str, is_real: bool = False) -> torch.Tensor:
        """The endpiont to extract a waveform from the last simulation run"""
        if self.curr_raw is None:
            raise RuntimeError("Need to run the simulation at least once")
        
        wave = self.curr_raw.get_wave(wave_name)
        dtype = torch.float64 if is_real else torch.complex128

        if is_real:
            return torch.from_numpy(wave).real.to(dtype=torch.float64)
        return torch.from_numpy(wave)
    
    def extract_scalar_variable_from_raw(self, var_name: str | List[str], is_real: bool = True) -> Dict[str, np.float64]:
        
        if not isinstance(var_name, list):
            var_name = [var_name]
        
        wave_form : Dict[str, np.float64] = {}
        for var in var_name:
            try:
                temp = self.extract_wave(var, is_real=is_real)
                wave_form[var] = np.float64(temp[0])
            except IndexError:
                self.logger.error(f"❌ Variable {var} not found in the raw file")
                wave_form[var] = np.float64(np.nan)
        return wave_form
    
    def run_and_wait(self, exe_log: bool = True) -> Tuple[RawRead | None, str | None, str]:
        """Runs the simulation and waits for it to complete, returning the RawRead instance (or None), the log filename (or None), and task name"""
        # (1) Pre-body
        logger = self.logger
        if self.runner is None or self.editor is None:
            raise RuntimeError("Runner or Editor not initialized")

        # (1.1) Create a dedicated folder for this run
        if isinstance(self.runner.output_folder, Path):
            self.runner.output_folder = self.runner.output_folder / f"run_{self.counter}"
            self.runner.output_folder.mkdir(parents=True, exist_ok=True)
            self.counter += 1

        # (2) Run the simulation with the parameters already in the editor instance
        task = self.runner.run(
            netlist=self.editor, 
            exe_log=exe_log)
        
        if task is None:
            raise RuntimeError("Failed to create a RunTask --- cannot proceed")
        
        # (3) Wait for the task to complete
        while task.is_alive():
            sleep(0.01)
            pass # wait so its done

        # (4) Get the results
        out = task.get_results()
        self.tasks_outputs[task.name] = out

        if isinstance(out, tuple) and len(out) == 2:
            raw_file, log_file = out
            self.curr_raw = RawRead(raw_filename=raw_file)
            self.curr_log = log_file
        else: 
            self.curr_raw = None
            self.curr_log = None

        # Move out of the run folder
        if isinstance(self.runner.output_folder, Path):
            self.runner.output_folder = self.runner.output_folder.parent

        return self.curr_raw, self.curr_log, task.name
    
    def load_raw(self, raw_file: Path | RawRead) -> None:
        """Loads a RawRead instance from the given raw_file path"""
        
        if isinstance(raw_file, RawRead):
            self.curr_raw = raw_file
            return
        
        if not raw_file.exists():
            raise FileNotFoundError(f"Raw file not found: {raw_file}")
        
        self.curr_raw = RawRead(raw_filename=raw_file)
    
    def clean_up(self) -> None:
        """Cleans up all the files generated during the simulation runs"""
        if self.runner is None:
            raise RuntimeError("Runner not initialized")
        
        self.runner.cleanup_files()
        
        run_folder = self.output_folder / f"run_{self.counter}"
        if run_folder.exists() and run_folder.is_dir():
            shutil.rmtree(run_folder)

        self.logger.debug(f"🧹 All simulation files cleaned up successfully {run_folder}")

    @classmethod
    def callback(cls, raw_file: str, log_file: str, traces_to_read: str):
        raw_read = RawRead(raw_filename=raw_file, traces_to_read=traces_to_read)
        return raw_read

    def get_logger(self) -> logging.Logger:
        if self.logger is None:
            raise RuntimeError("Logger not initialized")
        return self.logger

if __name__ == "__main__":
    logger = setup_loggers()
    logger.info("Spicelib_Wrapper module imported successfully")