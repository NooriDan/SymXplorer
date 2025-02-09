import os

import torch
import numpy as np
import sympy 
import tqdm


from spicelib import SimRunner, RawRead, SpiceEditor, AscEditor
# Import simulation runners
from spicelib.simulators.ltspice_simulator import LTspice
from spicelib.simulators.ngspice_simulator import NGspiceSimulator
from spicelib.simulators.xyce_simulator    import XyceSimulator

# For typing
from typing import List, Dict, Tuple
from spicelib.sim.simulator  import Simulator as SpicelibSimulatorClass
from spicelib.sim.run_task   import RunTask   as SpicelibRunTaskClass
from spicelib.editor.base_editor import ParameterNotFoundError, ComponentNotFoundError
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


