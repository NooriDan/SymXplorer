from spicelib import SpiceEditor, SimRunner, RawRead
from spicelib.simulators import ngspice_simulator 

import logging
import os
import shutil


from pathlib import Path


import logging

# --- Your notebook logger (unchanged) ---
logger = logging.getLogger("notebook_logger")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.propagate = False
formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
console_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(console_handler)
logger.info("Logger initialized.")

# --- NEW: Enable debug logging for spicelib ---
spicelib_logger = logging.getLogger("spicelib")
spicelib_logger.setLevel(logging.DEBUG)

# Optionally, attach the same console handler so spicelib logs show up too
if not spicelib_logger.handlers:
    spicelib_logger.addHandler(console_handler)

logger.info("spicelib logger set to DEBUG")



PATH_TO_NGSPICE = Path("/foss/tools/bin/ngspice")

PROJECT_NAME    = "tia-bpf-1"
SCHEMATIC_NAME  = "tb_ac"
SIZER_NAME      = "simple"

OUTPUT_DIR      = Path(f"./runs/{SIZER_NAME}/{PROJECT_NAME}")
INITIAL_NETLIST = Path(f"../../{PROJECT_NAME}/netlist/{SCHEMATIC_NAME}.spice")


if os.path.exists(OUTPUT_DIR):
    logger.warning(f"Output directory already exists, removing: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=False)

if not INITIAL_NETLIST.exists():
    raise FileNotFoundError(f"Initial netlist not found: {INITIAL_NETLIST}")


logger.info(f"Using ngspice from {PATH_TO_NGSPICE}")
logger.info(f"project: {PROJECT_NAME}, schematic: {SCHEMATIC_NAME}")


simulator = ngspice_simulator.NGspiceSimulator.create_from(path_to_exe=PATH_TO_NGSPICE)
simulator.set_compatibility_mode("a")

runner = SimRunner(
    simulator=simulator, 
    output_folder=OUTPUT_DIR,
    cwd=OUTPUT_DIR,
    )

runner.cwd = Path("./")

# Create a SpiceEditor Instance
editor = SpiceEditor(netlist_file=INITIAL_NETLIST)

# Nodes
nodes = editor.get_all_nodes()
logger.info(f"Nodes in the netlist: {nodes}")

# Parameters
params = editor.get_all_parameter_names()
tb_params  = [(param, editor.get_parameter(param)) for param in params if not "X_DUT" in param]
dut_params = [(param, editor.get_parameter(param)) for param in params if "X_DUT" in param]
logger.info(f"Testbench parameters: {tb_params}")
logger.info(f"DUT parameters: {dut_params}")



runtask = runner.run_now(
    netlist=INITIAL_NETLIST,
    exe_log=True )
runtask



