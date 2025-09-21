import sympy as sp

from pathlib import Path

from symxplorer.spice_engine import Spicelib_Wrapper, Sim_Execution_Type
from symxplorer.designer_tools import Nevergrad_Spice_Bode_Optimizer
from symxplorer.designer_tools.domains import Project_Setup
from symxplorer.designer_tools.tf_models import Second_Order_BP_TF, cascade_tf


print ("!!! Spicelib_Wrapper imported successfully !!!")



if __name__ == "__main__":
    # ----------------------------
    # Instantiations
    # ----------------------------
    project_setup_yaml = f"/foss/designs/eda/SymXplorer/examples/tunable-tia/ihp-sg13g2/spice/project_setup.yaml"
    
    # s = sp.symbols("s")
    # target_tf = (s + 1) / (s**2 + 24*s + 2)
    bpf_inst = Second_Order_BP_TF(q=10, fc=1e9, k_bp=1e3)
    target_tf = bpf_inst.get_tf()


    # (1) Load the project setup information
    PROJET_SETUP = Project_Setup.from_yaml(project_setup_yaml)
    print ("!!! Spicelib_Wrapper imported successfully !!!")

    # (2) Create the Spice Simulator Wrapper
    wrapper = Spicelib_Wrapper(
        project_name=PROJET_SETUP.project_name,
        netlist_filename= PROJET_SETUP.ws_root / PROJET_SETUP.netlist,
        output_folder=PROJET_SETUP.ws_root / PROJET_SETUP.outdir,
        sim_execution_t=Sim_Execution_Type.RUN_AND_WAIT,  # only RUN_AND_WAIT is supported as of now...
        verbose=False
        )
    wrapper_logger = wrapper.get_logger()

    
    # circuit_optimizer = Nevergrad_Spice_Bode_Optimizer(
    #     spicelib_wrapper=wrapper,
    #     target_tf=target_tf,
    #     output_node='vout',
    #     frequency_weight=None,
    #     optimizer_config=PROJET_SETUP.optimizer_config,
    # )

    # ----------------------------
    # Method Calls
    # ----------------------------




