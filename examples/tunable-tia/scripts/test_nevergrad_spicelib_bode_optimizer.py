import sympy as sp

from pathlib import Path

from symxplorer.spice_engine            import Spicelib_Wrapper, Sim_Execution_Type
from symxplorer.designer_tools          import Nevergrad_Spice_Bode_Optimizer
from symxplorer.designer_tools.utils    import Frequency_Weight
from symxplorer.designer_tools.domains  import Project_Setup
from symxplorer.designer_tools.tf_models import Second_Order_BP_TF, cascade_tf


print ("!!! Spicelib_Wrapper imported successfully !!!")



if __name__ == "__main__":
    # ----------------------------
    # Instantiations
    # ----------------------------
    project_setup_yaml = f"/foss/designs/eda/SymXplorer/examples/tunable-tia/ihp-sg13g2/spice/project_setup.yaml"
    
    # s = sp.symbols("s")
    # target_tf = (s + 1) / (s**2 + 24*s + 2)
    filter_inst = Second_Order_BP_TF(q=10, fc=1e9, k_bp=1e3)
    target_tf   = filter_inst.get_tf()

    # (1) Load the project setup information
    PROJECT_SETUP = Project_Setup.from_yaml(project_setup_yaml)
    print ("!!! Spicelib_Wrapper imported successfully !!!")

    # (2) Create the Spice Simulator Wrapper
    wrapper = Spicelib_Wrapper(
        project_name=PROJECT_SETUP.project_name,
        netlist_filename= PROJECT_SETUP.ws_root / PROJECT_SETUP.netlist,
        output_folder=PROJECT_SETUP.ws_root / PROJECT_SETUP.outdir,
        sim_execution_t=Sim_Execution_Type.RUN_AND_WAIT,  # only RUN_AND_WAIT is supported as of now...
        verbose=False
        )
    wrapper_logger = wrapper.get_logger()

    
    circuit_optimizer = Nevergrad_Spice_Bode_Optimizer(
        spicelib_wrapper=wrapper,
        target_tf=target_tf,
        output_node='vout',
        frequency_weight=Frequency_Weight(lower=1, upper=1e12),
        setup_obj=PROJECT_SETUP
    )

    # ----------------------------
    # Method Calls
    # ----------------------------
    circuit_optimizer.parameterize()
    circuit_optimizer.create_experiment()
    circuit_optimizer.optimize()


