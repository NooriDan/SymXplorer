import sympy as sp
import logging

from pathlib import Path

from symxplorer.spice_engine            import Spicelib_Wrapper, Sim_Execution_Type
from symxplorer.designer_tools          import Nevergrad_Spice_Bode_Optimizer
from symxplorer.designer_tools.utils    import Frequency_Weight
from symxplorer.designer_tools.domains  import Project_Setup
from symxplorer.designer_tools.tf_models import Second_Order_BP_TF, cascade_tf

from symxplorer.logging import setup_loggers

logger = logging.getLogger("SymXplorer")
logger.info("!!! Spicelib_Wrapper imported successfully !!!")



if __name__ == "__main__":
    # ----------------------------
    # Instantiations
    # ----------------------------
    project_setup_yaml = Path(f"/foss/designs/eda/SymXplorer/examples/tunable-tia/ihp-sg13g2/spice/project_setup.yaml")
    setup_loggers()
    
    # s = sp.symbols("s")
    # target_tf = (s + 1) / (s**2 + 24*s + 2)
    filter_inst = Second_Order_BP_TF(q=10, fc=1e9, k_bp=1e3)
    target_tf   = filter_inst.get_tf()

    # (1) Load the project setup information
    PROJECT_SETUP = Project_Setup.load(project_setup_yaml)

    # (2) Create the Spice Simulator Wrapper
    wrapper = Spicelib_Wrapper(
        project_name=PROJECT_SETUP.name,
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
    circuit_optimizer.optimize()

    circuit_optimizer.plot_loss(save_path=project_setup_yaml.parent / "loss_curve.html")

    out = circuit_optimizer.get_best_params()
    if out is not None: 
        best_param, loss = out


