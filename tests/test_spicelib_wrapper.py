from pathlib import Path

from symxplorer.spice_engine import Spicelib_Wrapper, Sim_Execution_Type

print ("!!! Spicelib_Wrapper imported successfully !!!")



if __name__ == "__main__":
    ws_root = "/foss/designs/eda"
    netlist_path = f"{ws_root}/SymXplorer/examples/tunable-tia/ihp-sg13g2/spice/tb_ac.spice"
    
    # ----------------------------
    # Instantiation
    # ----------------------------
    wrapper = Spicelib_Wrapper(
        netlist_filename=Path(netlist_path),
        project_name="Tunable-TIA",
        output_folder=Path(f"{ws_root}/SymXplorer/examples/tunable-tia/scripts/optimizer_output"),
        sim_execution_t=Sim_Execution_Type.RUN_AND_WAIT,  # Change to RUN_AND_WAIT or RUN_WITH_CALLBACK as needed
        verbose=False
        )
    
    logger = wrapper.get_logger()

    
    # ----------------------------
    # Method Calls
    # ----------------------------

    # (1) run a sanity check that:
    #   a - Prints the nodes and parameters (DUT and Testbench) in the netlist
    #   b - Runs a simulation with the original netlist (checking ngspice is working)
    #   c - Checks if the output log file contains any errors (dependancy/include issues, netlist syntax errors, etc)
    wrapper.run_sanity_check(use_editor=True)

    # (2) Modify a parameter and run the simulation again
    dict_of_new_params = {
        "X_DUT_CAP_W" : 5e-9
    }
    wrapper.update_params(
        parameterization=dict_of_new_params
    )
    
    logger.info("Printing the circuit info after parameter update:")
    wrapper.print_circuit_info()
    
    # (3) Run the simulation with the updated parameters
    wrapper.run_and_wait()

    # (4) Extract a waveform (frequency response of output node "out" in this case)
    trace_name = "v(vop)"
    wave = wrapper.extract_wave(wave_name=trace_name, is_real=False)



