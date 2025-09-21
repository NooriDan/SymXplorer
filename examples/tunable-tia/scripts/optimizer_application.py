from pathlib import Path

from symxplorer.spice_engine import Spicelib_Wrapper


print ("Spicelib_Wrapper imported successfully")


if __name__ == "__main__":
    ws_root = "/foss/designs/eda"
    netlist_path = f"{ws_root}/SymXplorer/examples/tunable-tia/ihp-sg13g2/spice/tb_ac.spice"
    # Instantiation
    wrapper = Spicelib_Wrapper(
        netlist_filename=Path(netlist_path),
        project_name="Tunable-TIA",
        output_folder=Path(f"{ws_root}/SymXplorer/examples/tunable-tia/scripts/optimizer_output"),
        verbose=True
        )
    # Method Calls
    wrapper.run_sanity_check(use_editor=True)
