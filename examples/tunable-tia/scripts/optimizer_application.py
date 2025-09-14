from pathlib import Path

from symxplorer.spice_engine import Spicelib_Wrapper


print ("Spicelib_Wrapper imported successfully")


if __name__ == "__main__":
    netlist_path = "/foss/designs/eda/SymXplorer/examples/tunable-tia/ihp-sg13g2/spice/tb_ac.spice"
    wrapper = Spicelib_Wrapper(Path(netlist_path))
    wrapper.run_sanity_check(use_editor=True)