from pathlib import Path

from symxplorer.spice_engine import Spicelib_Wrapper


print ("Spicelib_Wrapper imported successfully")


if __name__ == "__main__":
    wrapper = Spicelib_Wrapper(Path("/foss/designs/eda/SymXplorer/examples/tunable-tia/tia-bpf-1/netlist/tb_ac.spice"))
    wrapper.run_sanity_check(use_editor=True)