# 5-Transistor OTA Example

This example demonstrates a complete design and sizing flow for a classic 5-transistor Operational Transconductance Amplifier (OTA), a fundamental building block in analog integrated circuits.

## Circuit Overview

The 5T-OTA consists of a differential pair with a current mirror load. It's a simple yet versatile amplifier used in many applications, such as filters, integrators, and voltage-controlled oscillators.

## Directory Structure

This example is structured for the IHP SG13G2 open-source PDK.

-   **/ihp-sg13g2/xschem/**: Contains the circuit schematics (`.sch` files) for the OTA and its various testbenches, created using Xschem.
-   **/ihp-sg13g2/spice/**: Holds the SPICE netlists for simulation. This includes the main OTA netlist and testbench files for loop gain, noise, transient, and AC analysis.
-   **/ihp-sg13g2/sizing/**: Contains Jupyter notebooks that use `SymXplorer` to perform automated sizing of the OTA. It showcases using both the `Ax` (Bayesian) and `Nevergrad` (Evolutionary) optimizers.
-   **/ihp-sg13g2/klayout/**: Contains the physical layout files (`.gds` or `.oas`) for the OTA, designed in KLayout.
