# Tunable Transimpedance Amplifier (TIA) Example

This example explores the symbolic analysis and automated sizing of a tunable Transimpedance Amplifier (TIA), a critical component in optical communication receivers.

## Circuit Overview

A TIA converts a weak input current signal (typically from a photodiode) into a usable output voltage. This example focuses on a Common-Gate (CG) TIA topology, which is popular for its wide bandwidth. The "tunable" aspect refers to the ability to adjust the amplifier's characteristics, such as its gain or bandwidth, by changing bias conditions.

## Directory Structure

-   **/sym-exploration/**: Contains Jupyter notebooks that use `SymXplorer` for the symbolic analysis and sizing of the CG-TIA. 
    -   `CG_TIA_Exploration.ipynb`: Focuses on the symbolic derivation of the TIA's transfer function.
    -   `CG_TIA_Sym_Sizing.ipynb`: Demonstrates a basic sizing approach.
    -   `CG_TIA_Sym_Sizing_w_Optimizer.ipynb`: Implements a full automated sizing flow using an optimization engine.
-   **/ihp-sg13g2/** and **/gf180/**: These directories contain technology-specific files for the IHP SG13G2 and GlobalFoundries 180nm open-source PDKs, respectively. They include `xschem` schematics and SPICE files needed to simulate and verify the TIA design in these specific processes.
-   **/docs/**: Contains relevant academic papers and technical documents that provide background and context for TIA design.
