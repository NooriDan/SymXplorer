# Tunable Transimpedance Amplifier (TIA) Example

This example explores the symbolic analysis and automated sizing of a tunable Transimpedance Amplifier (TIA), a critical component in biomedical insturmentation. 

## Circuit Overview

A Transimpedance Amplifier (TIA) is a crucial component in many sensor interfaces, designed to convert a small input current into a usable output voltage. In the context of **biomedical instrumentation**, TIAs are often used to amplify faint currents from biosensors that detect neural signals, DNA, or other biological markers.

This specific example focuses on a tunable Common-Gate (CG) TIA that is configured to act as a **bandpass filter**. This is critical for biomedical applications where the signal of interest exists within a specific frequency band (e.g., the action potentials of neurons). The bandpass characteristic allows the amplifier to selectively amplify these target signals while rejecting out-of-band noise and interference. The "tunable" nature of the TIA allows its center frequency and bandwidth to be adjusted, making it adaptable for different types of biomedical sensors and signals.

## Directory Structure

-   **/sym-exploration/**: Contains Jupyter notebooks that use `SymXplorer` for the symbolic analysis and sizing of the CG-TIA. 
    -   `CG_TIA_Exploration.ipynb`: Focuses on the symbolic derivation of the TIA's transfer function.
    -   `CG_TIA_Sym_Sizing.ipynb`: Demonstrates a basic sizing approach.
    -   `CG_TIA_Sym_Sizing_w_Optimizer.ipynb`: Implements a full automated sizing flow using an optimization engine.
-   **/ihp-sg13g2/** and **/gf180/**: These directories contain technology-specific files for the IHP SG13G2 and GlobalFoundries 180nm open-source PDKs, respectively. They include `xschem` schematics and SPICE files needed to simulate and verify the TIA design in these specific processes.
-   **/docs/**: Contains relevant academic papers and technical documents that provide background and context for TIA design.
