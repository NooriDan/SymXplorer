# SymXplorer: Symbolic and Optimization-Based Analog Design Automation

[![PyPI version](https://badge.fury.io/py/SymXplorer.svg)](https://badge.fury.io/py/SymXplorer)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**SymXplorer** is an open-source Python toolbox for the symbolic analysis and optimization-driven design of analog circuits. It leverages the power of `SymPy` for symbolic calculations and integrates with SPICE simulators and advanced optimization libraries to automate the circuit design process.

From transistor-level circuits to active filters with multiple feedback loops, SymXplorer provides tools to analyze, size, and explore the design space of analog topologies.

## Core Features

-   **Symbolic Circuit Analysis**:
    -   Derive and analyze symbolic transfer functions, impedances, and nodal equations using `SymPy`.
    -   Includes pre-defined symbolic models for various topologies like `Sallen-Key`, `Multiple-Feedback`, `Differential`, and `Dual-Amplifier` circuits.
    -   Optionally automate the extraction of nodal equations from netlists using `lcapy`.
    -   Model non-idealities of circuit components, such as the T-matrix representation of FETs.

-   **Optimization-Based Sizing**:
    -   Utilize SPICE-in-the-loop optimization to size transistors and passive components based on performance specifications.
    -   A high-level `Orchestrator` (`optimization/orchestrator.py`) simplifies running complex sizing tasks by managing the interaction between the chosen optimizer, the SPICE engine, and the design configuration.
    -   Supports multiple optimization strategies:
        -   **Bayesian Optimization** via [Ax](https://ax.dev/).
        -   **Evolutionary Algorithms** via [Nevergrad](https://facebookresearch.github.io/nevergrad/).
    -   Define objective functions and constraints to guide the optimization process.

-   **Design Space Exploration & Visualization**:
    -   Explore higher-order transfer functions for filter design.
    -   A built-in `Visualizer` (`designer_tools/visualizer.py`) plots crucial data for understanding design trade-offs, including optimization loss curves and interactive parallel coordinate plots of the design space.

-   **SPICE Integration**:
    -   Interfaces with ngspice and other SPICE-based simulators through the `spicelib` library.
    -   Run simulations to validate symbolic expressions and evaluate circuit performance during optimization.

-   **Advanced Capabilities**:
    -   **Transfer Function Modeling**: The `designer_tools/tf_models.py` module provides templates for defining ideal filter responses (e.g., 1st/2nd order low-pass, band-pass) from poles, zeros, and gain. These can be cascaded to create higher-order filters (e.g., Butterworth, Chebyshev) and used as targets for optimization.
    -   **Automated Reporting**: Generate automatic LaTeX reports summarizing the results of your analysis and optimization runs.
    -   Includes functionality for device modeling using TensorFlow (`tf_models`), allowing for more accurate and flexible component models.
    -   Generate automatic LaTeX reports summarizing the results of your analysis and optimization runs.

## Architecture and Vision

### Swappable Optimization Engines

The optimization framework is built on an abstract base class, allowing the core optimization engine to be easily swapped. This class-based hierarchy makes `SymXplorer` an ideal platform for research and development in optimization strategies. You can seamlessly switch between industry-proven optimizers like **Ax** (for Bayesian Optimization) and **Nevergrad** (for Evolutionary Algorithms), or integrate your own custom-built optimizer. This flexibility is key for comparing optimization techniques and advancing the field of analog circuit design automation.

### The Optimization Plan: `project_setup.yaml`

Every optimization task is guided by a `project_setup.yaml` file, which acts as a comprehensive "Optimization Plan" document. This file orchestrates the entire sizing process:

-   **Metrics & Goals**: The optimization targets are defined under `target_specs`. Each metric directly corresponds to a `.MEAS` statement in the SPICE netlist. This powerful feature means that if you can measure a performance metric in ngspice, you can optimize for it.
-   **Parameters**: The YAML file clearly separates tunable device parameters (`dut_params`) from fixed testbench parameters (`testbench_params`), giving you precise control over the optimization variables.
-   **Constraints**: Technology-specific constraints (e.g., min/max transistor dimensions for a given PDK) are defined in the `tech_spec` section, ensuring the optimizer explores a valid and manufacturable design space.

### Vision: Towards Agentic Design Workflows

The long-term vision for `SymXplorer` is to serve as a core tool within an agentic design workflow. The `project_setup.yaml` is not just a configuration file; it's a blueprint for an optimization task. In the future, we envision Large Language Models (LLMs) acting as AI design assistants, automatically generating these "Optimization Plans".

Recent research has shown that the convergence of optimizers like Bayesian Optimization is critically dependent on having a good starting point or a well-defined trust region. LLMs are increasingly capable of:
-   Understanding circuit topologies and symmetry considerations.
-   Applying circuit design guidelines (e.g., for biasing).
-   Interpreting performance metrics.

By leveraging these capabilities, an LLM agent could intelligently create the optimization plan, setting up a more efficient and effective design process. `SymXplorer` is being developed to be the engine that these future agents will use to explore, size, and verify analog circuits.


## Installation

You can use the package from PyPI:
```bash
pip install symxplorer
```

Or if you are interested in modifying code you can lone the repository and install the necessary dependencies. It is recommended to use a virtual environment.

```bash
git clone https://github.com/Nooridan/SymXplorer.git
cd SymXplorer
pip install -e .
```

This will install the `SymXplorer` package in editable mode and all the required dependencies listed in `pyproject.toml`.


## Getting Started

A simple way to see SymXplorer in action is to run the pre-defined symbolic exploration script from the command line:

```bash
run-symbolix
```

This command executes the symbolic analysis for a differential common-gate circuit defined in `src/symxplorer/symbolic_exploration/main.py`.

## Examples

The `examples` directory contains comprehensive Jupyter notebooks and project setups demonstrating how to use SymXplorer for various circuit design tasks.

### 1. 5-Transistor OTA (`/examples/5t-ota`)

This example provides a complete design flow for a 5-transistor OTA using the IHP SG13G2 open-source PDK. It includes:
-   **Schematics**: `xschem` directory for circuit diagrams.
-   **SPICE**: Netlists and simulation setups.
-   **Sizing**: Jupyter notebooks for automated sizing using both `Ax` and `Nevergrad`.
    -   `5t_ota_sizing_with_multi_spec_constraint_sat_ax.ipynb`
    -   `5t_ota_sizing_with_multi_spec_constraint_sat_nevergrad.ipynb`
-   **Layout**: `klayout` directory for the physical layout.

For more details, see the [5t-ota README](./examples/5t-ota/readme.md).

### 2. CMFM Filter (`/examples/CMFM-filter`)

This example showcases the symbolic exploration and sizing of a current-mode, multiple-feedback (CMFM) biquadratic filter.
-   **Symbolic Exploration**: A rich set of notebooks for analyzing filter transfer functions, including the effects of Gain-Bandwidth product limitations.
    -   `CMMF_filters.ipynb`
    -   `TIA_CMMF_filters_with_GB_limitation.ipynb`
-   **Sizing**: Demonstrates multiple approaches to sizing the filter.
    -   `ltspice-sizing`: Sizing using LTspice.
    -   `sym-sizing-bode-ax`: Sizing with symbolic expressions and Ax.
    -   `sym-sizing-bode-nevergrad`: Sizing with symbolic expressions and Nevergrad.

For more details, see the [CMFM-filter README](./examples/CMFM-filter/readme.md).

### 3. Tunable TIA (`/examples/tunable-tia`)

This example explores the design of a tunable transimpedance amplifier (TIA).
-   **Symbolic Exploration**: Notebooks for symbolic analysis of a common-gate TIA.
    -   `CG_TIA_Exploration.ipynb`
    -   `CG_TIA_Sym_Sizing.ipynb`
-   **Automated Sizing**: A notebook demonstrating automated sizing of the TIA using an optimizer.
    -   `CG_TIA_Sym_Sizing_w_Optimizer.ipynb`

For more details, see the [tunable-tia README](./examples/tunable-tia/readme.md).
## Citation

If you use **SymXplorer** in your research, please cite:

### Journal Paper
```bibtex
@Article{electronics15030515,
  AUTHOR = {Noori Zadeh, Danial and Elamien, Mohamed B.},
  TITLE = {SymXplorer: Symbolic Analog Topology Exploration of a Tunable Common-Gate Bandpass TIA for Radio-over-Fiber Applications},
  JOURNAL = {Electronics},
  VOLUME = {15},
  YEAR = {2026},
  NUMBER = {3},
  ARTICLE-NUMBER = {515},
  DOI = {10.3390/electronics15030515}
}
```

### Conference Paper
```bibtex
@INPROCEEDINGS{11244398,
  author={Zadeh, Danial Noori and Elamien, Mohamed B.},
  booktitle={2025 IEEE 68th International Midwest Symposium on Circuits and Systems (MWSCAS)},
  title={SymXplorer: A Designer's Toolbox for Automated Analog Circuit Topology Exploration},
  year={2025},
  pages={651-655},
  doi={10.1109/MWSCAS53549.2025.11244398}
}
```

## License

This project is licensed under the GNU General Public License v3 (GPLv3). See the [LICENSE](LICENSE) file for details.
