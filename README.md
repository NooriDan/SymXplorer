# SymXplorer: Symbolic and Optimization-Based Analog Design Automation

Welcome to the **SymXplorer** project! This is an opensource symbolic toolbox for analyzing analog circuits based on [SymPy](https://www.sympy.org/en/index.html) in python. Everything from transistor level circuits to active analog filters with multiple feedback. We're actively developing new features. **We have added class-based optimization algorithms that use SPICE-in-the-loop simulations** to help designers size, understand and visualize their cicrcuit topology. A python-based toolbox means endless opportunities for future integeration with scientifict, machine learning, and optimization toolboxes such as [PyTorch](https://pytorch.org/), [Ax](https://ax.dev/), and [Nevergrad](https://facebookresearch.github.io/nevergrad/).

## Features
- Customize nodal equation and impedance combination for new circuits under test (optional: derive the symbolic nodal equations using [lcapy](https://lcapy.readthedocs.io/en/latest/))
- **Analyze** the possible filters, oscillators, and more!
- Model non-idealities of circuit components (e.g., FET transistors through their T matrix)
- **Explore** possiblem higher-order (2+) transfer functions for filter design.
- **Size** (through [Ax](https://ax.dev/) for Bayesian Optimization tool and [Nevergrad](https://github.com/facebookresearch/nevergrad) for Evolutionary algorithms) and visualize a filter's response quickly.
- **Visualize** the design space and the understand the performance boundries of your design in a certain PDK.
- **Constraint satisfaction and single objective based optimization is available.**
- Interfaces with opensource simulator **ngspice** or other SPICE-based simulators (via [spicelib](https://github.com/nunobrum/spicelib)) and PDKs. Suggested to use in [IIC-OSIC docker container](https://github.com/iic-jku/IIC-OSIC-TOOLS).
- Generate automatic LaTeX report of your runs
- Symbolic Exploration: Currently have pre-defined demo-circuits for Common-gate (CG) and Common-source (CS) differential input/output ([here](src/symcircuit/demo/differential.py)), multipl-feedback filter designs in current and voltage mode ([here](src/symcircuit/demo/multiple_feedback.py)), sallen-key topology ([here](src/symcircuit/demo/sallen_key.py)), and dual-amplifier ([here](src/symcircuit/demo/dual_amplifier.py)).
- Circuit sizing: Explore [5-transistor OTA](examples/5t-ota) or [Differential Common Gate TIA](examples/5t-ota) examples directory that use the open-source IHP PDK.
# Examples

# How to get started
First Git clone the repo, and follow the instructions below you'll find information and links to the key notebooks.

## Installation
After cloning the repository, open a terminal in the project directory and run the following command:

```bash
pip install -e .
```
## Notes
- The main codeblocks are under [src](src/symxplorer/) 
- Quickly get started by running "run-symbolix" in a CLI to run the symbolic experiment defined in [common_gate_setup](src/symcircuit/demo/differential.py) and [main](src/symcircuit/symbolic_solver/main.py)
- Check out summary report of previous the latest runs in [Examples](examples)
- Find Previous papers under [Papers](docs/Papers)
- Refer to the jupyternotebooks on how to use the SymXplorer API for [topology exploration](examples/tunable-tia/sym-exploration) and [example automated sizing as solution to a constraint satisfaction problem.](examples/tunable-tia/ihp-sg13g2/sizing/tunable_tia_sizing_with_multi_spec_constraint_sat.ipynb).
- A demonstration video will be available soon

## License
This project is licensed under the GNU General Public License v3 (GPLv3) - see the [LICENSE](LICENSE) file for details.
