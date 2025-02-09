# SymXplorer

Welcome to the **SymXplorer** project! This is an opensource symbolic toolbox for analyzing analog circuits based on [SymPy](https://www.sympy.org/en/index.html) in python. Everything from transistor level circuits to active analog filters with multiple feedback. We're actively developing new features. A python-based toolbox means endless opportunities for future integeration with scientifict and machine learning ([PyTorch](https://pytorch.org/)) toolboxes.

## Features
- Customize nodal equation and impedance combination for new circuits under test (derive the symbolic nodal equations using [lcapy](https://lcapy.readthedocs.io/en/latest/))
- Analyze the possible filters, oscillators, and more!
- Model non-idealities of circuit components (e.g., FET transistors through their T matrix)
- Explore possiblem higher-order (2+) transfer functions for filter design.
- Size (through [Ax](https://ax.dev/) for Bayesian Optimization tool and [Nevergrad](https://github.com/facebookresearch/nevergrad) for Evolutionary algorithms) and visualize a filter's response quickly for design.
- Generate automatic LaTeX report of your runs
- Currently have pre-defined demo-circuits for Common-gate (CG) and Common-source (CS) differential input/output ([here](src/symcircuit/demo/differential.py)), multipl-feedback filter designs in current and voltage mode ([here](src/symcircuit/demo/multiple_feedback.py)), sallen-key topology ([here](src/symcircuit/demo/sallen_key.py)), and dual-amplifier ([here](src/symcircuit/demo/dual_amplifier.py)).

## Installation
After cloning the repository, open a terminal in the project directory and run the following command:

```bash
pip install -e .
```

# How to get started
First Git clone the repo, and follow the instructions below you'll find information and links to the key notebooks.

- The main codeblocks are under [src](src/macanalog_symbolix/) 
- Quickly get started by running "run-symbolix" in a CLI to run the symbolic experiment defined in [common_gate_setup](src/symcircuit/demo/differential.py) and [main](src/symcircuit/symbolic_solver/main.py)
- Check out summary report of previous the latest runs in [Runs](Runs)
- Find Previous papers under [Papers](docs/Papers)
- Refer to the [scripts](scripts) for jupyternotebooks on how to use the tool.
- A demonstration video will be available soon

## License
This project is licensed under the GNU General Public License v3 (GPLv3) - see the [LICENSE](LICENSE) file for details.
