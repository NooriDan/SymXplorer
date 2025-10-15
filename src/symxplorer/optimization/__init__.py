"""The Circuit Optimization Module"""
# Module imports
from .orchestrator  import Circuit_Optimizer_Orchestrator_with_SPICE, SPICE_OPTIMIZER_CLASSES, Optimizer_Type_Enum
from .nevergrad     import Nevergrad_Spice_Bode_Optimizer, Nevergrad_Spice_Constraint_Satisfaction,  Nevergrad_Spice_Single_Objective
from .bayesian_ax   import Ax_Spice_Constraint_Satisfaction, Ax_Spice_Single_Objective


# ------------------ Module Exports ------------------

__all__ = [
    # --------------------------------
    # One endpoint for all optimizer types
    'SPICE_OPTIMIZER_CLASSES',
    'Optimizer_Type_Enum',
    'Circuit_Optimizer_Orchestrator_with_SPICE',
    # --------------------------------

    # ** For backward compatability **
    # --------------------------------
    # => Evolutionary-based Optimizers
    'Nevergrad_Spice_Bode_Optimizer', 
    'Nevergrad_Spice_Constraint_Satisfaction',
    'Nevergrad_Spice_Single_Objective',
    # => BO-based Optimizers
    'Ax_Spice_Constraint_Satisfaction',
    'Ax_Spice_Single_Objective'
    # --------------------------------
    ]