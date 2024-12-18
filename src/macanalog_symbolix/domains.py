from dataclasses import dataclass, field
from typing      import Dict, List, Optional
import sympy
# Custom imports
from .utils import Impedance


@dataclass
class Circuit:
    """The generic form of the infromation needed to set up the circuit"""
    impedances: List[Impedance]
    nodal_equations: List[sympy.Equality]
    solve_for: List[sympy.Basic]
    impedancesToDisconnect: Optional[List[sympy.Basic]] = field(default_factory=list)  # Safely handle mutable defaults

    def __post_init__(self):
        """assumes all impedances can be disconnected if not specified"""
        if not self.impedancesToDisconnect:
            for impedace in self.impedances:
                self.impedancesToDisconnect.append(impedace.Z)

    def hasSolution(self) -> bool:
        """
        Checks if the system of nodal equations can be solved using the 
        provided solveFor array and accounts for the free symbols from 
        both the solveFor variables and the impedanceBlocks.
        """
        nonImpedanceSymbols = set()
        for eq in self.nodal_equations:
            nonImpedanceSymbols.update(eq.free_symbols)

        for _z in self.impedancesToDisconnect:
            nonImpedanceSymbols.discard(_z)

        return len(self.solve_for) <= len(nonImpedanceSymbols)
    

@dataclass
class FilterClassification:
    zCombo: List[sympy.Basic]
    transferFunc: sympy.Basic  # SymPy expression
    valid: bool = False
    fType: Optional[str] = "None"
    parameters: Optional[dict] = field(default_factory=dict)  # Safely handle mutable defaults
    filterOrder: Optional[str] = "None"  

    def __eq__(self, other) -> bool:
        if not isinstance(other, FilterClassification):
            return NotImplemented
        return (self.fType == other.fType) and (self.filterOrder == other.filterOrder)

    def __repr__(self) -> str:
        return (
            f"FilterClassification("
            f"zCombo={self.zCombo}, transferFunc={self.transferFunc}, "
            f"valid={self.valid}, fType={self.fType}, parameters={self.parameters})"
        )
    
@dataclass
class ExperimentResult():
    baseHs: sympy.Basic
    classifications: List[FilterClassification]
