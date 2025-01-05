from dataclasses import dataclass, field
from typing      import Dict, List, Optional, Callable
import sympy
from sympy import symbols, Matrix
# Custom imports


class Impedance_Block:
    def __init__(self, name: str):
        self.name = str(name)
        self.symbol: sympy.Basic = sympy.Symbol(f"Z_{name}")

        s = sympy.symbols("s")
        self.Z_R: sympy.Basic = sympy.symbols(f"R_{name}", real=True)
        self.Z_L: sympy.Basic = s * sympy.symbols(f"L_{name}", real=True)
        self.Z_C: sympy.Basic = 1 / (s * sympy.symbols(f"C_{name}", real=True))

        # to be computed
        self.allowedConnections: List[sympy.Basic] = []

        # control variables
        self.zDictionary: Dict[str, sympy.Basic] = {
            "R" : self.Z_R,
            "L" : self.Z_L,
            "C" : self.Z_C
        }
        self.conectionSymbols: Dict[str, Callable] = {
            "|" : self.parallel,
            "+" : self.series
        }
        self.startOfFunctionToken: str  = "*START*"
        self.endOfFunctionToken:   str  = "*END*"
    
    def __repr__(self):
        return str(self.symbol)

    def simplify(self):
        for i, _impedance in enumerate(self.allowedConnections):
            # self.allowedConnections[i] = sympy.expand(_impedance) # Inefficient
            self.allowedConnections[i] = sympy.cancel(_impedance)
            print(f"batch {i}")

    
    def series(self, list_of_impedances: List[sympy.Basic]):
        
        equivalentZ = list_of_impedances[0]
        for impedance in list_of_impedances[1:]:
            equivalentZ += impedance
        # print(f"{self.symbol} = {sympy.factor(equivalentZ)}")
        return sympy.factor(equivalentZ)
    
    def parallel(self,list_of_impedances: List[sympy.Basic]):
        
        equivalentG = 1/list_of_impedances[0]
        for impedance in list_of_impedances[1:]:
            equivalentG += 1/impedance
        # print(f"{self.symbol} = {sympy.factor(1/equivalentG)}")
        return sympy.factor(1/equivalentG)
    
    def setAllowedImpedanceConnections(self, allowedConnections_texts: List[str]):
        """
        Reads from allowedConnections_texts and converts each string representation
        of the impedance connections to its symbolic expression.
        """
        for conn_text in allowedConnections_texts:
            parsed = self.parse_expression(conn_text)
            self.allowedConnections.append(parsed)

    def parse_expression(self, expression: str):
        """
        Parse a string expression to build the symbolic impedance representation.
        """
        # print(f"Original Expression: {expression}")

        # Replace component symbols with their symbolic equivalents
        for key, value in self.zDictionary.items():
            expression = expression.replace(key, f"self.zDictionary['{key}']")
        # print(f"1 - After Replacing Symbols: {expression}")

        # Handle nested parentheses
        while "(" in expression:
            start = expression.rfind("(")
            end = expression.find(")", start)
            if end == -1:
                raise ValueError("Unmatched parentheses in expression.")
            inner = expression[start + 1:end]
            inner_parsed = self._replace_operators(inner)
            expression = expression[:start] + inner_parsed + expression[end + 1:]
            # print(f" 2 - After Parsing Parentheses: {expression}")

        # Final replacement for top-level operators
        expression = self._replace_operators(expression)
        # print(f"Final Parsed Expression: {expression}")

        # Safely evaluate the expression
        try:
            expression = expression.replace(self.startOfFunctionToken, "(")
            expression = expression.replace(self.endOfFunctionToken, ")")
            result = sympy.simplify(eval(expression))
        except Exception as e:
            raise ValueError(f"Failed to parse expression: {expression}. Error: {e}")

        return result

    def _replace_operators(self, expression: str):
        """
        Replace connection operators in the expression:
        - "|" -> "self.parallel([...])"
        - "+" -> "self.series([...])"
        & -> '('
        """
        if "+" in expression:
            terms = expression.split("+")
            replaced = ", ".join(terms)
            return f"self.series{self.startOfFunctionToken} [{replaced}] {self.endOfFunctionToken}"  # Corrected with parentheses

        if "|" in expression:
            terms = expression.split("|")
            replaced = ", ".join(terms)
            return f"self.parallel {self.startOfFunctionToken} [{replaced}] {self.endOfFunctionToken}"  # Corrected with parentheses

        # If no operators are found, return the expression as is
        return expression

class TransmissionMatrix:
    def __init__(self, defaultType: str="Symbolic", element_name: str="a"):
        self.defaultType = defaultType

        # Variables global to the class
        gm, ro, Cgd, Cgs    = symbols(f'g_m r_o C_gd C_gs', real=True)
        t11, t12, t21, t22  = symbols(f'{element_name}_11 {element_name}_12 {element_name}_21 {element_name}_22')
        s = symbols("s")

        self.transmission_matrix_dict: Dict[str, Matrix] ={
        "simple"          : Matrix([[0, -1/gm],[0, 0]]),
        "symbolic"        : Matrix([[t11, t12],[t21, t22]]),
        "some_parasitic"  : Matrix([[-1/(gm*ro), -1/gm],[0, 0]]),
        "full_parasitic"  : Matrix([[(1/ro + s*Cgd)/(s*Cgd - gm), 1/(s*Cgd - gm)],[(Cgd*Cgs*ro*s + Cgd*gm*ro + Cgs + Cgd)*s/(s*Cgd - gm), (Cgs+Cgd)*s/(s*Cgd - gm)]])
        }

    def getTranmissionMatrix(self, transmission_matrix_type = "symbolic") -> Matrix:
            if self.transmission_matrix_dict.get(transmission_matrix_type) is None:
                print(f"Invalide Transmission Matrix Selected ({transmission_matrix_type})")
                raise KeyError
            return self.transmission_matrix_dict.get(transmission_matrix_type)
    
    def get_element(self, row: int, col: int, transmission_matrix_type = "symbolic") -> sympy.Basic:

        transmission_matrix = self.getTranmissionMatrix(transmission_matrix_type)
            
        if (row>=transmission_matrix.shape[0] or col>=transmission_matrix.shape[1]):
            print(f"Invalide Row Col ({row}, {col}) Accessed in the Selected Transmission Matrix of size {self.getTranmissionMatrix(transmission_matrix_type).shape} (type: {transmission_matrix_type})")
            raise IndexError
        
        return transmission_matrix[row, col]

@dataclass
class Circuit:
    """The generic form of the infromation needed to set up the circuit"""
    impedances: List[Impedance_Block]
    nodal_equations: List[sympy.Equality]
    solve_for: List[sympy.Basic]
    impedancesToDisconnect: Optional[List[sympy.Basic]] = field(default_factory=list)  # Safely handle mutable defaults

    def __post_init__(self):
        """assumes all impedances can be disconnected if not specified"""
        if not self.impedancesToDisconnect:
            for impedace in self.impedances:
                self.impedancesToDisconnect.append(impedace.symbol)

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
class Filter_Classification:
    zCombo: List[sympy.Basic]
    transferFunc: sympy.Basic  # SymPy expression
    valid: bool = False
    fType: Optional[str] = "None"
    parameters: Optional[dict] = field(default_factory=dict)  # Safely handle mutable defaults
    filterOrder: Optional[str] = "None"  

    def __eq__(self, other) -> bool:
        if not isinstance(other, Filter_Classification):
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
    classifications: List[Filter_Classification]
