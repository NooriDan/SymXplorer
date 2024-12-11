import sympy
import itertools
from   sympy  import solve, simplify, symbols
from   sympy  import oo as inf
from   typing import Dict, List
# Custom Imports
import  Global  as GlobalVariables
from    Utils   import Impedance, TransmissionMatrix

class Circuit:
    def __init__(self, 
                solveFor:   List[sympy.Basic],
                nodalEquations:  List[sympy.Equality],
                impedanceBlocks: List[Impedance]):
        self.nodalEquations = nodalEquations
        self.impedances = impedanceBlocks
        self.solveFor = solveFor

    def hasSolution(self) -> bool:
        """
        Checks if the system of nodal equations can be solved using the 
        provided solveFor array and accounts for the free symbols from 
        both the solveFor variables and the impedanceBlocks.
        """
        nonImpedanceSymbols = set()
        for eq in self.nodalEquations:
            nonImpedanceSymbols.update(eq.free_symbols)

        for _z in self.impedances:
            nonImpedanceSymbols.discard(_z)

        return len(self.solveFor) <= len(nonImpedanceSymbols)


class CircuitSolver:
    """Implementation of algorithm 1 -> finds the base transfer function"""
    def __init__(self, 
                circuit:    Circuit,
                _output:    List[sympy.Basic], 
                _input:     List[sympy.Basic],
                transmissionMatrixType: str,
                impedancesToDisconnect:    List[Impedance] = GlobalVariables.impedancesToDisconnect,
                alwaysConnectedImpedances: List[Impedance] = [],
                transmissionMatrix: TransmissionMatrix = TransmissionMatrix() # Default T matrix (all symbolic)
                ):
        # Extract information from the Circuit object
        self.equations:  List[sympy.Basic]  = circuit.nodalEquations
        self.solveFor:   List[sympy.Basic]  = circuit.solveFor
        self.impedances: List[Impedance]    = circuit.impedances
        
        # Solver specific variables
        self.output:     List[sympy.Basic]  = _output
        self.input:      List[sympy.Basic]  = _input
        self.T_type:     str                = transmissionMatrixType
        self.T_analysis: TransmissionMatrix = transmissionMatrix  
        self.impedancesToDisconnect:    List[sympy.Basic] = impedancesToDisconnect
        self.alwaysConnectedImpedances: List[sympy.Basic] = alwaysConnectedImpedances

        # variables to be computed for
        self.impedanceConnections: List[Dict[str, List[sympy.Basic]]] = []
        # self._setPossibleImpedanceConnections()

        self.baseSymbolicHs: sympy.Basic             = None
        self.baseHs:         sympy.Basic             = None
        self.baseHsDict:     Dict[str, sympy.Basic]  = {}

    # Custom string representation for BaseTransferFunction
    def __repr__(self):
        return f"BaseTranferFunction object -> T_type {self.T_type}\n H(s) = ({self.output[0]} - {self.output[1]}) / ({self.input[0]} - {self.input[1]})"

    def isSolved(self):
        return self.baseHs is not None and self.baseSymbolicHs is not None
    
    def getImpedanceConnections(self, comboSize = 1):
        if self.baseHsDict is not None:
            try:
                return self.impedanceConnections[comboSize-1]
            except IndexError:
                print(f"Combo size is out of bound (max is {len(self.impedancesToDisconnect)})")
        else:
            print("The circuit needs to be solved first")
    
    def getBaseHs(self, key):
        tf = self.baseHsDict.get(key)
        if tf is None:
            print(f"!!! The key {key} doesn't exist or the circuit is not solved !!!")
            raise KeyError
        return tf

    def solve(self):
        print(f"====Solving the Circuit====")
        self._setPossibleImpedanceConnections()
        self._solveSymbolicTransmissionMatrix()
        self._solveForAllImpedancesConnected()
        self._findHsForImpedanceConnections()
        print(f"=====*Circuit Solved*=====")

    # Helper Functions for the solve method
    def _solveSymbolicTransmissionMatrix(self):

        oPos, oNeg = self.output
        iPos, iNeg = self.input

        print(" ----------------------------")
        print(f" Solving for ({oPos} - {oNeg}) / ({iPos} - {iNeg})")
        print(f"Intermediate Variables: {self.solveFor}")
        print(" ----------------------------")

        # Define nodal equations
        print("(1) set up the nodal equation")
        # equations = self._getNodalEquations()

        # Solve for generic transfer function
        solution = solve(self.equations, self.solveFor)
        print("(2) solved the base transfer function (symbolic [T])")

        if solution:
            print("FOUND THE BASE TF")
            baseHs: sympy.Basic = (solution[oPos] - solution[oNeg]) / (solution[iPos] - solution[iNeg])
            baseHs = simplify((baseHs.factor()))
            self.baseSymbolicHs = baseHs
            return baseHs
        print("!!!!!! COULD NOT SOLVE THE NODAL EQUATIONS !!!!!!")
    
    def _solveForAllImpedancesConnected(self):
        print(f"*** --- Solving for T type: {self.T_type} --- ***")
        sub_dict = {
            symbols("a11"): self.T_analysis.getTranmissionMatrix(self.T_type)[0, 0],
            symbols("a12"): self.T_analysis.getTranmissionMatrix(self.T_type)[0, 1],
            symbols("a21"): self.T_analysis.getTranmissionMatrix(self.T_type)[1, 0],
            symbols("a22"): self.T_analysis.getTranmissionMatrix(self.T_type)[1, 1],
        }

        Hs = self.baseSymbolicHs.subs(sub_dict)  # Substitute the impedance values into the base function
        Hs = simplify(Hs.factor())  # Simplify and factor the resulting expression
        self.baseHs = Hs
        print(f"*** --- Done --- ***")
        return Hs
    
    def _setPossibleImpedanceConnections(self):
        print(f"--- Computing the possible impedance connections for {self.impedancesToDisconnect} ---")
        variables = self.impedancesToDisconnect

        self.impedanceConnections = []

        # Convert alwaysConnectedImpedances to a single string to append to each key
        always_connected_str = '_'.join(str(var).replace('_', '') for var in self.alwaysConnectedImpedances)

        # Loop through different combination sizes (from 1 to len(variables))
        for combination_size in range(1, len(variables) + 1):
            # Generate combinations of the specified size
            combinations = itertools.combinations(variables, combination_size)
            # print(f"combination size = {combination_size}")
            myDict = {}
            for comb in combinations:
                # print(f"processing: {comb}")
                key = '_'.join(str(var).replace('_', '') for var in comb)

                # Append the always connected impedances to the key
                if always_connected_str:
                    key = f"{key}_{always_connected_str}"

                myDict[key] =  comb

            self.impedanceConnections.append(myDict)

        print("--- Impedance connections stored in CircuitSetUp.impedanceConnections---")
        return self.impedanceConnections

    def _applyLimitsToBase(self, variables_to_limit: List[sympy.Symbol], limitingValue: sympy.Basic = sympy.oo):
        """Applies sympy.limit to a set of variables."""
        baseHs = self.baseHs  # Local variable, doesn't modify self.baseHs
        for var in variables_to_limit:
            baseHs = sympy.limit(baseHs, var, limitingValue)
        return baseHs

    def _findHsForImpedanceConnections(self):
        # self.impedanceConnections is a list of Dictionaries where every index corresponds to a dictionary of combinations of size = index + 1
        if not hasattr(self, 'impedancesToDisconnect'):
            raise AttributeError("impedancesToDisconnect is not defined.")
        
        # Iterate through each list of combinations (which is a dictionary)
        for index, combinations in enumerate(self.impedanceConnections):
            print(f"processing combo size {index+1}")
            for key, symbols in combinations.items():
                # print(f"key = {key}, symbolds = {symbols}")
                # Filter the impedances that are not in the current combination
                impedance_list = [_zi for _zi in self.impedancesToDisconnect if _zi not in symbols]
                # print(f"impedances to kill: {impedance_list}")
                self.baseHsDict[key] = self._applyLimitsToBase(impedance_list)

