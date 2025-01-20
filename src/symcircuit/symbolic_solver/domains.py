from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing      import Dict, List, Tuple, Optional, Callable, TYPE_CHECKING
import sympy
from sympy import symbols, Matrix, latex

import pandas as pd
import _pickle as pickle # cPickle
# Custom imports
if TYPE_CHECKING:
    # Imports for type checking only
    from .solver import Circuit_Solver

class Impedance_Block:
    def __init__(self, name: str):
        self.name = str(name)
        self.symbol: sympy.Basic = sympy.Symbol(f"Z_{name}")

        s = sympy.symbols("s")
        self.Z_R: sympy.Basic = sympy.symbols(f"R_{name}", real=True, positive=True)
        self.Z_L: sympy.Basic = s * sympy.symbols(f"L_{name}", real=True, positive=True)
        self.Z_C: sympy.Basic = 1 / (s * sympy.symbols(f"C_{name}", real=True, positive=True))

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
    

class ExperimentResult:

    def __init__(self, 
                experiment_name:      str,
                output_directory:     Optional[str] = "Runs/EXPERIMENT_NAME",
                circuit_solver:       'Circuit_Solver' = None,
                base_tfs_dict:        Optional[Dict[str, sympy.Basic]]  = {},                   # Safely handle mutable defaults
                classifications_dict: Optional[Dict[str, List[Filter_Classification]]] = {}):  # Safely handle mutable defaults
                
        self.experiment_name:      str = experiment_name
        self.output_directory:     Optional[str]   = output_directory
        self.circuit_solver:       'Circuit_Solver'  = circuit_solver
        self.base_tfs_dict:        Optional[Dict[str, sympy.Basic]] = base_tfs_dict         
        self.classifications_dict: Optional[Dict[str, List[Filter_Classification]]] = classifications_dict

        if output_directory == "Runs/EXPERIMENT_NAME":
            self.output_directory = f"Runs/{self.experiment_name}"
        os.makedirs(self.output_directory, exist_ok=True)
        
        self.paths_to_pkl_file: str = self.find_results_file(start_dir=self.output_directory)

    def add_all(self, impedance_key, baseHs, classifications):
        
        self.classifications_dict[impedance_key] = classifications
        self.base_tfs_dict[impedance_key] = baseHs

    def add_tf_only(self, impedance_key:str, classifications: List[Filter_Classification]):
        self.classifications_dict[impedance_key] = classifications

    def get(self, impedance_key: str) -> Tuple[List[Filter_Classification], sympy.Basic]:
        if self.classifications_dict.get(impedance_key) is None : 
            raise IndexError(f"The impedance key {impedance_key} does not exist in {self.classifications_dict.keys()}")
        
        if self.base_tfs_dict.get(impedance_key) is None: 
            raise IndexError(f"The impedance key {impedance_key} does not exist in {self.base_tfs_dict.keys()}")
        
        return self.classifications_dict[impedance_key], self.base_tfs_dict[impedance_key]

    def flatten_classifications(self) -> pd.DataFrame:
        """
        Flatten the classifications_dict into a Pandas DataFrame.
        Each row represents one Filter_Classification object, 
        along with its associated impedance_key.
        
        Returns:
            pd.DataFrame: Flattened DataFrame with attributes of Filter_Classification.
        """
        rows = []
        for impedance_key, classifications in self.classifications_dict.items():
            for classification in classifications:
                rows.append({
                    "impedance_key": impedance_key,
                    "zCombo": classification.zCombo,
                    "transferFunc": classification.transferFunc,
                    "transferFunc_latex": latex(classification.transferFunc),
                    "valid": classification.valid,
                    "fType": classification.fType,
                    "parameters": classification.parameters,
                    "filterOrder": classification.filterOrder,
                })

        return pd.DataFrame(rows)
    
    def flatten_tfs(self, output_name: str = "tfs") -> pd.DataFrame:
        """
        Flatten the baseHs_dict into a Pandas DataFrame.
        Each row represents an impedance key and its associated transfer function.

        Returns:
            pd.DataFrame: Flattened DataFrame with 'impedance_key' and 'transferFunc'.
        """
        rows = []
        for impedance_key, transfer_func in self.base_tfs_dict.items():
            rows.append({
                "impedance_key": impedance_key,
                "transferFunc": transfer_func,
                "transferFunc_latex": latex(transfer_func)
            })

        return pd.DataFrame(rows)

    def to_csv(self, run_detail_comment: str = ""):
        print(f"Writting {self.experiment_name} to csv files")

        filename = f"{self.output_directory}/classifications_{run_detail_comment}.csv"
        self.flatten_classifications().to_csv(filename)
        print(f"flattened all the classifications to {filename}")

        filename = f"{self.output_directory}/tfs_{run_detail_comment}.csv"
        self.flatten_tfs().to_csv(filename)
        print(f"flattened all the classifications to {filename}")

    def save(self, filename: str = "results.pkl") -> None:
        """
        Save the ExperimentResult object to results.pkl in Runs/self.experiment_name folder
        """

        filename = f"{self.output_directory}/{filename}"
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
            print(f"\n=== ExperimentResult saved successfully to {filename} - {os.path.getsize(filename)/1000.} kb ===\n")
        except Exception as e:
            print(f"Error saving ExperimentResult: {e}")

    def load(self, filename: str = "DEFAULT") -> ExperimentResult:
        """
        Load an ExperimentResult object from a file.

        Args:
            filename (str): The filename to load the object from.

        Returns:
            ExperimentResult: The loaded ExperimentResult object, or None if an error occurred.
        """


        if filename == "DEFAULT":
            filename = f"{self.output_directory}/results.pkl"
        
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)

            # # Ensure the loaded object is an instance of ExperimentResult
            # if type(obj) == type(ExperimentResult("dummy")):
            #     print(f"ExperimentResult loaded successfully from {filename}")
            # else:
            #     print(f"Loaded object is not of type ExperimentResult.")
            
            return obj

        except Exception as e:
            print(f"Error loading ExperimentResult: {e}")
            return None

    def find_results_file(self, start_dir: str = "./", filename: str = "results.pkl"):
        # List to store all the directories that contain the file
        directories = []
        
        for root, dirs, files in os.walk(start_dir):
            if filename in files:
                directories.append(root)
        
        return directories
    
    def update(self, where_to_look: str = "") -> str:
        if where_to_look == "":
            where_to_look = self.output_directory

        self.paths_to_pkl_file: str = self.find_results_file(start_dir=where_to_look)

        if len(self.paths_to_pkl_file) == 1:
            print(f"Found a unique results.pkl file at {self.paths_to_pkl_file[0]}")
            obj: 'ExperimentResult' = self.load(f"{self.paths_to_pkl_file[0]}/results.pkl")
            # update the current object
            self.experiment_name = obj.experiment_name
            self.base_tfs_dict   = obj.base_tfs_dict
            self.classifications_dict = obj.classifications_dict

            if (self.circuit_solver is None) or (obj.circuit_solver.input == self.circuit_solver.input and obj.circuit_solver.output == self.circuit_solver.output):
                self.circuit_solver = obj.circuit_solver
            else: 
                raise TypeError("Attempted to import the results for an experiment with different setup")

            return self.paths_to_pkl_file[0]

        print(f"could not ressolve the path to the pkl file (found {len(self.paths_to_pkl_file)})")
        return ""
    
    def update_solver_only(self, where_to_look: str = "") -> str:
        if where_to_look == "":
            where_to_look = self.output_directory

        filename = "results_circuit_solution.pkl"
        self.paths_to_pkl_file: str = self.find_results_file(start_dir=where_to_look, filename=filename)

        if len(self.paths_to_pkl_file) == 1:
            print(f"\n--- Found a unique pkl file at {self.paths_to_pkl_file[0]}/{filename}")
            obj: 'ExperimentResult' = self.load(f"{self.paths_to_pkl_file[0]}/{filename}")

            if (self.circuit_solver is None) or (obj.circuit_solver.input == self.circuit_solver.input and obj.circuit_solver.output == self.circuit_solver.output):
                # update the current object
                self.experiment_name = obj.experiment_name
                self.circuit_solver = obj.circuit_solver
                print("--- Updated the circuit solver")
            else: 
                raise TypeError("\n!!! Attempted to import the results for an experiment with different setup\n")

            return f"{self.paths_to_pkl_file[0]}/{filename}"

        print(f"\n!!! could not ressolve the path to the pkl file (found {len(self.paths_to_pkl_file)})\n")
        return ""

        
if __name__=="__main__":
    result_obj = ExperimentResult("Test")
    directories = result_obj.find_results_file()
    print(f"founded results.pkl in {directories}")

    loaded_objs: List[ExperimentResult]  = []
    for dir in directories:
        loaded_objs.append(result_obj.load(f"{dir}/results.pkl"))
        result_obj.update(where_to_look=dir)

    print("==============")
    for obj in loaded_objs:
        print(f"name: {obj.experiment_name}, dir: ./{obj.output_directory}, keys: {obj.base_tfs_dict.keys()}")
        
