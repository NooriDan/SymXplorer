"""
This is the module where core solver logic is implmented. 
Circuit_Solver      (Class)    => solves the generic form of the transfer function for a set of impedance blocks connected
Impedance_Analyzer  (Class)    => sweeps over the possible configuration of each impedance block, computes its filter parameters and 
run_experiment      (function) => A pre-configured application of the two class, handling data fetching, run orders, and report generation
"""
import itertools
from   pprint    import pprint
from   typing    import Dict, List, Tuple, Set, Optional
import sympy
from   sympy     import oo as inf
from   sympy     import symbols, Poly, numer, denom, solve, simplify
from   tqdm      import tqdm

# Custom Imports
from   .domains import Circuit, Impedance_Block, TransmissionMatrix, ExperimentResult
from   .filter  import Filter_Classifier
from   .utils   import FileSave


# ----------  Class Definitions  ----------
class Circuit_Solver:
    """Finds the base transfer function"""
    def __init__(self, 
                circuit:    Circuit,
                _output:    List[sympy.Basic], 
                _input:     List[sympy.Basic],
                transmissionMatrixType: str,
                transmissionMatrix: TransmissionMatrix = TransmissionMatrix() # Default T matrix (all symbolic)
                ):
        # Extract information from the Circuit object
        self.equations:  List[sympy.Basic]  = circuit.nodal_equations
        self.solveFor:   List[sympy.Basic]  = circuit.solve_for
        self.impedances: List[Impedance_Block]    = circuit.impedances
        self.impedancesToDisconnect:    List[sympy.Basic] = [impedance.symbol for impedance in circuit.impedances]

        # Solver specific variables
        self.output:     List[sympy.Basic]  = _output
        self.input:      List[sympy.Basic]  = _input
        self.T_type:     str                = transmissionMatrixType
        self.T_analysis: TransmissionMatrix = transmissionMatrix  

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
        print(f"1 - Solving for ({oPos} - {oNeg}) / ({iPos} - {iNeg})")
        print(f"2 - Intermediate Variables: {self.solveFor}")
        print(" ----------------------------")

        # Solve for generic transfer function
        solutions = solve(self.equations, self.solveFor, dict=True)
        print("3 - solved the base transfer function (symbolic [T])")
        # print(solutions)
        

        if solutions:
            solution = solutions[0]  # Take the first solution
            print("4 - FOUND THE BASE TF")
            baseHs: sympy.Basic = (solution.get(oPos, oPos) - solution.get(oNeg, oNeg)) / (solution.get(iPos, iPos) - solution.get(iNeg, iNeg))
            baseHs = sympy.cancel((baseHs.factor()))
            self.baseSymbolicHs = baseHs
            return baseHs
        print("!!!!!! COULD NOT SOLVE THE NODAL EQUATIONS !!!!!!")
    
    def _solveForAllImpedancesConnected(self):
        print(f"*** --- Solving for T type: {self.T_type} --- ***")
        sub_dict = {
            self.T_analysis.get_element(0, 0, "symbolic"): self.T_analysis.get_element(0, 0, self.T_type),
            self.T_analysis.get_element(0, 1, "symbolic"): self.T_analysis.get_element(0, 1, self.T_type),
            self.T_analysis.get_element(1, 0, "symbolic"): self.T_analysis.get_element(1, 0, self.T_type),
            self.T_analysis.get_element(1, 1, "symbolic"): self.T_analysis.get_element(1, 1, self.T_type)
        }

        Hs = self.baseSymbolicHs.subs(sub_dict)  # Substitute the impedance values into the base function
        Hs = sympy.cancel(Hs.factor())  # Simplify and factor the resulting expression
        self.baseHs = Hs
        print(f"*** --- Done --- ***")
        return Hs
    
    def _setPossibleImpedanceConnections(self):
        print(f"--- Computing the possible impedance connections for {self.impedancesToDisconnect} ---")
        variables = self.impedancesToDisconnect

        self.impedanceConnections = []

        # Loop through different combination sizes (from 1 to len(variables))
        for combination_size in range(1, len(variables) + 1):
            # Generate combinations of the specified size
            combinations = itertools.combinations(variables, combination_size)
            # print(f"combination size = {combination_size}")
            myDict = {}
            for comb in combinations:
                # print(f"processing: {comb}")
                key = '_'.join(str(var).replace('_', '') for var in comb)
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

class Impedance_Analyzer:
    """Main class putting everything together"""
    def __init__(self, _experimentName: str, circuit_solver: Circuit_Solver):
         
        self.experimentName:    str = _experimentName
        self.circuit_solver:    Circuit_Solver = circuit_solver
        self.impedance_blocks:  List[Impedance_Block] = circuit_solver.impedances
        self.impedance_symbols: List[sympy.Basic]     = circuit_solver.impedancesToDisconnect

        self.classifier: Filter_Classifier = Filter_Classifier()
        self.fileSave:   FileSave = FileSave(outputDirectory=f"Runs/{self.experimentName}")

    def isCircuitSolved(self):
        return self.circuit_solver.isSolved()
    
    def getComboKeys(self):
        return self.circuit_solver.baseHsDict.keys()
    
    def getZcombos(self):
        results = {}
        for combos in self.circuit_solver.impedanceConnections:
            for key, array in combos.items():
                myList = []
                for i, zi in enumerate(self.circuit_solver.impedancesToDisconnect):
                    if zi in array:
                        myList.append(self.impedance_blocks[i].allowedConnections)
                    else:
                        myList.append([inf])

                results[key] = (itertools.product(*myList))
        
        print("Zcombos")
        return results
    
    def computeTransferFunction(self, baseHs, zCombo):

        sub_dict: Dict[sympy.Symbol, sympy.Basic] = {}
        for i, impedance  in enumerate(self.impedance_symbols):
            sub_dict[impedance] = zCombo[i]   # Assumes zCombo is ordered the same as circuit.impedance_symbols

        Hs = baseHs 
        # if sub_dict[key] != inf:
        Hs = Hs.subs(sub_dict)  # Substitute the impedance values into the base function

        Hs = sympy.together(Hs)
        
        # Extract the numerator and denominator as polynomials
        s = symbols("s")
        try:
            Hs_num = Poly(numer(Hs), s)
            Hs_den = Poly(denom(Hs), s)
            
            Hs = Hs_num/Hs_den
            
        except sympy.PolynomialError:
            Hs = sympy.simplify(Hs)
            # print(f"Polynomial error when computing {Hs}")

        return Hs
    
    def computeTFs(self, comboKey="all", clearRecord=True):
        # Clear previous records if necessary

        print(f"combo key = {comboKey}")
        self.classifier.transferFunctionsList = []
        self.classifier.impedanceList = []
        # self.classifier.clearFilter()

        # Ensure the comboKey is valid
        try:
            impedanceBatch = list(self.getZcombos()[comboKey])
        except KeyError:
            raise ValueError(f"Invalid comboKey '{comboKey}' provided.")

        # Prepare the base transfer function
        baseHs = self.circuit_solver.baseHsDict.get(comboKey)
        if baseHs is None:
            raise ValueError(f"BaseHs for the comboKey '{comboKey}' is not found.")

        # Use list comprehension for efficient transfer function computation
        solvedTFs = [
            self.computeTransferFunction(baseHs, zCombo)
            for zCombo in tqdm(impedanceBatch, desc="Getting the TFs (CG)", unit="combo")
        ]
        
        # Add the computed transfer functions to the classifier
        self.classifier.addTFs(solvedTFs, impedanceBatch)

        # Output the number of transfer functions computed
        print(f"Number of transfer functions found: {len(solvedTFs)}")

    # Reporting methods (generate pdf)
    def reportAll(self, experimentName, Z_arr):
        self.fileSave.generateLaTeXReport(self.classifier.classifications, 
                            output_filename= f"{experimentName}_{Z_arr}_all",
                            subFolder=f"{experimentName}_{Z_arr}")
    
    def reportType(self, fType, experimentName, Z_arr):
        self.fileSave.generateLaTeXReport(self.classifier.clusteredByType[fType], 
                            output_filename= f"{experimentName}_{Z_arr}_{fType}",
                            subFolder=f"{experimentName}_{Z_arr}")
    
    def reportSummary(self, experimentName, Z_arr):
        if(self.classifier.clusteredByType):
            if(self.circuit_solver.baseHsDict.get(Z_arr)):
                self.fileSave.generateLaTeXSummary(self.circuit_solver.baseHsDict[Z_arr], 
                                                    self.classifier.clusteredByType,
                                                    output_filename= f"{experimentName}_{Z_arr}_summary",
                                                    subFolder=f"{experimentName}_{Z_arr}")
            else:
                print("!!! INVALID Z_arr !!!")
        else:
            print("==** Need to first summarize the filters")

    def compilePDF(self):
        self.fileSave.compile()

    def export(self, filename):
        self.fileSave.export(self, filename)

# ----------  Function Definitions  ----------

def solve_circuit(experimentName: str,     # Arbitrary name (affectes where the report is saved)
                    T_type: str,            # Options are "symbolic", "simple", "some-parasitic", and "full-parasitic"
                    circuit: Circuit,       # the circuit object containing the information about the circuit under test (impedance blocks, nodal equations, variable to solve for)
                    outputFrom: List[str],  # Where the output is taken differentially from (TIA -> Vop Von)
                    inputFrom: List[str]    # Where the input is taken differentially from (TIA -> Iip Iin)
                    ) -> Circuit_Solver:
    
    if (len(outputFrom)!= 2) or (len(inputFrom)!=2):
        raise IndexError(f"Exactly two values are required.\ninput_size = {len(inputFrom)}, output_size = {len(outputFrom)}")
    
    # -------- Experiment hyper-parameters --------
    _output = [symbols(sym) for sym in outputFrom]
    _input  = [symbols(sym) for sym in inputFrom]
    experimentName += "_" + T_type  

    # -------- Create a Solver Object --------------

    solver = Circuit_Solver(circuit,
                        _output, 
                        _input,
                        transmissionMatrixType=T_type)
    
    circuit_solver_history = ExperimentResult(experimentName)
    circuit_solver_history.update()

    if circuit_solver_history is None or  (circuit_solver_history.circuit_solver is None ) or (not circuit_solver_history.circuit_solver.isSolved()):
        print(f"Solving the circuit for the first time")
        solver.solve()
        circuit_solver_history.circuit_solver = solver
    else:
        print(f"Loading the circuit solver from past runs")
        solver = circuit_solver_history.circuit_solver
    
    circuit_solver_history.save("results_circuit_solution.pkl")
    
    return solver

def get_impedance_keys_systematic(keys_to_remove: Set[str], solver: Circuit_Solver, minNumOfActiveImpedances: int, maxNumOfActiveImpedances: int)-> Tuple[ List[ Dict[str, Tuple[sympy.Basic]] ], int]:
    # solver.impedanceConnections is of type (List[Dict[str, List[sympy.Basic]]]), 
    # where str is the string representation of the collection of impedances (e.g., "Z1_Z2" : [Z_1, Z_2])
    impedanceKeys = solver.impedanceConnections[(minNumOfActiveImpedances-1):maxNumOfActiveImpedances]
    count_of_new_keys = 0
    for key_len, keypairs in enumerate(solver.impedanceConnections[(minNumOfActiveImpedances-1):maxNumOfActiveImpedances], 0):
        impedanceKeys[key_len] = {}
        for key in keypairs.keys():
            if not (key in keys_to_remove):
                impedanceKeys[key_len][key] = keypairs[key]
                count_of_new_keys += 1
    
    return impedanceKeys, count_of_new_keys

def get_impedance_keys_overwrite(keys_to_remove: Set[str], impedanceKeysOverwrite: List[str]) -> Tuple[List[str], int]:
    impedanceKeys: List[str] = impedanceKeysOverwrite
    # Filter `impedanceKeys`
    impedanceKeys = [key for key in impedanceKeys if key not in keys_to_remove]
    count_of_new_keys: int = len(impedanceKeys)

    return impedanceKeys, count_of_new_keys

def run_experiment(experimentName: str,     # Arbitrary name (affectes where the report is saved)
                    T_type: str,            # Options are "symbolic", "simple", "some-parasitic", and "full-parasitic"
                    circuit: Circuit,      # the circuit object containing the information about the circuit under test (impedance blocks, nodal equations, variable to solve for)
                    minNumOfActiveImpedances: int,      # define the boundries for search (NumOfActiveImpedance = 2 means all Zi_Zj combinations (i≠j))     
                    maxNumOfActiveImpedances: int,
                    impedanceKeysOverwrite: Optional[List[str]],  # If provided overwrites the systematic simulation run
                    outputFrom: List[str],  # Where the output is taken differentially from (TIA -> Vop Von)
                    inputFrom: List[str]    # Where the input is taken differentially from (TIA -> Iip Iin)
                    ) -> ExperimentResult:

    """
    Runs a symbolic experiment on a differential CG circuit based on the specified parameters.

    Args:
        experimentName (str): Arbitrary name (affectes where the report is saved)
        T_type (str): The type of the transmission matrix used. Acceptable values are "symbolic", "simple", "some-parasitic", and "full-parasitic"
        minNumOfActiveImpedances (int): The minimum number of active impedances to consider in the search space. For example, a value of 2 means all combinations of Zi_Zj (i≠j).
        maxNumOfActiveImpedances (int): The maximum number of active impedances to consider in the search space.
        impedanceKeysOverwrite (Optional[List[str]]): A list of impedance keys that, if provided, will override the default systematic simulation. 
                                                      If None, the default simulation is run.
        outputFrom (List[str]): A list of parameters to take as output from the experiment. These are typically voltage or current outputs (e.g., TIA -> Vop Von).
        inputFrom (List[str]): A list of parameters to use as input to the experiment. These are typically voltage or current inputs (e.g., TIA -> Iip Iin).

    Returns:
        None: This function does not return any value. The results of the experiment are typically saved in a report file.
    """

    # -------- get the impedance combinations --------
    solver = solve_circuit(
        experimentName=experimentName,
        T_type=T_type,
        circuit=circuit,
        outputFrom=outputFrom,
        inputFrom=inputFrom
    )

    # -------- get the impedance combinations --------
    experiment_results_history = ExperimentResult(experimentName, circuit_solver=solver)
    experiment_results_history.update() # load previous results (if exists in Run/EXPERIMENT_NAME folder)

    # Get the keys to be removed
    keys_to_remove = set(experiment_results_history.base_tfs_dict.keys())
    print(f"** found {len(keys_to_remove)} keys already computed")

    if not impedanceKeysOverwrite:
        print(f"Performing a systematic search: min_num_of_active_z = {minNumOfActiveImpedances} max_num_of_active_z = {maxNumOfActiveImpedances}")
        impedanceKeys, count_of_new_keys = get_impedance_keys_systematic(
            keys_to_remove=keys_to_remove,
            minNumOfActiveImpedances=minNumOfActiveImpedances,
            maxNumOfActiveImpedances=maxNumOfActiveImpedances
            )

    else:
        impedanceKeys, count_of_new_keys = get_impedance_keys_overwrite(
            keys_to_remove=keys_to_remove,
            impedanceKeysOverwrite=impedanceKeysOverwrite
            )
        print(f"Experiment keys: {impedanceKeys}")

    print(f"Experiment will be ran for {count_of_new_keys} keys: {impedanceKeys}")

    
    # -------- Experiment Loop -----------------------

    if (not impedanceKeysOverwrite):
        for i, dictionary in enumerate(impedanceKeys, 1):
            comboSize       = len(impedanceKeys)
            dictionarySize  = len(dictionary.keys())
            for j, key in enumerate(dictionary.keys(), 1):
                print(f"==> Running the {experimentName} Experiment for {key} (progress: {j}/{dictionarySize} combo size: {i}/{comboSize})\n")
                analysis = Impedance_Analyzer(experimentName, solver)
                analysis.computeTFs(comboKey=key)
                
                analysis.classifier.classifyBiQuadFilters()
                analysis.classifier.summarizeFilterType()

                analysis.reportSummary(experimentName, key)
                analysis.compilePDF()

                experiment_results_history.add_result(key, analysis.circuit_solver.baseHsDict[key], analysis.classifier.classifications)
    else: 
        for i, key in enumerate(impedanceKeys, 1):
            print(f"--> Running the {experimentName} Experiment for {key} ({i}/{len(impedanceKeys)})\n")
            analysis = Impedance_Analyzer(experimentName, solver)
            analysis.computeTFs(comboKey=key)
            
            analysis.classifier.classifyBiQuadFilters()
            analysis.classifier.summarizeFilterType()

            analysis.reportSummary(experimentName, key)
            analysis.compilePDF()

            experiment_results_history.add_result(key, analysis.circuit_solver.baseHsDict[key], analysis.classifier.classifications)

    print("<----> END OF EXPERIMENT <---->")
    if(count_of_new_keys):
        print(f"Impedance Keys analyzed (count: {count_of_new_keys}): ")
        pprint(impedanceKeys)

    experiment_results_history.save()

    return experiment_results_history