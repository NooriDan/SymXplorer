import sympy
from   tqdm      import tqdm
from   sympy     import symbols, Poly, numer, denom, sign
from   sympy     import oo as inf
from   symengine import expand as SE_expand, sympify as SE_sympify
from   itertools import product
from   typing    import Dict, List
from   dataclasses import dataclass

# Custom imports
# from   .global  as GlobalVariables
from   .domains import FilterClassification
from   .circuit import CircuitSolver
from   .filter  import FilterClassifier
from   .utils   import FileSave, Impedance



class SymbolixExperiment:
    """Main class putting everything together"""
    def __init__(self, _experimentName: str, circuit_solver: CircuitSolver):
         
         self.experimentName = _experimentName
         self.circuit_solver: CircuitSolver = circuit_solver
         self.ZZ: List[Impedance] = circuit_solver.impedances

         self.classifier: FilterClassifier = FilterClassifier()
         self.fileSave: FileSave = FileSave(outputDirectory=f"Runs/{self.experimentName}")

         self.transferFunctions: List[sympy.Basic] = []
         self.solvedCombos: List[int]= []
         self.numOfComputes: int = 0

    def isCircuitSolved(self):
        return self.circuit_solver.isSolved()
    
    def getComboKeys(self):
        return self.circuit_solver.baseHsDict.keys()
    
    def getZcombos(self):
        results = {}
        for combos in self.circuit_solver.impedanceConnections:
            for key, array in combos.items():
                myList = []
                for i, zi in enumerate(self.circuit_solver.impedancesToDisconnect + self.circuit_solver.alwaysConnectedImpedances):
                    if zi in array:
                        myList.append(self.ZZ[i].allowedConnections)
                    else:
                        myList.append([inf])

                results[key] = (product(*myList))
        
        return results
    
    def computeTransferFunction(self, baseHs, zCombo):
        _Z1, _Z2, _Z3, _Z4, _Z5, _ZL = zCombo
        s = symbols("s")
        sub_dict = {
            symbols("Z_1"): _Z1,
            symbols("Z_2"): _Z2,
            symbols("Z_3"): _Z3,
            symbols("Z_4"): _Z4,
            symbols("Z_5"): _Z5,
            symbols("Z_L"): _ZL
        }
        Hs = baseHs 
        # if sub_dict[key] != inf:
        Hs = Hs.subs(sub_dict)  # Substitute the impedance values into the base function
        
        # Hs = simplify(Hs.factor())  # Simplify and factor the resulting expression (experimenting showed its not needed and we can achieved higher speed)
        # Hs = SE_sympify(SE_expand(Hs))

        # Handle unsupported terms (replace sign or other functions if present)
        # Hs = Hs.replace(sign, lambda x: 1)  # Replace sign with 1 (adjust as needed)

        Hs = sympy.together(Hs)
        # Extract the numerator and denominator as polynomials
        try:
            Hs_num = Poly(numer(Hs), s)
            Hs_den = Poly(denom(Hs), s)
            
            Hs = Hs_num / Hs_den
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
        self.numOfComputes += 1

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