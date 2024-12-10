from   sympy          import symbols
# Custom Imports
from   Utils          import clear_terminal, print_specs
import Global       as GlobalVariables              # Global vairables
from   Circuit  import CircuitSolver, Circuit
from   Experiment     import SymbolixExperimentCG

clear_terminal()
print_specs()

# -------- Experiment hyper-parameters --------
_output = [symbols("Vop"), symbols("Von")]
_input  = [symbols("Iip"), symbols("Iin")]
T_type  = "simple"              # options are "symbolic", "simple", "some-parasitic", and "full-parasitic"
experimentName = "TIA"          # Arbitrary name (affectes where the report is saved)
experimentName += "_" + T_type  
# define the boundries for search (NumOfActiveImpedance = 2 means all Zi_Zj combinations (i≠j))
minNumOfActiveImpedances = 4    
maxNumOfActiveImpedances = 5

# -------- Create a Circuit Object --------------

circuit = Circuit(nodalEquations  = GlobalVariables.nodalEquations,
                  solveFor        = GlobalVariables.solveFor,
                  impedanceBlocks = GlobalVariables.zz)

print(f"=> Circuit can be solved?: {circuit.hasSolution()}")     


# -------- Create a Solver Object --------------

solver = CircuitSolver( circuit,
                       _output, 
                       _input,
                       transmissionMatrixType=T_type,
                       impedancesToDisconnect=GlobalVariables.impedancesToDisconnect)
solver.solve()


# -------- get the impedance combinations --------
# circuit.impedanceConnections (List[Dict[str, List[sympy.Basic]]])
impedanceKeys = solver.impedanceConnections
impedanceKeys = impedanceKeys[(minNumOfActiveImpedances-1):maxNumOfActiveImpedances]


overwrite = False
# Uncomment to enforce manual keys by adding to the list (set overwrite = True)
# impedanceKeys = ["Z1_ZL"] 
# impedanceKeys = [
#             "Z1_ZL",
#             "Z2_ZL",
#             "Z3_ZL",
#             "Z4_ZL",
#             "Z5_ZL",
#             "Z1_Z2_ZL",
#             "Z1_Z3_ZL",
#             "Z1_Z4_ZL",
#             "Z1_Z5_ZL",
#             "Z2_Z3_ZL",
#             "Z2_Z4_ZL",
#             "Z2_Z5_ZL",
#             "Z3_Z4_ZL",
#             "Z3_Z5_ZL",
#             "Z4_Z5_ZL"]

# -------- Experiment Loop -----------------------


if (not overwrite):
    for i, dictionary in enumerate(impedanceKeys, 1):
        comboSize       = len(impedanceKeys)
        dictionarySize  = len(dictionary.keys())
        for j, key in enumerate(dictionary.keys(), 1):
            print(f"==> Running the {experimentName} Experiment for {key} (progress: {j}/{dictionarySize} combo size: {i}/{comboSize})\n")
            experiment = SymbolixExperimentCG(experimentName, solver)
            experiment.computeTFs(comboKey=key)
            
            experiment.classifier.classifyBiQuadFilters()
            experiment.classifier.summarizeFilterType()

            experiment.reportSummary(experimentName, key)
            experiment.compilePDF()
else: 
    for i, key in enumerate(impedanceKeys, 1):
        print(f"--> Running the {experimentName} Experiment for {key} ({i}/{len(impedanceKeys)})\n")
        experiment = SymbolixExperimentCG(experimentName, solver)
        experiment.computeTFs(comboKey=key)
        
        experiment.classifier.classifyBiQuadFilters()
        experiment.classifier.summarizeFilterType()

        experiment.reportSummary(experimentName, key)
        experiment.compilePDF()

print("<----> END OF EXPERIMENT <---->")
print("Impedance Keys are: ")
print(impedanceKeys)