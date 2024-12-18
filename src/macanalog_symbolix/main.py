from   sympy          import symbols
from   pprint         import pprint
from   typing         import List, Optional
import argparse

# Custom Imports
from   macanalog_symbolix.utils          import clear_terminal, print_specs
from   macanalog_symbolix.circuit        import CircuitSolver, Circuit
from   macanalog_symbolix.experiment     import SymbolixExperiment
from   macanalog_symbolix.common_gate_setup   import common_gate_circuit # Used as demo data

def run_cg_experiment(experimentName: str,   # Arbitrary name (affectes where the report is saved)
                      T_type: str,        # Options are "symbolic", "simple", "some-parasitic", and "full-parasitic"
                      minNumOfActiveImpedances: int,      # define the boundries for search (NumOfActiveImpedance = 2 means all Zi_Zj combinations (i≠j))     
                      maxNumOfActiveImpedances: int,
                      impedanceKeysOverwrite: Optional[List[str]],  # If provided overwrites the systematic simulation run
                      outputFrom: List[str],  # Where the output is taken differentially from (TIA -> Vop Von)
                      inputFrom: List[str]    # Where the input is taken differentially from (TIA -> Iip Iin)
                      ) -> None:

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

    if (len(outputFrom)!= 2) or (len(inputFrom)!=2):
        raise argparse.ArgumentTypeError(f"Exactly two values are required.\ninput_size = {len(inputFrom)}, output_size = {len(outputFrom)}")
    
    # -------- Experiment hyper-parameters --------
    _output = [symbols(sym) for sym in outputFrom]
    _input  = [symbols(sym) for sym in inputFrom]
    experimentName += "_" + T_type  

    # -------- Create a Circuit Object --------------

    circuit = Circuit(nodal_equations  = common_gate_circuit.nodal_equations,
                    solve_for        = common_gate_circuit.solve_for,
                    impedances = common_gate_circuit.impedances)

    print(f"=> Circuit can be solved?: {circuit.hasSolution()}")     


    # -------- Create a Solver Object --------------

    solver = CircuitSolver( circuit,
                        _output, 
                        _input,
                        transmissionMatrixType=T_type)
    solver.solve()

    # -------- get the impedance combinations --------

    if not impedanceKeysOverwrite:
        # solver.impedanceConnections is of type (List[Dict[str, List[sympy.Basic]]]), 
        # where str is the string representation of the collection of impedances (e.g., "Z1_Z2" : [Z_1, Z_2])
        impedanceKeys = solver.impedanceConnections
        impedanceKeys = impedanceKeys[(minNumOfActiveImpedances-1):maxNumOfActiveImpedances]

    else:
        impedanceKeys = impedanceKeysOverwrite


    # -------- Experiment Loop -----------------------


    if (not impedanceKeysOverwrite):
        for i, dictionary in enumerate(impedanceKeys, 1):
            comboSize       = len(impedanceKeys)
            dictionarySize  = len(dictionary.keys())
            for j, key in enumerate(dictionary.keys(), 1):
                print(f"==> Running the {experimentName} Experiment for {key} (progress: {j}/{dictionarySize} combo size: {i}/{comboSize})\n")
                experiment = SymbolixExperiment(experimentName, solver)
                experiment.computeTFs(comboKey=key)
                
                experiment.classifier.classifyBiQuadFilters()
                experiment.classifier.summarizeFilterType()

                experiment.reportSummary(experimentName, key)
                experiment.compilePDF()
    else: 
        for i, key in enumerate(impedanceKeys, 1):
            print(f"--> Running the {experimentName} Experiment for {key} ({i}/{len(impedanceKeys)})\n")
            experiment = SymbolixExperiment(experimentName, solver)
            experiment.computeTFs(comboKey=key)
            
            experiment.classifier.classifyBiQuadFilters()
            experiment.classifier.summarizeFilterType()

            experiment.reportSummary(experimentName, key)
            experiment.compilePDF()

    print("<----> END OF EXPERIMENT <---->")
    print("Impedance Keys analyzed: ")
    pprint(impedanceKeys)


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Running the symbolic simulation for common-gate circuit, for a set of impedance connections")

    # Add arguments
    parser.add_argument("--name", 
                        type= str,
                        required=False,
                        default="TIA",
                        help="The name of the directory to store the result"
                        )
    
    parser.add_argument( "--type",
                        type= str,
                        required=False,
                        choices  = ["simple", "symbolic", "some-parasitic", "full-parasitic"],  # Restrict to specific choices
                        default  = "simple",
                        help     = "Specify the type of transmission matrix to use. Choices: simple, symbolic, some-parasitic, full-parasitic"
                        )

    parser.add_argument( "--minNumOfActiveImpedances",
                        required=False,
                        type = int,  # Ensure the input is an integer
                        default = 2,
                        help = "minimum number of impedances in a combination."
                        )
    
    parser.add_argument( "--maxNumOfActiveImpedances",
                        required=False,
                        type = int,  # Ensure the input is an integer
                        default = 3,
                        help = "maximum number of impedances in a combination."
                        )
    
    parser.add_argument("--impedanceKeys", 
                        required=False,
                        nargs="+", 
                        help="List of impedances to examine seperated by space (e.g. Z1_Z2 Z2_Z3_ZL ...)."
                        )
    
    parser.add_argument("--output", 
                        required=False,
                        default= ["Vop", "Von"], # The case for TIA
                        nargs=2,  # Ensure exactly two values are provided
                        help="List of impedances to examine seperated by space (e.g. Z1_Z2 Z2_Z3_ZL ...)."
                        )

    parser.add_argument("--input", 
                        required=False,
                        default= ["Iip", "Iin"], # The case for TIA
                        nargs=2,  # Ensure exactly two values are provided
                        help="List of impedances to examine seperated by space (e.g. Z1_Z2 Z2_Z3_ZL ...)."
                        )

    # Parse arguments
    args = parser.parse_args()

    
    clear_terminal()
    print_specs()

    print("Command line arguments:")
    pprint(args._get_kwargs())
    print("\n")

    # Use arguments
    run_cg_experiment(experimentName = args.name, 
                      T_type = args.type, 
                      minNumOfActiveImpedances = args.minNumOfActiveImpedances, 
                      maxNumOfActiveImpedances = args.maxNumOfActiveImpedances,
                      impedanceKeysOverwrite   = args.impedanceKeys,
                      outputFrom = args.output,
                      inputFrom  = args.input)

if __name__ == "__main__":
    main()