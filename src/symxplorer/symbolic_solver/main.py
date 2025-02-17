from   pprint         import pprint
import argparse

# Custom Imports
from   symxplorer.symbolic_solver.utils          import clear_terminal, print_specs
from   symxplorer.symbolic_solver.domains        import Circuit
from   symxplorer.symbolic_solver.solver         import run_experiment
from   symxplorer.demo.examples                  import select_circuit  

def get_parser():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Running the symbolic simulation")

    # Add arguments
    parser.add_argument("--name", 
                        type= str,
                        required=False,
                        default="TIA",
                        help="The name of the directory to store the result"
                        )
    
    parser.add_argument("--demoCircuit", 
                        type= str,
                        required=False,
                        choices  = ["CG", "CS", "DIVIDER"],
                        default="CG",
                        help="Select the demo circuit (use --help for options)"
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
                        help="The name of the node properties to take differential output (e.g., for TIA, Vop Von)"
                        )

    parser.add_argument("--input", 
                        required=False,
                        default= ["Iip", "Iin"], # The case for TIA
                        nargs=2,  # Ensure exactly two values are provided
                        help="The name of the node properties to take differential input (e.g., for TIA, Iip Iin)."
                        )
    
    parser.add_argument(
        "--createCSV",
        action="store_true",  # This will set the flag to True if the argument is provided
        help="Flag to create a CSV file.",
    )

    parser.add_argument(
        "--loadHistory",
        action="store_true",  # This will set the flag to True if the argument is provided
        help="Flag to load results of previous runs stored in results.pkl (could be time consuming for large files)",
    )


    # Parse arguments
    args = parser.parse_args()
    return args


def main():

    args = get_parser()
     
    clear_terminal()
    print_specs()

    print("Command line arguments:")
    pprint(args._get_kwargs())
    print("\n")

    # -------- Create a Circuit Object --------------

    circuit = select_circuit(args.demoCircuit)
    if circuit is None:
        print(f"Selected demo circuit ({args.demoCircuit}) cannot be ressolved :(")
        return None
    args.name = f"{args.demoCircuit}_{args.name}"

    # --------    Run the experiment   --------------
    results = run_experiment(experimentName = args.name, 
                            T_type = args.type, 
                            circuit= circuit,
                            minNumOfActiveImpedances = args.minNumOfActiveImpedances, 
                            maxNumOfActiveImpedances = args.maxNumOfActiveImpedances,
                            impedanceKeysOverwrite   = args.impedanceKeys,
                            outputFrom = args.output,
                            inputFrom  = args.input,
                            loadHistory = args.loadHistory)
    
    if args.createCSV:
        print("Creating CSV file...")
        results.to_csv()
    else:
        print("CSV creation not requested.")

if __name__ == "__main__":
    main()