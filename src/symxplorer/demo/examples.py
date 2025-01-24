import symxplorer.demo.differential            as differential_examples
import symxplorer.demo.multiple_feedback       as multiple_feedback_examples
import symxplorer.demo.sallen_key              as sallen_key_examples
import symxplorer.demo.dual_amplifier          as dual_amplifier_examples

from symxplorer.symbolic_solver.domains   import Circuit # for type-checking



def select_circuit(circuit_select: str, printflag: bool = False) -> Circuit:
    demo_circuit_dict = {
        # Differential Circuit Examples
        "CG"           : differential_examples.Common_Gate.circuit,
        "CS"           : differential_examples.Common_Source.circuit,
        "DIVIDER"      : differential_examples.Voltage_Divider.circuit,
        # Multiple Feedback Filters (implemented using Op-Amps)
        "CUSTOM-CMMF"  : multiple_feedback_examples.Customized_Current_Mode_Multiple_Feedback.circuit,
        "CUSTOM-CMMF-BW-LIMITATION": multiple_feedback_examples.Customized_Current_Mode_Multiple_Feedback_With_Limitation.circuit,
        "VMMF"         : multiple_feedback_examples.Voltage_Mode_Multiple_Feedback.circuit,
        "CMMF"         : multiple_feedback_examples.Current_Mode_Multiple_Feedback.circuit,
        # Sallen-Key Filters Topology
        "SK-IDEAL"     : sallen_key_examples.Sallen_Key_Ideal_Op_Amp.circuit,
        "SK-GB-LIMITED": sallen_key_examples.Sallen_Key_GBW_limited_Op_Amp.circuit,
        # Dual Amplifier Band
        "DA-IDEAL"     : dual_amplifier_examples.Dual_Amplifier_Ideal_Op_Amp.circuit,
        "DA-GB-LIMITED": dual_amplifier_examples.Dual_Amplifier_GBW_limited_Op_Amp.circuit
    }

    circuit: Circuit = demo_circuit_dict.get(circuit_select)

    if circuit is None:
        raise KeyError(f"{circuit_select} is not a valid key. choose from {demo_circuit_dict.keys()}")
        return None
    
    if printflag:
        print(f"=> Circuit ({circuit_select}) can be solved?: {circuit.hasSolution()}") 

    return circuit

if __name__ == "__main__":
    import datetime, tqdm

    start = datetime.datetime.now()
    for i in tqdm.tqdm(range(50000000)):
        select_circuit("CG")
    end = datetime.datetime.now()
    duration = (end - start)

    print(f"duration {duration} after {i} runs")