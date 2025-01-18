import symcircuit.demo.differential            as differential_examples
import symcircuit.demo.multiple_feedback       as multiple_feedback_examples

from symcircuit.symbolic_solver.domains   import Circuit # for type-checking



def select_demo_circuit(circuit_select: str, printflag: bool = False) -> Circuit:
    demo_circuit_dict = {
        # Differential Circuit Examples
        "CG"           : differential_examples.Common_Gate.circuit,
        "CS"           : differential_examples.Common_Source.circuit,
        "DIVIDER"      : differential_examples.Voltage_Divider.circuit,
        # Multiple Feedback Filters (implemented using Op-Amps)
        "CUSTOM-CMMF"  : multiple_feedback_examples.Customized_Current_Mode_Multiple_Feedback.circuit,
        "VMMF"         : multiple_feedback_examples.Voltage_Mode_Multiple_Feedback.circuit,
        "CMMF"         : multiple_feedback_examples.Current_Mode_Multiple_Feedback.circuit
    }

    circuit: Circuit = demo_circuit_dict.get(circuit_select)

    if circuit is None:
        return None
    
    if printflag:
        print(f"=> Circuit ({circuit_select}) can be solved?: {circuit.hasSolution()}") 

    return circuit

if __name__ == "__main__":
    import datetime, tqdm

    start = datetime.datetime.now()
    for i in tqdm.tqdm(range(50000000)):
        select_demo_circuit("CG")
    end = datetime.datetime.now()
    duration = (end - start)

    print(f"duration {duration} after {i} runs")