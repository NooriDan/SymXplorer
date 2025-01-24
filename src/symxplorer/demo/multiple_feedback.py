
from sympy import symbols, Eq
from dataclasses import dataclass
# Cusom Imports
from symxplorer.symbolic_solver.domains   import Circuit, Impedance_Block


# Example 1 -- Customized Current Mode Multiple Feedback Filter
# ===================================================================
@dataclass(frozen=True)
class Customized_Current_Mode_Multiple_Feedback:
    """Op-Amp circuit Input take single-endedly from ZL"""
    # (1) Define possible impedances (assign names)
    # ---------------------------------------
    # (1.1) Declare the impedance blocks in the problem
    z1_block = Impedance_Block("1")
    z2_block = Impedance_Block("2")
    z3_block = Impedance_Block("3")
    z4_block = Impedance_Block("4")
    z5_block = Impedance_Block("5")
    z6_block = Impedance_Block("6")

    # (1.2) Define the possible impedance types for each block AND assign it by calling self.setAllowedImpedanceConnections
    z1_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z1_block.setAllowedImpedanceConnections(z1_possible_combinations)

    z2_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z2_block.setAllowedImpedanceConnections(z2_possible_combinations)

    z3_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z3_block.setAllowedImpedanceConnections(z3_possible_combinations)

    z4_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z4_block.setAllowedImpedanceConnections(z4_possible_combinations)

    z5_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z5_block.setAllowedImpedanceConnections(z5_possible_combinations)

    z6_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z6_block.setAllowedImpedanceConnections(z6_possible_combinations)

    # (1.3) Store all the declared impedances to be passed to the program
    zz = [z1_block, z2_block, z3_block, z4_block, z5_block, z6_block]

    # End of step (1)
    # ---------------------------------------


    # (2)   NODAL EQUATION DEFINITION
    # ---------------------------------------
    # (2.1) Define symbolic variables (CG)
    Iin          = symbols('Iin')
    Vo1, Vo2, Va = symbols('Vo1 Vo2 Va')

    # (2.2) List all the nodal vairables to be fed into the solver
    solveFor = [
                Vo1, Vo2, Va, # Voltages
                Iin
                ]
    

    # # (2.3) Select a transmission matrix (default = symbolic matrix)
    # T_a = TransmissionMatrix().getTranmissionMatrix()
    # T_b = TransmissionMatrix().getTranmissionMatrix()

    # Need to extract the Z of the impedance block object 
    Z1 = z1_block.symbol
    Z2 = z2_block.symbol
    Z3 = z3_block.symbol
    Z4 = z4_block.symbol
    Z5 = z5_block.symbol
    Z6 = z6_block.symbol


    # (2.4) Define the impedances that can be disconnected in the circuit
    # impedancesToDisconnect = [Z1, Z2, Z4]
    impedancesToDisconnect = [Z1, Z2, Z3, Z4, Z5, Z6]

    # (2.5) Define the Nodal Equations
    nodalEquations = [
                Eq(Iin, (0 - Vo1)/Z1 + (0 - Va)/Z2),
                Eq(0, (Vo1 - Va)/Z3 + (0 - Va)/Z2 + (0 - Va)/Z4 + (Va - 0)/Z5),
                Eq((Va - 0)/Z5, (0 - Vo2)/Z6)
            ]
    # End of step (2)
    # ---------------------------------------

    # Create a circuit instance to store the experiment parameters (in this case for our case study -- a common gate differential amplifier)
    circuit = Circuit(impedances=zz, nodal_equations=nodalEquations, solve_for=solveFor, impedancesToDisconnect=impedancesToDisconnect)
    
    def update_circuit(self) -> Circuit:
        self.circuit = Circuit(impedances=self.zz, nodal_equations=self.nodalEquations, solve_for=self.solveFor, impedancesToDisconnect=self.impedancesToDisconnect)
        return self.circuit
# ===================================================================
# End of Example 1

# Example 1 -- Customized Current Mode Multiple Feedback Filter - added non-ideality
# ===================================================================
@dataclass(frozen=True)
class Customized_Current_Mode_Multiple_Feedback_With_Limitation:
    """Op-Amp circuit Input take single-endedly from ZL"""
    # (1) Define possible impedances (assign names)
    # ---------------------------------------
    # (1.1) Declare the impedance blocks in the problem
    z1_block = Impedance_Block("1")
    z2_block = Impedance_Block("2")
    z3_block = Impedance_Block("3")
    z4_block = Impedance_Block("4")
    z5_block = Impedance_Block("5")
    z6_block = Impedance_Block("6")

    # (1.2) Define the possible impedance types for each block AND assign it by calling self.setAllowedImpedanceConnections
    z1_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z1_block.setAllowedImpedanceConnections(z1_possible_combinations)

    z2_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z2_block.setAllowedImpedanceConnections(z2_possible_combinations)

    z3_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z3_block.setAllowedImpedanceConnections(z3_possible_combinations)

    z4_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z4_block.setAllowedImpedanceConnections(z4_possible_combinations)

    z5_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z5_block.setAllowedImpedanceConnections(z5_possible_combinations)

    z6_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z6_block.setAllowedImpedanceConnections(z6_possible_combinations)

    # (1.3) Store all the declared impedances to be passed to the program
    zz = [z1_block, z2_block, z3_block, z4_block, z5_block, z6_block]

    # End of step (1)
    # ---------------------------------------


    # (2)   NODAL EQUATION DEFINITION
    # ---------------------------------------
    # (2.1) Define symbolic variables (CG)
    Iin          = symbols('Iin')
    Vo1, Vo2, Va, Vn1, Vn2 = symbols('Vo1 Vo2 Va Vn1 Vn2')
    GB = symbols("GB") # Gain bandwidth product of the op-amp
    s  = symbols("s")

    # (2.2) List all the nodal vairables to be fed into the solver
    solveFor = [
                Vo1, Vo2, Va, Vn1, Vn2, # Voltages
                Iin
                ]

    # Need to extract the Z of the impedance block object 
    Z1 = z1_block.symbol
    Z2 = z2_block.symbol
    Z3 = z3_block.symbol
    Z4 = z4_block.symbol
    Z5 = z5_block.symbol
    Z6 = z6_block.symbol


    # (2.4) Define the impedances that can be disconnected in the circuit
    # impedancesToDisconnect = [Z1, Z2, Z4]
    impedancesToDisconnect = [Z1, Z2, Z3, Z4, Z5, Z6]

    # (2.5) Define the Nodal Equations
    nodalEquations = [
                Eq((0 - Vn1) * GB/s, Vo1 ),
                Eq((0 - Vn2) * GB/s, Vo2 ),
                Eq(Iin, (Vn1 - Vo1)/Z1 + (Vn1 - Va)/Z2),
                Eq(0, (Vo1 - Va)/Z3 + (Vn1 - Va)/Z2 + (0 - Va)/Z4 + (Va - Vn2)/Z5),
                Eq((Va - Vn2)/Z5, (Vn2 - Vo2)/Z6)
            ]
    # End of step (2)
    # ---------------------------------------

    # Create a circuit instance to store the experiment parameters (in this case for our case study -- a common gate differential amplifier)
    circuit = Circuit(impedances=zz, nodal_equations=nodalEquations, solve_for=solveFor, impedancesToDisconnect=impedancesToDisconnect)
    
    def update_circuit(self) -> Circuit:
        self.circuit = Circuit(impedances=self.zz, nodal_equations=self.nodalEquations, solve_for=self.solveFor, impedancesToDisconnect=self.impedancesToDisconnect)
        return self.circuit
# ===================================================================
# End of Example 1 - added non-ideality

# Example 2 -- Voltage Mode Multiple Feedback Filter 
# ===================================================================
@dataclass(frozen=True)
class Voltage_Mode_Multiple_Feedback:
    """Voltage-mode multiple-feedback filter Paper by Carlosena et al."""
    # (1) Define possible impedances (assign names)
    # ---------------------------------------
    # (1.1) Declare the impedance blocks in the problem
    z1_block = Impedance_Block("1")
    z2_block = Impedance_Block("2")
    z3_block = Impedance_Block("3")
    z4_block = Impedance_Block("4")
    z5_block = Impedance_Block("5")

    # (1.2) Define the possible impedance types for each block AND assign it by calling self.setAllowedImpedanceConnections
    z1_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z1_block.setAllowedImpedanceConnections(z1_possible_combinations)

    z2_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z2_block.setAllowedImpedanceConnections(z2_possible_combinations)

    z3_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z3_block.setAllowedImpedanceConnections(z3_possible_combinations)

    z4_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z4_block.setAllowedImpedanceConnections(z4_possible_combinations)

    z5_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z5_block.setAllowedImpedanceConnections(z5_possible_combinations)


    # (1.3) Store all the declared impedances to be passed to the program
    zz = [z1_block, z2_block, z3_block, z4_block, z5_block]

    # End of step (1)
    # ---------------------------------------


    # (2)   NODAL EQUATION DEFINITION
    # ---------------------------------------
    # (2.1) Define symbolic variables (CG)
    Iin          = symbols('Iin')
    V1, V2, Va = symbols('V1 V2 Va')

    # (2.2) List all the nodal vairables to be fed into the solver
    solveFor = [
                V1, V2, Va # Voltages
                # Iin
                ]
    

    # # (2.3) Select a transmission matrix (default = symbolic matrix)
    # T_a = TransmissionMatrix().getTranmissionMatrix()
    # T_b = TransmissionMatrix().getTranmissionMatrix()

    # Need to extract the Z of the impedance block object 
    Z1 = z1_block.symbol
    Z2 = z2_block.symbol
    Z3 = z3_block.symbol
    Z4 = z4_block.symbol
    Z5 = z5_block.symbol


    # (2.4) Define the impedances that can be disconnected in the circuit
    impedancesToDisconnect = []

    # (2.5) Define the Nodal Equations
    nodalEquations = [
                Eq((V1-Va)/Z1 + (0 - Va)/Z4 + (0 - Va)/Z3 + (V2-Va)/Z2, 0),
                Eq((Va - 0)/Z3, (0 - V2)/Z5)
            ]
    # End of step (2)
    # ---------------------------------------

    # Create a circuit instance to store the experiment parameters (in this case for our case study -- a common gate differential amplifier)
    circuit = Circuit(impedances=zz, nodal_equations=nodalEquations, solve_for=solveFor, impedancesToDisconnect=impedancesToDisconnect)

    def update_circuit(self) -> Circuit:
        self.circuit = Circuit(impedances=self.zz, nodal_equations=self.nodalEquations, solve_for=self.solveFor, impedancesToDisconnect=self.impedancesToDisconnect)
        return self.circuit
# ===================================================================
# End of Example 2


# Example 3 -- Current Mode Multiple Feedback Filter 
# ===================================================================
@dataclass(frozen=True)
class Current_Mode_Multiple_Feedback:
    """Voltage-mode multiple-feedback filter Paper by Carlosena et al."""
    # (1) Define possible impedances (assign names)
    # ---------------------------------------
    # (1.1) Declare the impedance blocks in the problem
    z1_block = Impedance_Block("1")
    z2_block = Impedance_Block("2")
    z3_block = Impedance_Block("3")
    z4_block = Impedance_Block("4")
    z5_block = Impedance_Block("5")

    # (1.2) Define the possible impedance types for each block AND assign it by calling self.setAllowedImpedanceConnections
    z1_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z1_block.setAllowedImpedanceConnections(z1_possible_combinations)

    z2_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z2_block.setAllowedImpedanceConnections(z2_possible_combinations)

    z3_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z3_block.setAllowedImpedanceConnections(z3_possible_combinations)

    z4_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z4_block.setAllowedImpedanceConnections(z4_possible_combinations)

    z5_possible_combinations =[                                             
                            "R",
                            "C",
                            "R + C",
                            "R | C"
                            ]
    z5_block.setAllowedImpedanceConnections(z5_possible_combinations)


    # (1.3) Store all the declared impedances to be passed to the program
    zz = [z1_block, z2_block, z3_block, z4_block, z5_block]

    # End of step (1)
    # ---------------------------------------


    # (2)   NODAL EQUATION DEFINITION
    # ---------------------------------------
    # (2.1) Define symbolic variables (CG)
    I1, I2     = symbols('I1 I2')
    V1, V2, Va = symbols('V1 V2 Va')

    # (2.2) List all the nodal vairables to be fed into the solver
    solveFor = [
                V1, V2, Va, # Voltages
                I1, I2
                ]
    

    # # (2.3) Select a transmission matrix (default = symbolic matrix)
    # T_a = TransmissionMatrix().getTranmissionMatrix()
    # T_b = TransmissionMatrix().getTranmissionMatrix()

    # Need to extract the Z of the impedance block object 
    Z1 = z1_block.symbol
    Z2 = z2_block.symbol
    Z3 = z3_block.symbol
    Z4 = z4_block.symbol
    Z5 = z5_block.symbol


    # (2.4) Define the impedances that can be disconnected in the circuit
    impedancesToDisconnect = []

    # (2.5) Define the Nodal Equations
    nodalEquations = [
                Eq(I2, (V2 - Va)/Z5 + (V2 - V1)/Z2),
                Eq(I1, (0 - V1)/Z4  + (Va - V1)/Z3 + (V2 - V1)/Z2),
                Eq(I1, (V1 - 0)/Z1),
                Eq(V2, 0)
            ]
    # End of step (2)
    # ---------------------------------------

    # Create a circuit instance to store the experiment parameters (in this case for our case study -- a common gate differential amplifier)
    circuit = Circuit(impedances=zz, nodal_equations=nodalEquations, solve_for=solveFor, impedancesToDisconnect=impedancesToDisconnect)

    def update_circuit(self) -> Circuit:
        self.circuit = Circuit(impedances=self.zz, nodal_equations=self.nodalEquations, solve_for=self.solveFor, impedancesToDisconnect=self.impedancesToDisconnect)
        return self.circuit


# ===================================================================
# End of Example 2
