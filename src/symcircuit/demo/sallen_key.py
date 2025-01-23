from sympy import symbols, Eq
from dataclasses import dataclass
# Cusom Imports
from symcircuit.symbolic_solver.domains   import Circuit, Impedance_Block


# Example 1 -- Sallen-Key with ideal op-amp (not GBW limitation)
# ===================================================================
@dataclass(frozen=True)
class Sallen_Key_Ideal_Op_Amp:
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
                            "R"
                            ]
    z5_block.setAllowedImpedanceConnections(z5_possible_combinations)

    z6_possible_combinations =[                                             
                            "R"
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
    Vin, Vn, Vp, Va, Vo = symbols('Vin Vn Vp Va Vo')

    # (2.2) List all the nodal vairables to be fed into the solver
    solveFor = [
                Vin,Vn, Vp, Va, Vo, # Voltages
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
                Eq(Vn, Vp),
                Eq(Iin, (Vin - Va)/Z1),
                Eq(Iin, (Va-Vo)/Z3 + (Va-Vp)/Z4),
                Eq((Va - Vp)/Z2, (Vp - 0)/Z4),
                Eq((Va - Vo)/Z3, (Vo - Vn)/Z6),
                Eq((Vn - 0)/Z5, (Vo - Vn)/Z6)
            ]
    # End of step (2)
    # ---------------------------------------

    # Create a circuit instance to store the experiment parameters (in this case for our case study -- a sallen-key topology)
    circuit = Circuit(impedances=zz, nodal_equations=nodalEquations, solve_for=solveFor, impedancesToDisconnect=impedancesToDisconnect)
    
    def update_circuit(self) -> Circuit:
        self.circuit = Circuit(impedances=self.zz, nodal_equations=self.nodalEquations, solve_for=self.solveFor, impedancesToDisconnect=self.impedancesToDisconnect)
        return self.circuit
# ===================================================================
# End of Example 1

# Example 2 -- Sallen-Key with GBW limitation of the op-amp
# ===================================================================
@dataclass(frozen=True)
class Sallen_Key_GBW_limited_Op_Amp:
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
                            "R"
                            ]
    z5_block.setAllowedImpedanceConnections(z5_possible_combinations)

    z6_possible_combinations =[                                             
                            "R"
                            ]
    z6_block.setAllowedImpedanceConnections(z6_possible_combinations)

    # (1.3) Store all the declared impedances to be passed to the program
    zz = [z1_block, z2_block, z3_block, z4_block, z5_block, z6_block]

    # End of step (1)
    # ---------------------------------------


    # (2)   NODAL EQUATION DEFINITION
    # ---------------------------------------
    # (2.1) Define symbolic variables
    Iin          = symbols('Iin')
    Vin, Vn, Vp, Va, Vo = symbols('Vin Vn Vp Va Vo')

    # (2.2) List all the nodal vairables to be fed into the solver
    solveFor = [
                Vin,Vn, Vp, Va, Vo, # Voltages
                Iin
                ]
    
    # Symbolic variables to model the non-ideal op-amp
    GB = symbols("GB", real=True, positive=True)
    s  = symbols("s")

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
                Eq((Vp - Vn)* GB/s, Vo),
                Eq(Iin, (Vin - Va)/Z1),
                Eq(Iin, (Va-Vo)/Z3 + (Va-Vp)/Z4),
                Eq((Va - Vp)/Z2, (Vp - 0)/Z4),
                Eq((Va - Vo)/Z3, (Vo - Vn)/Z6),
                Eq((Vn - 0)/Z5, (Vo - Vn)/Z6)
            ]
    # End of step (2)
    # ---------------------------------------

    # Create a circuit instance to store the experiment parameters (in this case for our case study -- a sallen-key topology)
    circuit = Circuit(impedances=zz, nodal_equations=nodalEquations, solve_for=solveFor, impedancesToDisconnect=impedancesToDisconnect)
    
    def update_circuit(self) -> Circuit:
        self.circuit = Circuit(impedances=self.zz, nodal_equations=self.nodalEquations, solve_for=self.solveFor, impedancesToDisconnect=self.impedancesToDisconnect)
        return self.circuit
# ===================================================================
# End of Example 2