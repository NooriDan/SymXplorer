"""
Module Name: Example experiment setup to perform symbolic analysis

Dependencies:
- `sympy`: Used for defining the variables in 'nodal_equations'.
- '.domains': Defines the Experiment setup dataclass

Usage:
- This module is used to demonstrate a usecase of MacAnalog_Symbolix on a common gate differential amplifier.

Author: [Danial NZ]
Date: [Dec 2024]

"""

from sympy import symbols, Eq
from dataclasses import dataclass
# Cusom Imports
from symcircuit.symbolic_solver.domains   import Circuit, Impedance_Block, TransmissionMatrix


# Example 1 -- Common Gate Differential Circuit
# ===================================================================
@dataclass(frozen=True)
class Common_Gate:
    # (1) Define possible impedances (assign names)
    # ---------------------------------------
    # (1.1) Declare the impedance blocks in the problem
    z1_block = Impedance_Block("1")
    z2_block = Impedance_Block("2")
    z3_block = Impedance_Block("3")
    z4_block = Impedance_Block("4")
    z5_block = Impedance_Block("5")
    zL_block = Impedance_Block("L")

    # (1.2) Define the possible impedance types for each block AND assign it by calling self.setAllowedImpedanceConnections
    z1_possible_combinations =[                                             
                            "R",
                            "L",
                            "C",
                            "R | C",
                            "R + C",
                            "L + C",
                            "L | C",
                            "R + L + C",
                            "R | L | C",
                            "R + (L | C)",
                            "R | (L + C)"
                            ]
    z1_block.setAllowedImpedanceConnections(z1_possible_combinations)


    z2_possible_combinations = [
                            "R",
                            "C",
                            "R | C",
                            "R + C",
                            "L + C",
                            "R + L + C",
                            "R + (L | C)",
                            "R | (L + C)"   
                            ]
    z2_block.setAllowedImpedanceConnections(z2_possible_combinations)


    z3_possible_combinations = [
                            "R",
                            "C",
                            "R | C",
                            "R + C",
                            "L + C",
                            "L | C",
                            "R + L + C",
                            "R | L | C",
                            "R + (L | C)",
                            "R | (L + C)"
                            ]
    z3_block.setAllowedImpedanceConnections(z3_possible_combinations)


    z4_possible_combinations = [
                            "R",
                            "C",
                            "R | C",
                            "R + C",
                            "L + C",
                            "L | C",
                            "R + L + C",
                            "R | L | C",
                            "R + (L | C)",
                            "R | (L + C)"
                            ]
    z4_block.setAllowedImpedanceConnections(z4_possible_combinations)


    z5_possible_combinations = [
                            "R",
                            "C",
                            "R | C",
                            "R + C",
                            "L + C",
                            "L | C",
                            "R + L + C",
                            "R | L | C",
                            "R + (L | C)",
                            "R | (L + C)"
                            ]
    z5_block.setAllowedImpedanceConnections(z5_possible_combinations)


    zL_possible_combinations = [
                            "R",
                            "C",
                            "R | C",
                            "R + C",
                            "L + C",
                            "L | C",
                            "R + L + C",
                            "R | L | C",
                            "R + (L | C)",
                            "R | (L + C)"
                            ]
    zL_block.setAllowedImpedanceConnections(zL_possible_combinations)

    # (1.3) Store all the declared impedances to be passed to the program
    zz = [z1_block, z2_block, z3_block, z4_block, z5_block, zL_block]

    # End of step (1)
    # ---------------------------------------



    # (2)   NODAL EQUATION DEFINITION
    # ---------------------------------------
    # (2.1) Define symbolic variables (CG)
    Iip, Iin, I1a, I1b, I2a, I2b                       = symbols('Iip Iin I1a I1b I2a I2b')
    Vin, V2a, V2b, V1a, V1b, Va, Vb, Von, Vop, Vip, Vx = symbols('Vin V2a V2b V1a V1b Va Vb Von Vop Vip Vx')

    # (2.2) List all the nodal vairables to be fed into the solver
    solveFor = [
                Vin, V2a, V2b, V1a, V1b, Va, Vb, Von, Vop, Vip, Vx, # Voltages
                Iip, Iin, I1a, I1b, I2a, I2b                        # Currents
                ]

    # (2.3) Select a transmission matrix (default = symbolic matrix)
    T_a = TransmissionMatrix().getTranmissionMatrix()
    T_b = TransmissionMatrix().getTranmissionMatrix()

    # Need to extract the Z of the impedance block object 
    Z1 = z1_block.symbol
    Z2 = z2_block.symbol
    Z3 = z3_block.symbol
    Z4 = z4_block.symbol
    Z5 = z5_block.symbol
    ZL = zL_block.symbol

    # (2.4) Define the impedances that can be disconnected in the circuit (open circuited)
    impedancesToDisconnect = [Z1, Z2, Z3, Z4, Z5, ZL]

    # (2.5) Define the Nodal Equations
    nodalEquations = [
                # 4a
                Eq(0, (Iip + I1a + I2a + (0 - Vip)/Z1 + (Vop - Vip)/Z2 + (Von - Vip)/Z5)),
                # 4b
                Eq(0, (Iin + I1b + I2b + (0 - Vin)/Z1 + (Von - Vin)/Z2 + (Vop - Vin)/Z5)),
                # 4c
                Eq(I2a, ((Vip - Vop)/Z2 + (Vx - Vop)/Z3 + (Von - Vop)/Z4 + (Vin - Vop)/Z5 + (0 - Vop)/ZL)),
                # 4d
                Eq(I2b, ((Vin - Von)/Z2 + (Vx - Von)/Z3 + (Vop - Von)/Z4 + (Vip - Von)/Z5 + (0 - Von)/ZL )),
                # unranked equation in the paper (between 4d and 4e)
                Eq(I1a, ((Vop - Vx)/Z3 + (Von - Vx)/Z3 - I1b)),
                # 4e
                Eq(Vop, Vip + V2a),
                Eq(Von, Vin + V2b),
                # 4f
                Eq(Vip, Vx - V1a),
                Eq(Vin, Vx - V1b),
                # 3g
                Eq(V1a, T_a[0,0]*V2a - T_a[0,1]*I2a),
                Eq(V1b, T_b[0,0]*V2b - T_b[0,1]*I2b),
                # 3h
                Eq(I1a, T_a[1,0]*V2a + T_a[1,1]*I2a),
                Eq(I1b, T_b[1,0]*V2b + T_b[1,1]*I2b)
        ]
    # End of step (2)
    # ---------------------------------------

    # Create a circuit instance to store the experiment parameters (in this case for our case study -- a common gate differential amplifier)
    circuit = Circuit(impedances=zz, nodal_equations=nodalEquations, solve_for=solveFor, impedancesToDisconnect=impedancesToDisconnect)

    def update_circuit(self) -> Circuit:
        self.circuit = Circuit(impedances=self.zz, nodal_equations=self.nodalEquations, solve_for=self.solveFor, impedancesToDisconnect=self.impedancesToDisconnect)
        return self.circuit
# ===================================================================
# End of Example 1 -- Common Gate Differential Circuit



# Example 2 -- Common Source Differential Circuit
# ===================================================================
@dataclass(frozen=True)
class Common_Source:
    """The Equation and Impedance definition of differential Common Gate Amplifier"""
    # (1) Define possible impedances (assign names)
    # ---------------------------------------
    # (1.1) Declare the impedance blocks in the problem
    z1_block = Impedance_Block("1")
    z2_block = Impedance_Block("2")
    z3_block = Impedance_Block("3")
    z4_block = Impedance_Block("4")
    z5_block = Impedance_Block("5")
    zL_block = Impedance_Block("L")
    zS_block = Impedance_Block("S")

    # (1.2) Define the possible impedance types for each block AND assign it by calling self.setAllowedImpedanceConnections
    z1_possible_combinations =[                                             
                            "R",
                            "C",
                            "R | C",
                            "R + C",
                            "L + C"
                            ]
    z1_block.setAllowedImpedanceConnections(z1_possible_combinations)


    z2_possible_combinations = [                                             
                            "R",
                            "C",
                            "R | C",
                            "R + C",
                            "L + C"
                            ]
    z2_block.setAllowedImpedanceConnections(z2_possible_combinations)


    z3_possible_combinations = [
                            "R",
                            "C",
                            "L",
                            "R | C",
                            "R + C",
                            "L + C",
                            "L | C",
                            ]
    z3_block.setAllowedImpedanceConnections(z3_possible_combinations)


    z4_possible_combinations = [
                            "R",
                            "C",
                            "L",
                            "R | C",
                            "R + C",
                            "L + C",
                            "L | C",
                            ]
    z4_block.setAllowedImpedanceConnections(z4_possible_combinations)


    z5_possible_combinations = [
                            "R",
                            "C",
                            "L",
                            "R | C",
                            "R + C",
                            "L + C",
                            "L | C",
                            ]
    z5_block.setAllowedImpedanceConnections(z5_possible_combinations)


    zL_possible_combinations = [
                            "R",
                            "C",
                            "L",
                            "R | C",
                            "L | C"
                            ]
    zL_block.setAllowedImpedanceConnections(zL_possible_combinations)

    zS_possible_combinations = [
                            "R",
                            "C",
                            "L"
                            ]
    zS_block.setAllowedImpedanceConnections(zS_possible_combinations)

    # (1.3) Store all the declared impedances to be passed to the program
    zz = [z1_block, z2_block, z3_block, z4_block, z5_block, zL_block, zS_block]


    # End of step (1)
    # ---------------------------------------


    # (2)   NODAL EQUATION DEFINITION
    # ---------------------------------------
    # (2.1) Define symbolic variables (CG)
    Iip, Iin, I1a, I1b, I2a, I2b                       = symbols('Iip Iin I1a I1b I2a I2b')
    Vin, V2a, V2b, V1a, V1b, Va, Vb, Von, Vop, Vip, Vx = symbols('Vin V2a V2b V1a V1b Va Vb Von Vop Vip Vx')

    # (2.2) List all the nodal vairables to be fed into the solver
    solveFor = [
                Vin, V2a, V2b, V1a, V1b, Va, Vb, Von, Vop, Vip, Vx, # Voltages
                Iip, Iin, I1a, I1b, I2a, I2b                        # Currents
                ]

    # (2.3) Select a transmission matrix (default = symbolic matrix)
    T_a = TransmissionMatrix().getTranmissionMatrix()
    T_b = TransmissionMatrix().getTranmissionMatrix()

    # Need to extract the Z of the impedance block object 
    Z1 = z1_block.symbol
    Z2 = z2_block.symbol
    Z3 = z3_block.symbol
    Z4 = z4_block.symbol
    Z5 = z5_block.symbol
    ZL = zL_block.symbol
    ZS = zS_block.symbol

    # (2.4) Define the impedances that can be disconnected in the circuit
    impedancesToDisconnect = [Z1, Z2, Z3, Z4, Z5, ZL, ZS]

    # (2.5) Define the Nodal Equations
    nodalEquations = [
            # 3a
            Eq(0, (I1a + I2a + I1b + I2b + (Vb-Vx)/Z1 + (Von - Vx)/Z2 + (Vop - Vx)/Z2 + (Va - Vx)/Z1)),
            # 3b
            Eq(I1a, ((Vip - Va)/ZS + (Von - Va)/Z3 + (Vx - Va)/Z1 + (Vop - Va)/Z5)),
            # 3c
            Eq(I1b, (Vin - Vb)/ZS + (Vop - Vb)/Z3 + (Vx - Vb)/Z1 + (Von - Vb)/Z5),
            # 3d
            Eq(I2a, (Va - Von)/Z3 + (Vx - Von)/Z2 + (0 - Von)/ZL + (Vop - Von)/Z4 + (Vb - Von)/Z5),
            # 3e
            Eq(I2b, (Vb - Vop)/Z3 + (Vx - Vop)/Z2 + (0 - Vop)/ZL + (Von - Vop)/Z4 + (Va - Vop)/Z5),
            # 3f
            Eq(Vx + V2a, Vop),
            Eq(Vx + V2b, Von),
            # 3g
            Eq(Vx + V1a, Va),
            Eq(Vx + V1b, Vb),
            # 3h: Transistor a
            Eq(V1a, T_a[0,0]*V2a - T_a[0,1]*I2a),
            Eq(I1a, T_a[1,0]*V2a + T_a[1,1]*I2a),
            # 3i: Transistor b
            Eq(V1b, T_b[0,0]*V2b - T_b[0,1]*I2b),
            Eq(I1b, T_b[1,0]*V2b + T_b[1,1]*I2b)
            ]
    # End of step (2)
    # ---------------------------------------


    # Create a circuit instance to store the experiment parameters (in this case for our case study -- a common gate differential amplifier)
    circuit = Circuit(impedances=zz, nodal_equations=nodalEquations, solve_for=solveFor, impedancesToDisconnect=impedancesToDisconnect)
    def update_circuit(self) -> Circuit:
        self.circuit = Circuit(impedances=self.zz, nodal_equations=self.nodalEquations, solve_for=self.solveFor, impedancesToDisconnect=self.impedancesToDisconnect)
        return self.circuit
# ===================================================================
# End of Example 2 -- Common Source Differential Circuit

# Example 3 -- Voltage Divider Circuit
# ===================================================================
@dataclass(frozen=True)
class Voltage_Divider:
    """Differential Voltage Divider circuit with three Z blocks. Input take differentially from ZL"""
    # (1) Define possible impedances (assign names)
    # ---------------------------------------
    # (1.1) Declare the impedance blocks in the problem
    z1_block = Impedance_Block("1")
    z2_block = Impedance_Block("2")
    zL_block = Impedance_Block("L")

    # (1.2) Define the possible impedance types for each block AND assign it by calling self.setAllowedImpedanceConnections
    z1_possible_combinations =[                                             
                            "R",
                            "L",
                            "C"
                            ]
    z1_block.setAllowedImpedanceConnections(z1_possible_combinations)

    z2_possible_combinations =[                                             
                            "R",
                            "L",
                            "C"
                            ]
    z2_block.setAllowedImpedanceConnections(z2_possible_combinations)


    zl_possible_combinations = [                                             
                            "R",
                            "L",
                            "C"
                            ]
    zL_block.setAllowedImpedanceConnections(zl_possible_combinations)

    # (1.3) Store all the declared impedances to be passed to the program
    zz = [z1_block, z2_block, zL_block]

    # End of step (1)
    # ---------------------------------------


    # (2)   NODAL EQUATION DEFINITION
    # ---------------------------------------
    # (2.1) Define symbolic variables (CG)
    Iip, Iin, Iop, Ion = symbols('Iip Iin Iop Ion')
    Von, Vop, Vip, Vin = symbols('Von Vop Vip Vin')

    # (2.2) List all the nodal vairables to be fed into the solver
    solveFor = [
                Vin, Von, Vop, Vip, # Voltages
                Iip, Iin
                ]
    

    # (2.3) Select a transmission matrix (default = symbolic matrix)
    T_a = TransmissionMatrix().getTranmissionMatrix()
    T_b = TransmissionMatrix().getTranmissionMatrix()

    # Need to extract the Z of the impedance block object 
    Z1 = z1_block.symbol
    Z2 = z2_block.symbol
    ZL = zL_block.symbol

    # (2.4) Define the impedances that can be disconnected in the circuit
    impedancesToDisconnect = [Z1, ZL]

    # (2.5) Define the Nodal Equations
    nodalEquations = [
                Eq((Vip - Vop)/Z1, (Vop-Von)/ZL),
                Eq((Vip - Vin)/(Z1+ Z2 + ZL), (Vop-Von)/ZL),
                Eq(Iin, -1*Iip),
                Eq(Iin, (Vip - Vop)/Z1)
            ]
    # End of step (2)
    # ---------------------------------------

    # Create a circuit instance to store the experiment parameters (in this case for our case study -- a common gate differential amplifier)
    circuit = Circuit(impedances=zz, nodal_equations=nodalEquations, solve_for=solveFor, impedancesToDisconnect=impedancesToDisconnect)
    
    def update_circuit(self) -> Circuit:
        self.circuit = Circuit(impedances=self.zz, nodal_equations=self.nodalEquations, solve_for=self.solveFor, impedancesToDisconnect=self.impedancesToDisconnect)
        return self.circuit
# ===================================================================
# End of Example 3 -- Voltage Divider Circuit
