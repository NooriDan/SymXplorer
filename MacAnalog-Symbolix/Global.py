import os, sys
from sympy import symbols, Eq, init_printing
from typing import List, Dict, Optional   # for type checking
from Utils import Impedance, TransmissionMatrix

# Create the directory if it doesn't exist
print(f"# of cores: {os.cpu_count()}\nOS Name: {sys.platform}\nWorking Directory: {os.getcwd()}") # is 96 for TPU v2-8
init_printing()

class ExperimentSetUP():
    def __init__():
        pass


# (1) Define possible impedances (assign names)
# ---------------------------------------
# (1.1) Declare the impedance blocks in the problem
z1 = Impedance("1")
z2 = Impedance("2")
z3 = Impedance("3")
z4 = Impedance("4")
z5 = Impedance("5")
zL = Impedance("L")

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
z1.setAllowedImpedanceConnections(z1_possible_combinations)


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
z2.setAllowedImpedanceConnections(z2_possible_combinations)


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
z3.setAllowedImpedanceConnections(z3_possible_combinations)


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
z4.setAllowedImpedanceConnections(z4_possible_combinations)


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
z5.setAllowedImpedanceConnections(z4_possible_combinations)


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
zL.setAllowedImpedanceConnections(zL_possible_combinations)

# (1.3) Store all the declared impedances to be passed to the program
zz                     = [z1, z2, z3, z4, z5, zL]
# (1.4) Define the impedances that can be disconnected in the circuit
impedancesToDisconnect = [z1.Z , z2.Z , z3.Z , z4.Z , z5.Z, zL.Z]
# End of step (1)
# ---------------------------------------



# (2)   NODAL EQUATION DEFINITION
# ---------------------------------------
# (2.1) Define symbolic variables (CG)
Iip, Iin, I1a, I1b, I2a, I2b                       = symbols('Iip Iin I1a I1b I2a I2b')
Vin, V2a, V2b, V1a, V1b, Va, Vb, Von, Vop, Vip, Vx = symbols('Vin V2a V2b V1a V1b Va Vb Von Vop Vip Vx')

# (2.2) List all the nodal vairables to be fed into the solver
solveFor = [Vin, V2a, V2b, V1a, V1b, Va, Vb, Von, Vop, Vip, Vx,
            Iip, Iin, I1a, I1b, I2a, I2b ]

# (2.3) Select a transmission matrix (default = symbolic matrix)
T_a = TransmissionMatrix().getTranmissionMatrix()
T_b = TransmissionMatrix().getTranmissionMatrix()

# Need to extract the Z of the impedance object 
Z1 = z1.Z
Z2 = z2.Z
Z3 = z3.Z
Z4 = z4.Z
Z5 = z5.Z
ZL = zL.Z

# (2.4) Define the Nodal Equations
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


# # OLD CODE
# # Define possible impedances

# R1, R2, R3, R4, R5, RL, Rs      = symbols('R1 R2 R3 R4 R5 R_L R_s')
# C1, C2, C3, C4, C5, CL          = symbols('C1 C2 C3 C4 C5 C_L')
# L1, L2, L3, L4, L5, LL          = symbols('L1 L2 L3 L4 L5 L_L')

# Z1 , Z2 , Z3 , Z4 , Z5 , ZL, Zs = symbols('Z1 Z2 Z3 Z4 Z5 Z_L Z_s')
# impedancesToDisconnect = [Z1 , Z2 , Z3 , Z4 , Z5, ZL] # for the CG case

# Zz1 =[  R1,                                            # R
#         s*L1,                                          # L
#         1/(s*C1),                                      # C
#         R1/(1 + R1*C1*s),                              # R || C
#         R1 + 1/(C1*s),                                 # R + C
#         (s*L1 + 1/(s*C1)),                             # L + C
#         (L1*s)/(1 + L1*C1*s**2),                       # L || C
#         R1 + s*L1 + 1/(s*C1),                          # R + L + C
#         (1/R1 + s*C1+ 1/(s*L1))**-1,                   # R || L || C
#         R1 + (s*L1/(1 + L1*C1*s**2)),                  # R + (L || C)
#         R1*(s*L1 + 1/(s*C1))/(R1 + (s*L1 + 1/(s*C1)))  # R || (L + C)
#         ]
# Zz2 = [ R2,                                            # R
#         1/(s*C2),                                      # C
#         R2/(1 + R2*C2*s),                              # R || C
#         R2 + 1/(C2*s),                                 # R + C
#         s*L2 + 1/(s*C2),                               # L + C
#         R2 + s*L2 + 1/(s*C2),                          # R + L + C
#         R2 + (s*L2/(1 + L2*C2*s**2)),                  # R + (L || C)
#         R2*(s*L2 + 1/(s*C2))/(R2 + (s*L2 + 1/(s*C2)))  # R2 || (L2 + C2)
#         ]

# Zz3 = [ R3,                                            # R
#         1/(s*C3),                                      # C
#         R3/(1 + R3*C3*s),                              # R || C
#         R3 + 1/(C3*s),                                 # R + C
#         (s*L3 + 1/(s*C3)),                             # L + C
#         (L3*s)/(1 + L3*C3*s**2),                       # L || C
#         R3 + s*L3 + 1/(s*C3),                          # R + L + C
#         (1/R3 + s*C3+ 1/(s*L3))**-1,                   # R || L || C
#         R3 + (s*L3/(1 + L3*C3*s**2)),                  # R + (L || C)
#         R3*(s*L3 + 1/(s*C3))/(R3 + (s*L3 + 1/(s*C3)))  # R || (L + C)
#         ]


# Zz4 = [ R4,                                            # R
#         1/(s*C4),                                      # C
#         R4/(1 + R4*C4*s),                              # R || C
#         R4 + 1/(C4*s),                                 # R + C
#         (s*L4 + 1/(s*C4)),                             # L + C
#         (L4*s)/(1 + L4*C4*s**2),                       # L || C
#         R4 + s*L4 + 1/(s*C4),                          # R + L + C
#         (1/R4 + s*C4+ 1/(s*L4))**-1,                   # R || L || C
#         R4 + (s*L4/(1 + L4*C4*s**2)),                  # R + (L || C)
#         R4*(s*L4 + 1/(s*C4))/(R4 + (s*L4 + 1/(s*C4)))  # R || (L + C)
#         ]

# Zz5 = [ R5,                                            # R
#         1/(s*C5),                                      # C
#         R5/(1 + R5*C5*s),                              # R || C
#         R5 + 1/(C5*s),                                 # R + C
#         (s*L5 + 1/(s*C5)),                             # L + C
#         (L5*s)/(1 + L5*C5*s**2),                       # L || C
#         R5 + s*L5 + 1/(s*C5),                          # R + L + C
#         (1/R5 + s*C5+ 1/(s*L5))**-1,                   # R || L || C
#         R5 + (s*L5/(1 + L5*C5*s**2)),                  # R + (L || C)
#         R5*(s*L5 + 1/(s*C5))/(R5 + (s*L5 + 1/(s*C5)))  # R || (L + C)
#         ]

# ZzL = [ RL,                                            # R
#         1/(s*CL),                                      # C
#         RL/(1 + RL*CL*s),                              # R || C
#         RL + 1/(CL*s),                                 # R + C
#         (s*LL + 1/(s*CL)),                             # L + C
#         (LL*s)/(1 + LL*CL*s**2),                       # L || C
#         RL + s*LL + 1/(s*CL),                          # R + L + C
#         (1/RL + s*CL+ 1/(s*LL))**-1,                   # R || L || C
#         RL + (s*LL/(1 + LL*CL*s**2)),                  # R + (L || C)
#         RL*(s*LL + 1/(s*CL))/(RL + (s*LL + 1/(s*CL)))  # R || (L + C)
#     ]