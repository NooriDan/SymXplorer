import os, sys
import sympy                        # for symbolic modelling
from sympy import symbols, Matrix, Eq, simplify, solve, latex, denom, numer, sqrt, degree, init_printing, pprint, Poly
from tqdm import tqdm               # to create progress bars
from itertools import product       # for cartesian product
from typing import List, Optional   # for type checking

# Create the directory if it doesn't exist
print(f"# of cores: {os.cpu_count()}\nOS Name: {sys.platform}\nWorking Directory: {os.getcwd()}") # is 96 for TPU v2-8
init_printing()


# Define symbolic variables
s = symbols('s')
R1, R2, R3, R4, R5, RL, Rs      = symbols('R1 R2 R3 R4 R5 R_L R_s')
C1, C2, C3, C4, C5, CL          = symbols('C1 C2 C3 C4 C5 C_L')
L1, L2, L3, L4, L5, LL          = symbols('L1 L2 L3 L4 L5 L_L')
Z1 , Z2 , Z3 , Z4 , Z5 , ZL, Zs = symbols('Z1 Z2 Z3 Z4 Z5 Z_L Z_s')

# Get symbolic variables (CG)
Iip, Iin, I1a, I1b, I2a, I2b                       = symbols('Iip Iin I1a I1b I2a I2b')
Vin, V2a, V2b, V1a, V1b, Va, Vb, Von, Vop, Vip, Vx = symbols('Vin V2a V2b V1a V1b Va Vb Von Vop Vip Vx')

inf = sympy.oo # infinity symbol in SymPy

# Define possible impedances
Zz1 =[  R1,                                            # R
        s*L1,                                          # L
        1/(s*C1),                                      # C
        R1/(1 + R1*C1*s),                              # R || C
        R1 + 1/(C1*s),                                 # R + C
        (s*L1 + 1/(s*C1)),                             # L + C
        (L1*s)/(1 + L1*C1*s**2),                       # L || C
        R1 + s*L1 + 1/(s*C1),                          # R + L + C
        (1/R1 + s*C1+ 1/(s*L1))**-1,                   # R || L || C
        R1 + (s*L1/(1 + L1*C1*s**2)),                  # R + (L || C)
        R1*(s*L1 + 1/(s*C1))/(R1 + (s*L1 + 1/(s*C1)))  # R || (L + C)
        ]
Zz2 = [ R2,                                            # R
        1/(s*C2),                                      # C
        R2/(1 + R2*C2*s),                              # R || C
        R2 + 1/(C2*s),                                 # R + C
        s*L2 + 1/(s*C2),                               # L + C
        R2 + s*L2 + 1/(s*C2),                          # R + L + C
        R2 + (s*L2/(1 + L2*C2*s**2)),                  # R + (L || C)
        R2*(s*L2 + 1/(s*C2))/(R2 + (s*L2 + 1/(s*C2)))  # R2 || (L2 + C2)
        ]

Zz3 = [ R3,                                            # R
        1/(s*C3),                                      # C
        R3/(1 + R3*C3*s),                              # R || C
        R3 + 1/(C3*s),                                 # R + C
        (s*L3 + 1/(s*C3)),                             # L + C
        (L3*s)/(1 + L3*C3*s**2),                       # L || C
        R3 + s*L3 + 1/(s*C3),                          # R + L + C
        (1/R3 + s*C3+ 1/(s*L3))**-1,                   # R || L || C
        R3 + (s*L3/(1 + L3*C3*s**2)),                  # R + (L || C)
        R3*(s*L3 + 1/(s*C3))/(R3 + (s*L3 + 1/(s*C3)))  # R || (L + C)
        ]


Zz4 = [ R4,                                            # R
        1/(s*C4),                                      # C
        R4/(1 + R4*C4*s),                              # R || C
        R4 + 1/(C4*s),                                 # R + C
        (s*L4 + 1/(s*C4)),                             # L + C
        (L4*s)/(1 + L4*C4*s**2),                       # L || C
        R4 + s*L4 + 1/(s*C4),                          # R + L + C
        (1/R4 + s*C4+ 1/(s*L4))**-1,                   # R || L || C
        R4 + (s*L4/(1 + L4*C4*s**2)),                  # R + (L || C)
        R4*(s*L4 + 1/(s*C4))/(R4 + (s*L4 + 1/(s*C4)))  # R || (L + C)
        ]

Zz5 = [ R5,                                            # R
        1/(s*C5),                                      # C
        R5/(1 + R5*C5*s),                              # R || C
        R5 + 1/(C5*s),                                 # R + C
        (s*L5 + 1/(s*C5)),                             # L + C
        (L5*s)/(1 + L5*C5*s**2),                       # L || C
        R5 + s*L5 + 1/(s*C5),                          # R + L + C
        (1/R5 + s*C5+ 1/(s*L5))**-1,                   # R || L || C
        R5 + (s*L5/(1 + L5*C5*s**2)),                  # R + (L || C)
        R5*(s*L5 + 1/(s*C5))/(R5 + (s*L5 + 1/(s*C5)))  # R || (L + C)
        ]

ZzL = [ RL,                                            # R
        1/(s*CL),                                      # C
        RL/(1 + RL*CL*s),                              # R || C
        RL + 1/(CL*s),                                 # R + C
        (s*LL + 1/(s*CL)),                             # L + C
        (LL*s)/(1 + LL*CL*s**2),                       # L || C
        RL + s*LL + 1/(s*CL),                          # R + L + C
        (1/RL + s*CL+ 1/(s*LL))**-1,                   # R || L || C
        RL + (s*LL/(1 + LL*CL*s**2)),                  # R + (L || C)
        RL*(s*LL + 1/(s*CL))/(RL + (s*LL + 1/(s*CL)))  # R || (L + C)
    ]


# Transmission matrix coefficients
gm, ro, Cgd, Cgs    = symbols('g_m r_o C_gd C_gs')
a11, a12, a21, a22  = symbols('a11 a12 a21 a22')

transmissionMatrix ={
    "simple"          : Matrix([[0, -1/gm],[0, 0]]),
    "symbolic"        : Matrix([[a11, a12],[a21, a22]]),
    "some_parasitic"  : Matrix([[-1/(gm*ro), -1/gm],[0, 0]]),
    "full_parasitic"  : Matrix([[(1/ro + s*Cgd)/(s*Cgd - gm), 1/(s*Cgd - gm)],[(Cgd*Cgs*ro*s + Cgd*gm*ro + Cgs + Cgd)*s/(s*Cgd - gm), (Cgs+Cgd)*s/(s*Cgd - gm)]])
}

# class TransmissionMatrix:
#     def __init__(self):
#         self.a11
#         self.a12 
#         self.a21 
#         self.a22

