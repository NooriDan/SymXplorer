import os, sys
import sympy                        # for symbolic modelling
from sympy import symbols, Matrix, Eq, simplify, solve, latex, denom, numer, sqrt, degree, init_printing, pprint, Poly
from tqdm import tqdm               # to create progress bars
from itertools import product       # for cartesian product
from typing import List, Optional   # for type checking
# import dill                       # to save/load the results

# Create the directory if it doesn't exist
print(f"# of cores: {os.cpu_count()}\nOS Name: {sys.platform}\nWorking Directory: {os.getcwd()}") # is 96 for TPU v2-8
init_printing()
# # symbols? #use this to find documentation on any object/functionts


# Define symbolic variables
s = symbols('s')
R1, R2, R3, R4, R5, RL, Rs      = symbols('R1 R2 R3 R4 R5 R_L R_s')
C1, C2, C3, C4, C5, CL          = symbols('C1 C2 C3 C4 C5 C_L')
L1, L2, L3, L4, L5, LL          = symbols('L1 L2 L3 L4 L5 L_L')
Z1 , Z2 , Z3 , Z4 , Z5 , ZL, Zs = symbols('Z1 Z2 Z3 Z4 Z5 Z_L Z_s')

# Get symbolic variables (CG)
Iip, Iin, I1a, I1b, I2a, I2b = symbols('Iip Iin I1a I1b I2a I2b')
Vin, V2a, V2b, V1a, V1b, Va, Vb, Von, Vop, Vip, Vx = symbols('Vin V2a V2b V1a V1b Va Vb Von Vop Vip Vx')

inf = sympy.oo # infinity symbol in SymPy

# Transmission matrix coefficients
gm, ro, Cgd, Cgs    = symbols('g_m r_o C_gd C_gs')
a11, a12, a21, a22  = symbols('a11 a12 a21 a22')

transmissionMatrix ={
    "simple"          : Matrix([[0, -1/gm],[0, 0]]),
    "symbolic"        : Matrix([[a11, a12],[a21, a22]]),
    "some_parasitic"  : Matrix([[-1/(gm*ro), -1/gm],[0, 0]]),
    "full_parasitic"  : Matrix([[(1/ro + s*Cgd)/(s*Cgd - gm), 1/(s*Cgd - gm)],[(Cgd*Cgs*ro*s + Cgd*gm*ro + Cgs + Cgd)*s/(s*Cgd - gm), (Cgs+Cgd)*s/(s*Cgd - gm)]])
}
