from sympy import symbols, Matrix, simplify, solve, degree, collect, sqrt

# Define symbolic variables
s = symbols('s')
R1, R2, R3, R4, R5, RL, Rs = symbols('R1 R2 R3 R4 R5 RL Rs')
C1, C2, C3, C4, C5, CL = symbols('C1 C2 C3 C4 C5 CL')
L1, L2, L3, L4, L5, LL = symbols('L1 L2 L3 L4 L5 LL')
gm = symbols('gm')

# Define impedance arrays
Zzs = [Rs, 1/(s*C1), s*L1]
Zz1 = [R1, 1/(s*C1), R1/(1 + R1*C1*s), R1 + 1/(C1*s), s*L1 + 1/(s*C1)]
Zz2 = [R2, 1/(s*C2), R2/(1 + R2*C2*s), R2 + 1/(C2*s), s*L2 + 1/(s*C2)]
Zz3 = [R3, 1/(s*C3), s*L3, R3/(1 + R3*C3*s), (s*L3 + 1/(s*C3)), (L3*s)/(1 + L3*C3*s**2)]
Zz4 = [R4, 1/(s*C4), s*L4, R4/(1 + R4*C4*s), (s*L4 + 1/(s*C4)), (L4*s)/(1 + L4*C4*s**2)]
Zz5 = [R5, 1/(s*C5), s*L5, R5/(1 + R5*C5*s), (s*L5 + 1/(s*C5)), (L5*s)/(1 + L5*C5*s**2)]
ZzL = [RL, 1/(s*CL), s*LL, RL/(1 + RL*CL*s), (LL*s)/(1 + LL*CL*s**2)]

# Combine all impedances
impedance_combinations = []
for zs in Zzs:
    for z1 in Zz1:
        for z2 in Zz2:
            for z3 in Zz3:
                for z4 in Zz4:
                    for z5 in Zz5:
                        for zl in ZzL:
                            impedance_combinations.append([zs, z1, z2, z3, z4, z5, zl])

# Initialize results
transfer_functions = []

# Transmission matrix coefficients
a11, a12, a21, a22 = 0, -1/gm, 0, 0

# Loop through all impedance combinations
for combo in impedance_combinations:
    Zs, Z1, Z2, Z3, Z4, Z5, ZL = combo

    # Define nodal equations
    equations = [
        (symbols('Vip') - symbols('va'))/Zs + (symbols('Von') - symbols('va'))/Z3 - symbols('I1a') +
        (symbols('vx') - symbols('va'))/Z1 + (symbols('Vop') - symbols('va'))/Z5,
        
        (symbols('Vin') - symbols('vb'))/Zs + (symbols('Vop') - symbols('vb'))/Z3 - symbols('I1b') +
        (symbols('vx') - symbols('vb'))/Z1 + (symbols('Von') - symbols('vb'))/Z5,

        (symbols('va') - symbols('Von'))/Z3 + (symbols('vx') - symbols('Von'))/Z2 - symbols('I2a') +
        (0 - symbols('Von'))/ZL + (symbols('Vop') - symbols('Von'))/Z4 + (symbols('vb') - symbols('Von'))/Z5,

        (symbols('vb') - symbols('Vop'))/Z3 + (symbols('vx') - symbols('Vop'))/Z2 - symbols('I2b') +
        (0 - symbols('Vop'))/ZL + (symbols('Von') - symbols('Vop'))/Z4 + (symbols('va') - symbols('Vop'))/Z5,

        (symbols('Von') - symbols('vx'))/Z2 + (symbols('Vop') - symbols('vx'))/Z2 + 
        (symbols('va') - symbols('vx'))/Z1 + (symbols('vb') - symbols('vx'))/Z1 +
        symbols('I1a') + symbols('I1b') + symbols('I2a') + symbols('I2b'),
    ]

    # Transmission matrix relationships
    equations += [
        symbols('vx') + symbols('v2a') - symbols('Von'),
        symbols('vx') + symbols('v2b') - symbols('Vop'),
        symbols('vx') + symbols('v1a') - symbols('va'),
        symbols('vx') + symbols('v1b') - symbols('vb'),
        symbols('v1a') - a11*symbols('v2a') + a12*symbols('I2a'),
        symbols('I1a') - a21*symbols('v2a') + a22*symbols('I2a'),
    ]

    # Solve for transfer function
    solution = solve(equations, [
        symbols('Vop'), symbols('Von'), symbols('I1a'), symbols('I2a'),
        symbols('v2a'), symbols('v1a'), symbols('v1b'), symbols('v2b'),
        symbols('I1b'), symbols('I2b'), symbols('Vip'), symbols('Vin'), symbols('vx')
    ])

    if solution:
        # Compute transfer function
        Hs = (solution[symbols('Vop')] - solution[symbols('Von')]) / \
             (solution[symbols('Vip')] - solution[symbols('Vin')])
        transfer_functions.append(simplify(Hs))

# Output results
print("Number of transfer functions found: {}".format(len(transfer_functions)))
for i, tf in enumerate(transfer_functions, 1):
    print("TF {}: {}".format(i, tf))

