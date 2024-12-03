from Global import *

class BaseTransferFunction:
    """Implementation of algorithm 1 -> finds the base transfer function"""
    def __init__(self, _output = [Vop, Von], _input= [Vip, Vin],
                  transmissionMatrixType="symbolic", T_analysis=transmissionMatrix):
        
        self.T_type = transmissionMatrixType
        self.T_analysis = T_analysis

        self.solveFor = [
                  Vin, V2a, V2b, V1a, V1b, Va, Vb, Von, Vop, Vip, Vx,
                  Iip, Iin, I1a, I1b, I2a, I2b
            ]
        self.output = _output
        self.input  = _input

        # variables to be computed for
        self.baseHs = None
        self.T_a    = None
        self.T_b    = None

    # Custom string representation for BaseTransferFunction
    def __repr__(self):
        return f"BaseTranferFunction object -> T_type {self.T_type}\n H(s) = ({self.output[0]} - {self.output[1]}) / ({self.input[0]} - {self.input[1]})"

    def isSolved(self):
        return self.baseHs is not None
    
    def setT_type(self, transmissionMatrixType):
        self.T_type = transmissionMatrixType
    
    def _getTransmissionMatrix(self):
        self.T_a = self.T_analysis[self.T_type]
        self.T_b = self.T_a

        return self.T_a, self.T_b
    
    def _setEquations(self):
      # Get symbolic variables (CG)
      T_a, T_b = self._getTransmissionMatrix()

      # Define nodal equations (Eq. 4a-4h) -> list[ Eq(left-hand-side, right-hand-side), ... ]
      equations = [
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
      self.equations = equations
      return equations
    
    # Get transfer function (TF)
    def solve(self):
        print(f"====CommonGate====")
        oPos, oNeg = self.output
        iPos, iNeg = self.input
        # Define nodal equations
        equations = self._setEquations()
        print("(1) set up the nodal equation")

        # Solve for generic transfer function
        solution = solve(equations, self.solveFor)
        print("(2) solved the base transfer function")

        if solution:
            print("FOUND THE BASE TF")
            baseHs = (solution[oPos] - solution[oNeg]) / (solution[iPos] - solution[iNeg])
            baseHs = simplify((baseHs.factor()))
            self.baseHs = baseHs
            self.isSolved = True
            return baseHs

        return None
    
