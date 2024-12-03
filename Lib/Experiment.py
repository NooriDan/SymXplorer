from Global import *
from BaseTF import BaseTransferFunction
from Filter import FilterClassification, FilterClassifier
from Utils  import FileSave
import pickle

class SymbolixExperimentCG:
    """Main class putting everything together"""
    def __init__(self, baseHs: BaseTransferFunction):
         self.baseHsObject = baseHs
         self.filter: FilterClassifier = FilterClassifier()
         self.fileSave = FileSave()
         # to be computed
         self.baseHsDict = {}
         self.transferFunctions = []
         self.solvedCombos      = []
         self.numOfComputes     = 0

    def export(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)
        print(f"Object exported to {file}")

    @staticmethod
    def import_from(file):
        with open(file, 'rb') as f:
            obj = pickle.load(f)
        print(f"Object imported from {file}")
        return obj

    def isBaseSolved(self):
        return self.baseTF.isSolved
    
    def getComboKeys(self):
        return self.getPossibleZcombo().keys()
    
    def reportAll(self, experimentName, Z_arr):
        self.fileSave.generateLaTeXReport(self.filter.classifications, 
                            output_filename= f"{experimentName}_{Z_arr}_all",
                            subFolder=f"{experimentName}_{Z_arr}")
    
    def reportType(self, fType, experimentName, Z_arr):
        self.fileSave.generateLaTeXReport(self.filter.clusteredByType[fType], 
                            output_filename= f"{experimentName}_{Z_arr}_{fType}",
                            subFolder=f"{experimentName}_{Z_arr}")

    def computeTFs(self, comboKey = "all", clearRecord = True):
        solvedTFs = []

        if clearRecord:
             self.filter.clearFilter()
             self.numOfComputes     = 0

        impedanceBatch = list(self.getPossibleZcombo()[comboKey])
        self.getPossibleBase()
        baseHs = self.baseHsDict[comboKey]

        for zCombo in tqdm(impedanceBatch, desc="Getting the TFs (CG)", unit="combo"):
              Z1, Z2, Z3, Z4, Z5, ZL = zCombo
            #   print(f"Hs (before) : {self.baseHsObject.baseHs}")
              sub_dict = {symbols("Z1") : Z1,
                          symbols("Z2") : Z2,
                          symbols("Z3") : Z3,
                          symbols("Z4") : Z4,
                          symbols("Z5") : Z5,
                          symbols("Z_L") : ZL}
              
            #   print(f"sub_dict = {sub_dict}")
            #   print("=========")

              Hs = baseHs.subs(sub_dict)
              Hs = simplify((Hs.factor()))
              # record the Z combo and its H(s)
              solvedTFs.append(Hs)
      
        self.filter.addTFs(solvedTFs,impedanceBatch)
        self.numOfComputes += 1

        # Output summary of results
        print("Number of transfer functions found: {}".format(len(solvedTFs)))
        # for i, tf in enumerate(solvedTFs, 1):
        #     print("H(s) {}: {}".format(i, tf))

        return solvedTFs, impedanceBatch
    
    # (Defines the impedance arrays)
    def getPossibleZcombo(self):
        Zz1 =[ R1,                                           # R
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
        Zz2 = [R2,                                           # R
              1/(s*C2),                                      # C
              R2/(1 + R2*C2*s),                              # R || C
              R2 + 1/(C2*s),                                 # R + C
              s*L2 + 1/(s*C2),                               # L + C
              R2 + s*L2 + 1/(s*C2),                          # R + L + C
              R2 + (s*L2/(1 + L2*C2*s**2)),                  # R + (L || C)
              R2*(s*L2 + 1/(s*C2))/(R2 + (s*L2 + 1/(s*C2)))  # R2 || (L2 + C2)
              ]

        Zz3 = [R3,                                           # R
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


        Zz4 = [R4,                                           # R
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

        Zz5 = [R4,                                           # R
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

        ZzL = [RL,                                           # R
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
        # Combine Z
        return {
          "all"         : product(Zz1, Zz2, Zz3, Zz4, Zz5, ZzL),
          # all (Zi, ZL) combo
          "Z1_ZL"       : product(Zz1, [inf], [inf], [inf], [inf], ZzL),
          "Z2_ZL"       : product([inf], Zz2, [inf], [inf], [inf], ZzL),
          "Z3_ZL"       : product([inf], [inf], Zz3, [inf], [inf], ZzL),
          "Z4_ZL"       : product([inf], [inf], [inf], Zz4, [inf], ZzL),
          "Z5_ZL"       : product([inf], [inf], [inf], [inf], Zz5, ZzL),
          # 
          "Z2_Z4_ZL"    : product([inf], Zz2, [inf], Zz4, [inf], ZzL),
          "Z2_Z5_ZL"    : product([inf], Zz2, [inf], [inf], Zz5, ZzL),
          "Z3_Z5_ZL"    : product([inf], [inf], Zz3, [inf], Zz5, ZzL)
          }  
        
    def getPossibleBase(self):
        baseHs = self.baseHsObject.baseHs
        # oo is the symbol for infitity (Table II)
        # ------------------------------------------------------------
        baseHs_CG_Z1ZL      = sympy.limit(baseHs, Z2, inf)         # Z2, Z3, Z4, and Z5 are set to oo (infinity)
        baseHs_CG_Z1ZL      = sympy.limit(baseHs_CG_Z1ZL, Z3, inf)
        baseHs_CG_Z1ZL      = sympy.limit(baseHs_CG_Z1ZL, Z4, inf)
        baseHs_CG_Z1ZL      = sympy.limit(baseHs_CG_Z1ZL, Z5, inf)
        # ------------------------------------------------------------
        baseHs_CG_Z2ZL      = sympy.limit(baseHs, Z1, inf)         # Z2, Z3, Z4, and Z5 are set to oo (infinity)
        baseHs_CG_Z2ZL      = sympy.limit(baseHs_CG_Z2ZL, Z3, inf)
        baseHs_CG_Z2ZL      = sympy.limit(baseHs_CG_Z2ZL, Z4, inf)
        baseHs_CG_Z2ZL      = sympy.limit(baseHs_CG_Z2ZL, Z5, inf)
        # ------------------------------------------------------------
        baseHs_CG_Z4ZL      = sympy.limit(baseHs, Z1, inf)         # Z2, Z3, Z4, and Z5 are set to oo (infinity)
        baseHs_CG_Z4ZL      = sympy.limit(baseHs_CG_Z4ZL, Z2, inf)
        baseHs_CG_Z4ZL      = sympy.limit(baseHs_CG_Z1ZL, Z3, inf)
        baseHs_CG_Z4ZL      = sympy.limit(baseHs_CG_Z1ZL, Z5, inf)
        # ------------------------------------------------------------
        baseHs_CG_Z3ZL      = sympy.limit(baseHs, Z1, inf)         # Z1, Z2, Z4, and Z5 are set to oo (infinity)
        baseHs_CG_Z3ZL      = sympy.limit(baseHs_CG_Z3ZL, Z2, inf)
        baseHs_CG_Z3ZL      = sympy.limit(baseHs_CG_Z3ZL, Z4, inf)
        baseHs_CG_Z3ZL      = sympy.limit(baseHs_CG_Z3ZL, Z5, inf)
        # ------------------------------------------------------------
        baseHs_CG_Z5ZL      = sympy.limit(baseHs, Z1, inf)         # Z1, Z2, Z3, and Z4 are set to oo (infinity)
        baseHs_CG_Z5ZL      = sympy.limit(baseHs_CG_Z5ZL, Z2, inf)
        baseHs_CG_Z5ZL      = sympy.limit(baseHs_CG_Z5ZL, Z3, inf)
        baseHs_CG_Z5ZL      = sympy.limit(baseHs_CG_Z5ZL, Z4, inf)
        # ------------------------------------------------------------
        baseHs_CG_Z2Z4ZL    = sympy.limit(baseHs, Z1, inf)         # Z1, Z3 and Z5 are set to oo (infinity)
        baseHs_CG_Z2Z4ZL    = sympy.limit(baseHs_CG_Z2Z4ZL, Z3, inf)
        baseHs_CG_Z2Z4ZL    = sympy.limit(baseHs_CG_Z2Z4ZL, Z5, inf)
        # ------------------------------------------------------------
        baseHs_CG_Z2Z5ZL    = sympy.limit(baseHs, Z1, inf)         # Z1, Z3, and Z4 are set to oo (infinity)
        baseHs_CG_Z2Z5ZL    = sympy.limit(baseHs_CG_Z2Z5ZL, Z3, inf)
        baseHs_CG_Z2Z5ZL    = sympy.limit(baseHs_CG_Z2Z5ZL, Z4, inf)
        # ------------------------------------------------------------
        baseHs_CG_Z3Z5ZL    = sympy.limit(baseHs, Z1, inf)         # Z1, Z2, and Z4 are set to oo (infinity)
        baseHs_CG_Z3Z5ZL    = sympy.limit(baseHs_CG_Z3Z5ZL, Z2, inf)
        baseHs_CG_Z3Z5ZL    = sympy.limit(baseHs_CG_Z3Z5ZL, Z4, inf)
        # ------------------------------------------------------------

        self.baseHsDict =  {
            "all"       : baseHs,
            "Z1_ZL"     : baseHs_CG_Z1ZL,
            "Z2_ZL"     : baseHs_CG_Z2ZL,
            "Z3_ZL"     : baseHs_CG_Z3ZL,
            "Z4_ZL"     : baseHs_CG_Z4ZL,
            "Z5_ZL"     : baseHs_CG_Z5ZL,
            "Z2_Z4_ZL"  : baseHs_CG_Z2Z4ZL,
            "Z2_Z5_ZL"  : baseHs_CG_Z2Z5ZL,
            "Z3_Z5_ZL"  : baseHs_CG_Z3Z5ZL
        }

        return self.baseHsDict