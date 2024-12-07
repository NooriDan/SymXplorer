from Global import *
from BaseTF import BaseTransferFunction
from Filter import FilterClassification, FilterClassifier
from Utils  import FileSave
from typing import Dict
import sympy
import pickle

class ExperimentResult:
    def __init__(self, baseHs: sympy.Basic, classifications: List[FilterClassification]):
        self.baseHs = baseHs
        self.classifications = classifications 

    def at(self, idx):
        if idx < len(self.classifications) and idx > -1:
            return self.classifications[idx]
        else:
            raise IndexError
    def getType(self, fType):
        output = []
        for classification in self.classifications:
            if classification.fType == fType:
                output.append(classification)
        return output


class SymbolixExperimentCG:
    """Main class putting everything together"""
    def __init__(self, _experimentName: str, baseHs: BaseTransferFunction):
         
         self.experimentName = _experimentName
         self.baseHsObject: BaseTransferFunction = baseHs

         self.classifier: FilterClassifier = FilterClassifier()
         self.fileSave: FileSave = FileSave(outputDirectory=f"Runs/{self.experimentName}")
         
         # to be computed
         self.baseHsDict: Dict[str, sympy.Basic]= {}
         self.setPossibleBase()

         self.transferFunctions: List[sympy.Basic] = []
         self.solvedCombos: List[int]= []
         self.numOfComputes: int = 0

    def isBaseSolved(self):
        return self.baseTF.isSolved
    
    def getComboKeys(self):
        return self.getPossibleBase().keys()
    
    def getPossibleBase(self):
        return self.baseHsDict
        
    def applyLimitsToBase(self, variables_to_limit: List[sympy.Symbol], limitingValue: sympy.Basic = sympy.oo):
        """Applies sympy.limit to a set of variables."""
        baseHs = self.baseHsObject.baseHs  # Local variable, doesn't modify self.baseHs
        for var in variables_to_limit:
            baseHs = sympy.limit(baseHs, var, limitingValue)
        return baseHs

    def setPossibleBase(self):
        baseHs = self.baseHsObject.baseHs
        self.baseHsDict =  {
            "all"       : baseHs,
            "Z1_ZL"     : self.applyLimitsToBase([Z2, Z3, Z4, Z5]),
            "Z2_ZL"     : self.applyLimitsToBase([Z1, Z3, Z4, Z5]),
            "Z3_ZL"     : self.applyLimitsToBase([Z1, Z2, Z4, Z5]),
            "Z4_ZL"     : self.applyLimitsToBase([Z1, Z2, Z3, Z5]),
            "Z5_ZL"     : self.applyLimitsToBase([Z1, Z2, Z3, Z4]),
            "Z1_Z2_ZL"  : self.applyLimitsToBase([Z3, Z4, Z5]),
            "Z1_Z3_ZL"  : self.applyLimitsToBase([Z2, Z4, Z5]),
            "Z1_Z4_ZL"  : self.applyLimitsToBase([Z2, Z3, Z5]),
            "Z1_Z5_ZL"  : self.applyLimitsToBase([Z2, Z3, Z4]),
            "Z2_Z3_ZL"  : self.applyLimitsToBase([Z1, Z4, Z5]),
            "Z2_Z4_ZL"  : self.applyLimitsToBase([Z1, Z3, Z5]),
            "Z2_Z5_ZL"  : self.applyLimitsToBase([Z1, Z3, Z4]),
            "Z3_Z4_ZL"  : self.applyLimitsToBase([Z1, Z2, Z5]),
            "Z3_Z5_ZL"  : self.applyLimitsToBase([Z1, Z2, Z4]),
            "Z4_Z5_ZL"  : self.applyLimitsToBase([Z1, Z2, Z3]),
        }
    
    def getZcombos(self):
        """Assumes Zzi and inf are defined in Global.py"""
        return {
            "all"         : product(Zz1, Zz2, Zz3, Zz4, Zz5, ZzL),          # all (Zi, ZL) combo
            "Z1_ZL"       : product(Zz1, [inf], [inf], [inf], [inf], ZzL),
            "Z2_ZL"       : product([inf], Zz2, [inf], [inf], [inf], ZzL),
            "Z3_ZL"       : product([inf], [inf], Zz3, [inf], [inf], ZzL),
            "Z4_ZL"       : product([inf], [inf], [inf], Zz4, [inf], ZzL),
            "Z5_ZL"       : product([inf], [inf], [inf], [inf], Zz5, ZzL),
            "Z1_Z2_ZL"    : product(Zz1, Zz2, [inf], [inf], [inf], ZzL),
            "Z1_Z3_ZL"    : product(Zz1, [inf], Zz3, [inf], [inf], ZzL),
            "Z1_Z4_ZL"    : product(Zz1, [inf], [inf], Zz4, [inf], ZzL),
            "Z1_Z5_ZL"    : product(Zz1, [inf], [inf], [inf], Zz5, ZzL),
            "Z2_Z3_ZL"    : product([inf], Zz2, Zz3, [inf], [inf], ZzL),
            "Z2_Z4_ZL"    : product([inf], Zz2, [inf], Zz4, [inf], ZzL),
            "Z2_Z5_ZL"    : product([inf], Zz2, [inf], [inf], Zz5, ZzL),
            "Z3_Z4_ZL"    : product([inf], [inf], Zz3, Zz4, [inf], ZzL),
            "Z3_Z5_ZL"    : product([inf], [inf], Zz3, [inf], Zz5, ZzL),
            "Z4_Z5_ZL"    : product([inf], [inf], [inf], Zz4, Zz5, ZzL),
        }
    
    def computeTransferFunction(self, baseHs, zCombo):
        _Z1, _Z2, _Z3, _Z4, _Z5, _ZL = zCombo
        sub_dict = {
            symbols("Z1"): _Z1,
            symbols("Z2"): _Z2,
            symbols("Z3"): _Z3,
            symbols("Z4"): _Z4,
            symbols("Z5"): _Z5,
            symbols("Z_L"): _ZL
        }

        Hs = baseHs.subs(sub_dict)  # Substitute the impedance values into the base function
        Hs = simplify(Hs.factor())  # Simplify and factor the resulting expression
        Hs = sympy.together(Hs)
        return Hs
    
    def computeTFs(self, comboKey="all", clearRecord=True):
        # Clear previous records if necessary

        print(f"combo key = {comboKey}")
        self.classifier.transferFunctionsList = []
        self.classifier.impedanceList = []
        # self.classifier.clearFilter()

        # Ensure the comboKey is valid
        try:
            impedanceBatch = list(self.getZcombos()[comboKey])
        except KeyError:
            raise ValueError(f"Invalid comboKey '{comboKey}' provided.")

        # Prepare the base transfer function
        baseHs = self.baseHsDict.get(comboKey)
        if baseHs is None:
            raise ValueError(f"BaseHs for the comboKey '{comboKey}' is not found.")

        # Use list comprehension for efficient transfer function computation
        solvedTFs = [
            self.computeTransferFunction(baseHs, zCombo)
            for zCombo in tqdm(impedanceBatch, desc="Getting the TFs (CG)", unit="combo")
        ]
        
        # Add the computed transfer functions to the classifier
        self.classifier.addTFs(solvedTFs, impedanceBatch)
        self.numOfComputes += 1

        # Output the number of transfer functions computed
        print(f"Number of transfer functions found: {len(solvedTFs)}")


    # Reporting methods (generate pdf)
    def reportAll(self, experimentName, Z_arr):
        self.fileSave.generateLaTeXReport(self.classifier.classifications, 
                            output_filename= f"{experimentName}_{Z_arr}_all",
                            subFolder=f"{experimentName}_{Z_arr}")
    
    def reportType(self, fType, experimentName, Z_arr):
        self.fileSave.generateLaTeXReport(self.classifier.clusteredByType[fType], 
                            output_filename= f"{experimentName}_{Z_arr}_{fType}",
                            subFolder=f"{experimentName}_{Z_arr}")
    
    def reportSummary(self, experimentName, Z_arr):
        if(self.classifier.clusteredByType):
            if(self.baseHsDict.get(Z_arr)):
                self.fileSave.generateLaTeXSummary(self.baseHsDict[Z_arr], 
                                                    self.classifier.clusteredByType,
                                                    output_filename= f"{experimentName}_{Z_arr}_summary",
                                                    subFolder=f"{experimentName}_{Z_arr}")
            else:
                print("!!! INVALID Z_arr !!!")
        else:
            print("==** Need to first summarize the filters")

    def compilePDF(self):
        self.fileSave.compile()

    def export(self, filename):
        self.fileSave.export(self, filename)
        

    # # ---------- OLD CODE ----------
    # def computeTFs(self, comboKey = "all", clearRecord = True):
    #     solvedTFs = []

    #     if clearRecord:
    #          self.classifier.clearFilter()
    #          self.numOfComputes     = 0

    #     impedanceBatch = list(self.getZcombos()[comboKey])
    #     self.setPossibleBase()
    #     baseHs = self.baseHsDict[comboKey]

    #     for zCombo in tqdm(impedanceBatch, desc="Getting the TFs (CG)", unit="combo"):
    #           Z1, Z2, Z3, Z4, Z5, ZL = zCombo
    #         #   print(f"Hs (before) : {self.baseHsObject.baseHs}")
    #           sub_dict = {symbols("Z1") : Z1,
    #                       symbols("Z2") : Z2,
    #                       symbols("Z3") : Z3,
    #                       symbols("Z4") : Z4,
    #                       symbols("Z5") : Z5,
    #                       symbols("Z_L") : ZL}
              
    #         #   print(f"sub_dict = {sub_dict}")
    #         #   print("=========")

    #           Hs = baseHs.subs(sub_dict)
    #           Hs = simplify((Hs.factor()))
    #           # record the Z combo and its H(s)
    #           solvedTFs.append(Hs)
      
    #     self.classifier.addTFs(solvedTFs,impedanceBatch)
    #     self.numOfComputes += 1

    #     # Output summary of results
    #     print("Number of transfer functions found: {}".format(len(solvedTFs)))

    #     return solvedTFs, impedanceBatch