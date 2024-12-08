import sympy
from   tqdm import tqdm
from   sympy import denom, numer, degree, symbols, simplify, sqrt
from   typing import Dict, List, Optional
# Custom Imports
# from Global import *

class FilterClassification:
    def __init__(
        self,
        zCombo: List[sympy.Basic],
        transferFunc: sympy.Basic,  # SymPy expression
        valid: bool = False,
        fType: Optional[str] = "None",
        parameters: Optional[dict] = None,
        filterOrder: Optional[str] = "None"
    ):
        self.filterOrder = filterOrder
        self.zCombo = zCombo
        self.transferFunc = transferFunc
        self.valid = valid
        self.fType = fType
        self.parameters = parameters

    def __eq__(self, other) -> bool:
        if not isinstance(other, FilterClassification):
            return NotImplemented
        return (self.fType == other.fType) and (self.filterOrder == other.filterOrder)

    def __repr__(self) -> str:
        return (
            f"FilterClassification("
            f"zCombo={self.zCombo}, transferFunc={self.transferFunc}, "
            f"valid={self.valid}, fType={self.fType}, parameters={self.parameters})"
        )


class FilterClassifier():
    """Implementation of algorithm 2"""
    def __init__(self, transferFunctionList: List = [], impedanceBatch: List = [], fTypes: List[str] = ["HP", "BP", "LP", "BS", "GE", "AP"]):
        self.transferFunctionsList = transferFunctionList
        self.impedanceList         = impedanceBatch
        self.filterParameters = []
        self._fTypes = fTypes
        # To be computed
        self.classifications: List[FilterClassification] = []
        self.clusteredByType:  Dict[str, List[FilterClassification]]= {}
        self.countByType:      Dict[str, int] = {}

        # Constants
        self.possibleFilterOrders ={
            "BiQuad"        : self._getBiQuadParameters,
            "FirstOrder"    : self._getFirstOrderParameters,
            "ThirdOrder"    : self._getThirdOrderParameters
        }
        
    def hasTFs(self):
        return len(self.transferFunctionsList) > 0
    
    def isClassified(self):
        return len(self.transferFunctionsList) == len(self.classifications)
    
    def countTF(self):
        return len(self.transferFunctionsList)
    
    def countValid(self):
        countValid = 0
        for filter in self.classifications:
            if (filter.valid):
                countValid += 1
        return countValid, len(self.classifications)

    def addTFs(self, transferFunctions, impedanceBatch):
        if(len(impedanceBatch) != len(transferFunctions)):
            print("==== the TF and Z array size mismatch ====")
            return False
        
        for tf, zCombo in zip(transferFunctions, impedanceBatch):
            self.transferFunctionsList.append(tf)
            self.impedanceList.append(zCombo)
        return True

    def clearFilter(self):
        print("!!!! Clearning the filter !!!!")
        self.transferFunctionsList = []
        self.filterParameters = []
        self.classifications = []
    
    def addFilterType(self, types: list):
        for _type in types:
            self._fTypes.append(_type)

    def summarizeFilterType(self, filterTypes=["HP", "BP", "LP", "BS", "GE", "AP", "INVALID-NUMER", "INVALID-WZ", "INVALID-ORDER"]):
        if not self.isClassified():
            print("===============")
            print("Classify the TFs first")
            print("===============")

        counts = {}
        for fType in filterTypes:
            self.clusteredByType[fType], self.countByType[fType] = self._findFilterInClassification(fType)
        return self.clusteredByType, counts

    # HELPER FUNCTIONS (private)
    def _findFilterInClassification(self, filterType, printMessage=True):
        output = []
        count = 0
        for entity in self.classifications:
            if (entity.fType == filterType):
                output.append(entity)
                count += 1
        if printMessage:
            print(f"{filterType} : {len(output)}")

        return output, count

    def classifyBiQuadFilters(self):
        self.classifications = []
        
        # Wrap the zip iterator with tqdm for progress tracking
        for tf, impedanceCombo in tqdm(zip(self.transferFunctionsList, self.impedanceList),
                                        total=self.countTF(),
                                        desc="Computing filter parameters",
                                        unit="filter"):

            results = self._getBiQuadParameters(tf)
            if results['valid']:
                self.classifications.append(FilterClassification(
                    zCombo       = impedanceCombo,
                    transferFunc = tf,
                    valid= True,
                    fType        = results["fType"],
                    parameters   = results["parameters"]
                ))
            else:
                self.classifications.append(FilterClassification(
                    zCombo= impedanceCombo,
                    transferFunc= tf,
                    valid= False,
                    fType= results["fType"],
                    parameters= results["parameters"]
                ))
    
    def _getBiQuadParameters(self, tf):
        """
        Computes the parameters of a biquad filter given its transfer function.
        
        Assumes the filter follows the form:
            tf = (b2 * s^2 + b1 * s + b0) / (a2 * s^2 + a1 * s + a0)
        Compares tf to:
            H(s) = K * N_XY(s) / (s^2 + (wo/Q)*s + wo^2)
        
        Returns a dictionary of parameters or {'valid': False} if invalid.
        """
        # Define symbolic variable
        s = symbols('s')
        
        # Extract numerator and denominator
        denominator = denom(tf).expand()  # Denominator of tf
        numerator = numer(tf).expand()    # Numerator of tf

        # Determine orders
        den_order = degree(denominator, s)
        num_order = degree(numerator, s)

        # Extract denominator coefficients
        a2 = denominator.coeff(s, 2)
        a1 = denominator.coeff(s, 1)
        a0 = denominator.coeff(s, 0)

        # Validate filter form and coefficients
        if not all([a2, a1, a0]) or num_order > 2 or den_order > 2:
            return {'valid': False,
                    'fType': "INVALID-ORDER",
                    'parameters': None}

        # Compute natural frequency (wo), quality factor (Q), and bandwidth
        wo = sqrt(simplify(a0 / a2))
        Q = simplify((a2 / a1) * wo)
        bandwidth = wo / Q

        # Extract numerator coefficients
        b2 = numerator.coeff(s, 2)
        b1 = numerator.coeff(s, 1)
        b0 = numerator.coeff(s, 0)

        # Calculate filter constants
        #           b2 s^2 + b1 s^1 + b0                N_XY(s)
        #  H(s) = ------------------------- = K ------------------------
        #           a2 s^2 + a1 s^1 + a0          s^2 + (wo/Q)*s + wo^2
        
        # Possible N_XY(s): K = {K_HP, K_LP, K_BP, K_BS, K_GE}
        #   1 - N_HP = s^2
        #   2 - N_LP = wo^2
        #   3 - N_BP = wo/Q * s     ----> wo/Q = bandwidth
        #   4 - N_BS = s^2 + wo^2
        #   5 - N_GE = s^2 + (wz/Qz) * s + wz^2 ----> wz = wo, AP filter if Qz = -Q

        numeratorState = ((b2 != 0) << 2) | ((b1 != 0) << 1) | (b0 != 0)
        match numeratorState:
            case 0b100: # b2 s^2
                fType = "HP"
            case 0b010: # b1 *s 
                fType = "BP"
            case 0b001: # b0
                fType = "LP"
            case 0b101: # b2 * s^2 + b0
                fType = "BS"
            case 0b111:
                fType = "GE"    # GE or AP
            case 0b011:     # This situation is not accounted for in N_XY(s) scenarios
                fType = "INVALID-NUMER"
            case 0b110:     # This situation is not accounted for in N_XY(s) scenarios
                fType = "INVALID-NUMER"
            case _:         # catches other combos (i.e., 110, 011)
                fType = "INVALID-NUMER"


        # Compute zero's natural frequency (wz) if applicable
        valid = True
        if b2 != 0 and b0 != 0:
            wz = sqrt(simplify(b0 / b2))
        else:
            wz = None

        # Compute filter constants
        K_HP = simplify(b2 / a2)
        K_BP = simplify(b1 / (a2 * bandwidth))
        K_LP = simplify(b0 / (a2 * wo**2))


        # # compare wz to wo
        if (fType in ["BS", "GE"]) and (wz != wo):
            valid = False
            fType = "INVALID-WZ"

        # Additional parameter (Qz) for Generalized Equalizer (GE) filters
        Qz = simplify((b2 / b1) * wo) if b1 != 0 else None

        if (Qz) and (Qz == -Q):
            fType = "AP"

        # Return computed parameters
        return {
            "valid": valid,
            "fType": fType,
            "parameters": {
                "Q": Q,
                "wo": wo,
                "bandwidth": bandwidth,
                "K_LP": K_LP,
                "K_HP": K_HP,
                "K_BP": K_BP,
                "Qz": Qz,
                "Wz": wz
            }
        }

    def _getFirstOrderParameters(self, tf):
        print(" === inside FIRST ORDER Parameter Computation === ")
        pass

    def _getThirdOrderParameters(self, tf):
        print(" === inside THIRD ORDER Parameter Computation === ")

        pass

    def classifyFilter(self, filterOrder):
        self.classifications = []
        # Wrap the zip iterator with tqdm for progress tracking
        for tf, impedanceCombo in tqdm(zip(self.transferFunctionsList, self.impedanceList),
                                        total=self.countTF(),
                                        desc="Computing filter parameters",
                                        unit="filter"):

            results = self.possibleFilterOrders[filterOrder](tf)

            if results['valid']:
                self.classifications.append(FilterClassification(
                    zCombo       = impedanceCombo,
                    transferFunc = tf,
                    valid= True,
                    fType        = results["fType"],
                    parameters   = results["parameters"]
                ))
            
            else:
                self.classifications.append(FilterClassification(
                    zCombo= impedanceCombo,
                    transferFunc= tf,
                    valid= False,
                    fType= results["fType"],
                    parameters= results["parameters"]
                ))



class FirstOrderParameters():
    def __init__(self, filterOrder):
        pass

class BiQuadParameters():
    def __init__(self, filterOrder):
        pass

class ThirdOrderParameters():
    def __init__(self, filterOrder):
        pass
