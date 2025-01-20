import sympy
from   tqdm     import tqdm
from   sympy    import denom, numer, degree, symbols, sqrt, factor, cancel
from   typing   import Dict, List, Set
# Custom Imports
from .domains import Filter_Classification

import logging

import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Filter_Classifier():
    """Implementation of algorithm 2"""
    def __init__(self, transferFunctionList: List = [], impedanceBatch: List = [], fTypes: Set[str] = set(["HP", "BP", "LP", "BS", "GE", "AP", "X-INVALID-NUMER", "X-INVALID-WZ", "X-INVALID-ORDER", "X-PolynomialError"])):
        self.transferFunctionsList = transferFunctionList
        self.impedanceList         = impedanceBatch
        self.filterParameters = []
        self._fTypes = fTypes
        # To be computed
        self.classifications:  List[Filter_Classification] = []
        self.clusteredByType:  Dict[str, List[Filter_Classification]]= {}
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
            logger.warning("==== the TF and Z array size mismatch ====")
            return False
        
        for tf, zCombo in zip(transferFunctions, impedanceBatch):
            self.transferFunctionsList.append(tf)
            self.impedanceList.append(zCombo)
        return True

    def clear(self):
        logger.info("!!!! Clearning the filter classifications !!!!")
        self.filterParameters = []
        self.classifications = []
        self.clusteredByType = {}
        self.countByType     = {}
    
    def addFilterType(self, types: list):
        for _type in types:
            self._fTypes.append(_type)

    def summarizeFilterType(self, filterTypes: List[str] = None):
        if not self.isClassified():
            logger.warning("Classify the TFs first")

        if filterTypes is None:
            filterTypes = [t for t in self._fTypes] # overwrite by all the available types
        filterTypes.sort()

        print(f"summarizing for filters in {filterTypes}")

        for fType in filterTypes:
            self.clusteredByType[fType], self.countByType[fType] = self._findFilterInClassification(fType)
        return self.clusteredByType,  self.countByType

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
                                        desc="Computing Biquad filter parameters",
                                        unit="filter"):

            results = self._getBiQuadParameters(tf)

            self.classifications.append(Filter_Classification(
                zCombo       = impedanceCombo,
                transferFunc = tf,
                valid        = results.get('valid', False),
                fType        = results.get("fType", None),
                parameters   = results.get("parameters", None),
                filterOrder  = "BiQuad"
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
        try:
            den_order = degree(denominator, s)
            num_order = degree(numerator, s)
        except sympy.PolynomialError:

            return {'valid' : False,
                    'fType' : "X-PolynomialError",
                    'parameters' : None}

        # Extract denominator coefficients
        a2 = denominator.coeff(s, 2)
        a1 = denominator.coeff(s, 1)
        a0 = denominator.coeff(s, 0)

        # Validate filter form and coefficients
        if not all([a2, a1, a0]) or num_order > 2 or den_order > 2:
            return {'valid': False,
                    'fType': "X-INVALID-ORDER",
                    'parameters': None}

        # Compute natural frequency (wo), quality factor (Q), and bandwidth
        wo = sqrt(cancel(a0 / a2))
        Q = cancel((a2 / a1) * wo)
        bandwidth = (wo / Q)

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
                fType = "X-INVALID-NUMER"
            case 0b110:     # This situation is not accounted for in N_XY(s) scenarios
                fType = "X-INVALID-NUMER"
            case _:         # catches other combos (i.e., 110, 011)
                fType = "X-INVALID-NUMER"


        # Compute zero's natural frequency (wz) if applicable
        valid = True
        if b2 != 0 and b0 != 0:
            wz = sqrt(cancel(b0 / b2))
        else:
            wz = None

        # Compute filter constants
        K_HP = cancel(b2 / a2)
        K_BP = cancel(b1 / (a2 * bandwidth))
        K_LP = cancel(b0 / (a2 * wo**2))


        # # compare wz to wo
        if (fType in ["BS", "GE"]) and (wz != wo):
            valid = False
            fType = "X-INVALID-WZ"

        # Additional parameter (Qz) for GE filters
        Qz = cancel((b2 / b1) * wo) if (b1 != 0 and fType == "GE") else None

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
        # logger.debug(" === inside FIRST ORDER Parameter Computation === ")
        # Define symbolic variable
        s = symbols('s')
        
        # Extract numerator and denominator
        denominator = denom(tf).expand()  # Denominator of tf
        numerator = numer(tf).expand()    # Numerator of tf

        # logger.debug(f"Analyzing transfer function: {tf}")
        
        # Determine orders
        try:
            den_order = degree(denominator, s)
            num_order = degree(numerator, s)
        except sympy.PolynomialError:

            return {'valid' : False,
                    'fType' : "PolynomialError",
                    'parameters' : None}
        
        # logger.debug(f"num_order = {num_order}")
        
        # Extract denominator coeffients
        a1 = denominator.coeff(s, 1)
        a0 = denominator.coeff(s, 0)
        
        # Validate filter form and coefficients
        if (not all([a1, a0])) or num_order >= 2 or den_order >= 2:
            return {'valid': False,
                    'fType': "X-INVALID-ORDER",
                    'parameters': None}
        
        # Extract numerator coeffients
        b1 = numerator.coeff(s, 1)
        b0 = numerator.coeff(s, 0)

        numeratorState = ((b1 != 0) << 1) | (b0 != 0)
        match numeratorState:
            case 0b10: # b1 s^1
                fType = "HP"
            case 0b11: # b1 *s  + b0
                fType = "BP"
            case 0b01: # b0
                fType = "LP"
            case _: 
                fType = "X-INVALID-NUMER"
                return {'valid': False,
                    'fType': "X-INVALID-ORDER",
                    'parameters': None}

        wo = a0/a1
        if b1 != 0:
            wz = b0/b1
            K =  b1/a1
        else:
            wz = None
            K  = b0/a1
        # Return computed parameters
        return {
            "valid": True,
            "fType": fType,
            "parameters": {
                "wo" : wo,
                "wz" : wz,
                "K": K
            }
        }

    def _getThirdOrderParameters(self, tf):
        logger.debug(" === inside THIRD ORDER Parameter Computation === ")

        pass

    def classifyFilter(self, filterOrder):
        self.classifications = []
        # Wrap the zip iterator with tqdm for progress tracking
        for tf, impedanceCombo in tqdm(zip(self.transferFunctionsList, self.impedanceList),
                                        total=self.countTF(),
                                        desc=f"Computing filter parameters for {filterOrder}",
                                        unit="filter"):

            results = self.possibleFilterOrders[filterOrder](tf)

            self.classifications.append(Filter_Classification(
                zCombo       = impedanceCombo,
                transferFunc = tf,
                valid        = results.get('valid', None),
                fType        = results.get("fType", None),
                parameters   = results.get("parameters", None),
                filterOrder  = filterOrder
            ))

    def validate_stability(self, fType: str, resummarize: bool = True) -> List[Filter_Classification]:
        """Validates stability of the filter classifications and its tf (1st and 2nd)
        Returns the list of unstable filters. Updates the self.classifications field in-place"""
        # Define symbolic variable
        s = symbols('s')
        corrected_classifcations: List[Filter_Classification] = []
        # classifications: List[Filter_Classification] = self.classifications

        for i, classification in tqdm(enumerate(self.classifications), 
                                      total=len(self.classifications), 
                                      unit="filter"):

            corrected_type = fType

            if not (classification.fType == fType):
                continue

            tf = classification.transferFunc

            # Extract numerator and denominator
            denominator = denom(tf).expand()  # Denominator of tf
            numerator = numer(tf).expand()    # Numerator of tf

            if classification.filterOrder == "BiQuad":
                # Extract denominator coefficients
                a2 = denominator.coeff(s, 2)
                a1 = denominator.coeff(s, 1)
                # Extract numerator coefficients
                b2 = numerator.coeff(s, 2)
                b1 = numerator.coeff(s, 1)

                stable = True
                # Checks if the pole pairs could be in the RHP
                if sympy.ask(sympy.Q.negative(b2*b1)):
                    stable = False
                    corrected_type += "-UNSTABLE-POLE"
                # Checks if zero pairscould be in the RHP
                if sympy.ask(sympy.Q.negative(a2*a1)):
                    stable = False
                    corrected_type += "-UNSTABLE-ZERO"


            elif classification.filterOrder == "FirstOrder":
                # Extract denominator coeffients
                a1 = denominator.coeff(s, 1)
                a0 = denominator.coeff(s, 0)
                # Extract numerator coeffients
                b1 = numerator.coeff(s, 1)
                b0 = numerator.coeff(s, 0)

                stable = True
                # Checks if the pole could be in the RHP
                if sympy.ask(sympy.Q.negative(b1*b0)):
                    stable = False
                    corrected_type += "-UNSTABLE-POLE"
                # Checks if zero could be in the RHP
                if sympy.ask(sympy.Q.negative(a1*a0)):
                    stable = False
                    corrected_type += "-UNSTABLE-ZERO"

            if not stable:
                self.classifications[i].valid = False
                self.classifications[i].fType = corrected_type
                self._fTypes.add(corrected_type)
                corrected_classifcations.append(self.classifications[i])
        
        if resummarize:
            self.summarizeFilterType() # Re-summarize all filters

        return corrected_classifcations
    
    def validate_first_order_stability(self, tf):

        # Define symbolic variable
        s = symbols('s')
        # Extract numerator and denominator
        denominator = denom(tf).expand()  # Denominator of tf
        numerator = numer(tf).expand()    # Numerator of tf

        if degree(denominator)> 1 or degree(numerator)> 1:
            raise ValueError(f"Cannot use first_order_stability criteria on numer_deg = {degree(numerator)}, denom_deg = {degree(denominator)}")

        # Extract denominator coeffients
        a1 = denominator.coeff(s, 1)
        a0 = denominator.coeff(s, 0)
        # Extract numerator coeffients
        b1 = numerator.coeff(s, 1)
        b0 = numerator.coeff(s, 0)

        stable = True
        type_correection = ""
        # Checks if the pole could be in the RHP
        if sympy.ask(sympy.Q.negative(b1*b0)):
            stable = False
            type_correection += "-UNSTABLE-POLE"
        # Checks if zero could be in the RHP
        if sympy.ask(sympy.Q.negative(a1*a0)):
            stable = False
            type_correection += "-UNSTABLE-ZERO"

        return stable, type_correection

