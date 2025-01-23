import sympy
from   tqdm     import tqdm
from   sympy    import denom, numer, degree, symbols, sqrt, factor, cancel, Expr, Basic, Poly
from   typing   import Dict, List, Set, Tuple
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

    # Getters
    def hasTFs(self) -> bool:
        return len(self.transferFunctionsList) > 0
    
    def isClassified(self) -> bool:
        return len(self.transferFunctionsList) == len(self.classifications)
    
    def countTF(self) -> int:
        return len(self.transferFunctionsList)
    
    def countValid(self) -> Tuple[int, int]:
        countValid = 0
        for filter in self.classifications:
            if (filter.valid):
                countValid += 1
        return countValid, len(self.classifications)
    
    # Setters
    def clear(self) -> None:
        logger.info("!!!! Clearning the filter classifications !!!!")
        self.filterParameters = []
        self.classifications = []
        self.clusteredByType = {}
        self.countByType     = {}
    
    def addFilterType(self, types: list) -> None:
        for _type in types:
            self._fTypes.append(_type)

    def addTFs(self, transferFunctions, impedanceBatch) -> bool:
        if(len(impedanceBatch) != len(transferFunctions)):
            logger.warning("==== the TF and Z array size mismatch ====")
            return False
        
        for tf, zCombo in zip(transferFunctions, impedanceBatch):
            self.transferFunctionsList.append(tf)
            self.impedanceList.append(zCombo)
        return True
    
    def overwrite_classifications(self, classifications: List[Filter_Classification]):
        self.clear()
        self.classifications = classifications
        self.transferFunctionsList = [classification.transferFunc for classification in classifications]
        self.impedanceList         = [classification.zCombo for classification in classifications]

    # Inspection tools
    def summarizeFilterType(self, filterTypes: List[str] = None) -> Tuple[Dict[str, List[Filter_Classification]], Dict[str, int]]:
        if not self.isClassified():
            logger.warning("Classify the TFs first")

        if filterTypes is None:
            filterTypes = [t for t in self._fTypes] # overwrite by all the available types
        filterTypes.sort()

        print(f"summarizing for filters in {filterTypes}")

        for fType in filterTypes:
            self.clusteredByType[fType], self.countByType[fType] = self.findFilterInClassification(fType)
        return self.clusteredByType,  self.countByType

    def findFilterInClassification(self, filterType: str = None, denom_order: int = None, numer_order: int = None, printMessage=True) -> Tuple[List[Filter_Classification], int]:
        output = []
        count = 0
        # Define symbolic variable
        s = symbols('s')

        for entity in self.classifications:
            if (filterType is None) or (entity.fType == filterType):

                if (denom_order is not None) and entity.tf_denom_order != denom_order:
                    continue
                elif (numer_order is not None) and entity.tf_numer_order != numer_order:
                    continue

                output.append(entity)
                count += 1
        if printMessage:
            print(f"{filterType} : {len(output)}")

        return output, count

    # Filter Analysis Logic
    def classifyBiQuadFilters(self) -> None:
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
                filterOrder  = "BiQuad",
                tf_denom_order= results.get("tf_denom_order", -1),
                tf_numer_order= results.get("tf_numer_order", -1)
            ))

    def classifyFilter(self, filterOrder) -> None:
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

        corrected_classifcations: List[Filter_Classification] = []
        # classifications: List[Filter_Classification] = self.classifications

        for i, classification in tqdm(enumerate(self.classifications), 
                                      total=len(self.classifications), 
                                      unit="filter"):

            if not (classification.fType == fType):
                continue

            tf = classification.transferFunc

            if classification.filterOrder == "BiQuad":
                stable, type_correction = self.validate_second_order_stability(tf=tf)

            elif classification.filterOrder == "FirstOrder":
                stable, type_correction = self.validate_first_order_stability(tf=tf)
            
            corrected_type = fType + type_correction

            if not stable:
                self.classifications[i].valid = False
                self.classifications[i].fType = corrected_type
                self._fTypes.add(corrected_type)
                corrected_classifcations.append(self.classifications[i])
        
        if resummarize:
            self.summarizeFilterType() # Re-summarize all filters

        return corrected_classifcations
    
    def validate_first_order_stability(self, tf) -> Tuple[bool, str]:
        # Define symbolic variable
        s = symbols('s')
        # Extract numerator and denominator
        denominator = denom(tf).expand()  # Denominator of tf
        numerator = numer(tf).expand()    # Numerator of tf

        if degree(denominator, s) != 1 or degree(numerator, s) > 1:
            raise ValueError(f"Cannot use first_order_stability criteria on numer_deg = {degree(numerator, s)}, denom_deg = {degree(denominator, s)}")

        # Extract denominator coeffients
        a1 = denominator.coeff(s, 1)
        a0 = denominator.coeff(s, 0)
        # Extract numerator coeffients
        b1 = numerator.coeff(s, 1)
        b0 = numerator.coeff(s, 0)

        stable = True
        type_correction: str = ""
        # Checks if the pole could be in the RHP
        if sympy.ask(sympy.Q.negative(b1*b0)):
            stable = False
            type_correction += "-UNSTABLE-POLE"
        # Checks if zero could be in the RHP
        if sympy.ask(sympy.Q.negative(a1*a0)):
            stable = False
            type_correction += "-UNSTABLE-ZERO"

        return stable, type_correction
    
    def validate_second_order_stability(self, tf) -> Tuple[bool, str]:
        # Define symbolic variable
        s = symbols('s')
        # Extract numerator and denominator
        denominator = denom(tf).expand()  # Denominator of tf
        numerator = numer(tf).expand()    # Numerator of tf

        if degree(denominator, s) != 2 or degree(numerator, s) > 2:
            raise ValueError(f"Cannot use second_order_stability criteria on numer_deg = {degree(numerator, s)}, denom_deg = {degree(denominator, s)}")

        # Extract denominator coefficients
        a2 = denominator.coeff(s, 2)
        a1 = denominator.coeff(s, 1)
        # Extract numerator coefficients
        b2 = numerator.coeff(s, 2)
        b1 = numerator.coeff(s, 1)

        stable = True
        type_correction: str = ""
        # Checks if the pole pairs could be in the RHP
        if sympy.ask(sympy.Q.negative((b2*b1).simplify())):
            stable = False
            type_correction += "-UNSTABLE-POLE"
        # Checks if zero pairscould be in the RHP
        if sympy.ask(sympy.Q.negative((a2*a1).simplify())):
            stable = False
            type_correction += "-UNSTABLE-ZERO"

        return stable, type_correction

    def factorize_tsf(self, filterType: str = None, denom_order: int = None, numer_order: int = None):
        """Factorizes the numer and denom with respect to s and return two arrays (numerator and denominator) of type List[sympy.Basic]"""
        s: Basic = symbols("s")

        numer_factors: List[Basic] = []
        denom_factors: List[Basic] = []


        return numer_factors, denom_factors

    def decompose_tf(self, transfer_function: Expr) -> Tuple[Expr, List[Poly], List[Poly]]:
        """
        Decomposes the transfer function H(s) into:
        - K (constant factor),
        - Numer (list of normalized numerator factors),
        - Denom (list of normalized denominator factors).
        
        Each factor is normalized to have a leading coefficient of 1.
        """
        s = symbols("s")

        # Extract the numerator and denominator
        numerator, denominator = transfer_function.as_numer_denom()

        # Convert to Poly for better manipulation
        numerator_poly = Poly(numerator, s)
        denominator_poly = Poly(denominator, s)

        # Get leading coefficients
        numer_lead_coeff = numerator_poly.TC()
        denom_lead_coeff = denominator_poly.TC()

        # Normalize the leading coefficients
        constant_factor = numer_lead_coeff / denom_lead_coeff
        normalized_numerator = numerator / numer_lead_coeff
        normalized_denominator = denominator / denom_lead_coeff

        # Factorize strictly with respect to `s`
        numerator_factors   = sympy.factor(normalized_numerator).as_ordered_factors()
        denominator_factors = sympy.factor(normalized_denominator).as_ordered_factors()

        # Separate constant factors and normalize factors
        numer_factors_filtered: List[Poly] = []
        denom_factors_filtered: List[Poly] = []

        for factor in numerator_factors:
            if s not in factor.free_symbols:
                constant_factor *= factor
            else:
                poly_factor = Poly(factor, s)
                numer_factors_filtered.append(Poly(poly_factor / poly_factor.TC(), s))  # Normalize factor

        for factor in denominator_factors:
            if s not in factor.free_symbols:
                constant_factor /= factor
            else:
                poly_factor = Poly(factor, s)
                denom_factors_filtered.append(Poly(poly_factor / poly_factor.TC(), s))  # Normalize factor

        return constant_factor, numer_factors_filtered, denom_factors_filtered

    def is_poly_stable(self, poly: sympy.Poly) -> bool:

        if poly.degree() == 2:
            a2 = poly.as_dict().get((2,), None)
            a1 = poly.as_dict().get((1,), None)
            a0 = poly.as_dict().get((0,), None)
            if a1 is None or a2 is None:
                raise ValueError(f"Invalid second order poly: a2 = {a2} a1 = {a1} in {poly}")
            if sympy.ask(sympy.Q.negative(a2*a1).simplify()) or sympy.ask(sympy.Q.negative(a2*a0).simplify()):
                return False
            
        elif poly.degree() == 1:
            a1 = poly.as_dict().get((1,), None)
            a0 = poly.as_dict().get((0,), None)
            if a1 is None or a0 is None:
                raise ValueError(f"Invalid first order poly: a2 = {a1} a1 = {a0} in {poly}")
            
            if sympy.ask(sympy.Q.negative((a1*a0).simplify())):
                return False

        else:
            raise ValueError(f"Unsupported poly order: {poly.degree()}")
        
        return True
    
    # HIGHER ORDER FILTER ANALYSIS
    def get_3rd_order_lp(self, check_for_stability: bool = False) -> Dict:
        classifications, count = self.findFilterInClassification(denom_order=3, numer_order=0, printMessage=False)
        print(f"{count} candidates for 3rd-order LP")

        output = []
        count = 0
        count_valid = 0
        for classification in tqdm(classifications, total=len(classifications)):
            count += 1
            tf = classification.transferFunc
            k, numer, denom = self.decompose_tf(tf)

            valid = True
            for poly in denom:
                order = poly.degree()
                if order == 1: 
                    if (len(poly.as_dict()) != 2) or (check_for_stability and not self.is_poly_stable(poly)):
                        valid = False
                        break
                    
                elif order == 2:
                    if (len(poly.as_dict()) != 3) or (check_for_stability and not self.is_poly_stable(poly)):
                        valid = False
                        break

                else:
                    valid = False
                    break

            if valid:
                count_valid += 1
                # print(f"ID {count} - valid")

            output += [{
                "valid": valid,
                "zCombo": classification.zCombo,
                "classification" : classification,
                "k" :  k,
                "numer": numer,
                "denom": denom,
                "num-factor-count":len(numer),
                "denom-factor-count": len(denom)
            }]

        print(f"{count_valid} verified filters")

        return output

    def get_4th_order_bp(self, check_for_stability: bool = False) -> Dict:
        classifications, count = self.findFilterInClassification(denom_order=4, numer_order=2, printMessage=False)
        print(f"{count} candidates for 4th-order BP")

        output = []
        count = 0
        count_valid = 0
        for classification in tqdm(classifications, total=len(classifications)):
            count += 1
            tf = classification.transferFunc
            k, numer, denom = self.decompose_tf(tf)

            valid = True
            for poly in denom:
                order = poly.degree()
                if order == 1: 
                    if (len(poly.as_dict()) != 2) or (check_for_stability and not self.is_poly_stable(poly)):
                        valid = False
                        break
                    
                elif order == 2:
                    if (len(poly.as_dict()) != 3) or (check_for_stability and not self.is_poly_stable(poly)):
                        valid = False
                        break

                else:
                    valid = False
                    break

            if len(numer) != 1:
                valid = False

            if valid:
                count_valid += 1
                # print(f"ID {count} - valid")

            output += [{
                "valid": valid,
                "zCombo": classification.zCombo,
                "classification" : classification,
                "k" :  k,
                "numer": numer,
                "denom": denom,
                "num-factor-count":len(numer),
                "denom-factor-count": len(denom)
            }]

        print(f"{count_valid} verified filters")

        return output


    # HELPER FUNCTIONS (private)
    def _getBiQuadParameters(self, tf) -> Dict:
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
                    'parameters' : None
                    }

        # Extract denominator coefficients
        a2 = denominator.coeff(s, 2)
        a1 = denominator.coeff(s, 1)
        a0 = denominator.coeff(s, 0)

        # Validate filter form and coefficients
        if not all([a2, a1, a0]) or num_order > 2 or den_order > 2:
            return {'valid': False,
                    'fType': "X-INVALID-ORDER",
                    'parameters': None,                     
                    'tf_numer_order' : num_order,
                    'tf_denom_order' : den_order}

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
            },                     
            'tf_numer_order' : num_order,
            'tf_denom_order' : den_order
        }

    def _getFirstOrderParameters(self, tf) -> Dict:
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
                    'parameters' : None,                     
                    'tf_numer_order' : num_order,
                    'tf_denom_order' : den_order}
        
        # logger.debug(f"num_order = {num_order}")
        
        # Extract denominator coeffients
        a1 = denominator.coeff(s, 1)
        a0 = denominator.coeff(s, 0)
        
        # Validate filter form and coefficients
        if (not all([a1, a0])) or num_order >= 2 or den_order >= 2:
            return {'valid': False,
                    'fType': "X-INVALID-ORDER",
                    'parameters': None,                     
                    'tf_numer_order' : num_order,
                    'tf_denom_order' : den_order}
        
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
                    'parameters': None,                     
                    'tf_numer_order' : num_order,
                    'tf_denom_order' : den_order}

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
            },                     
            'tf_numer_order' : num_order,
            'tf_denom_order' : den_order
        }

    def _getThirdOrderParameters(self, tf):
        logger.debug(" === inside THIRD ORDER Parameter Computation === ")

        pass
