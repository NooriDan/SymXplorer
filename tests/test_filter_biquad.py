import pytest
from sympy import symbols, simplify

from macanalog_symbolix.filter import Filter_Classifier

s = symbols('s')

import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level to DEBUG
    format="%(asctime)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Output logs to the console
    ]
)

@pytest.mark.parametrize("tf, expected", [
    # Valid cases
    (simplify((s**2) / (s**2 + s + 1)), {
        "valid": True, 
        "fType": "HP", 
        "parameters": {"wo": 1, "Q": 1, "bandwidth": 1, "K_LP": 0, "K_HP": 1, "K_BP": 0, "Qz": None, "Wz": None}
    }),
    (simplify(1 / (s**2 + s + 1)), {
        "valid": True, 
        "fType": "LP", 
        "parameters": {"wo": 1, "Q": 1, "bandwidth": 1, "K_LP": 1, "K_HP": 0, "K_BP": 0, "Qz": None, "Wz": None}
    }),
    (simplify(s / (s**2 + s + 1)), {
        "valid": True, 
        "fType": "BP", 
        "parameters": {"wo": 1, "Q": 1, "bandwidth": 1, "K_LP": 0, "K_HP": 0, "K_BP": 1, "Qz": None, "Wz": None}
    }),
    (simplify((s**2 + 1) / (s**2 + s + 1)), {
        "valid": True, 
        "fType": "BS", 
        "parameters": {"wo": 1, "Q": 1, "bandwidth": 1, "K_LP": 0, "K_HP": 1, "K_BP": 0, "Qz": None, "Wz": 1}
    }),
    (simplify((s**2 + s + 1) / (s**2 + 2*s + 4)), {
        "valid": True, 
        "fType": "GE", 
        "parameters": {"wo": 2, "Q": 1, "bandwidth": 2, "K_LP": 0.25, "K_HP": 1, "K_BP": (13.0)**0.5 / 4.0, "Qz": 1, "Wz": 1}
    }),

    # Invalid cases
    (simplify(s**3 / (s**2 + s + 1)), {
        "valid": False, 
        "fType": "INVALID-ORDER", 
        "parameters": None
    }),
    (simplify(1 / (s**3 + s**2 + s + 1)), {
        "valid": False, 
        "fType": "INVALID-ORDER", 
        "parameters": None
    }),
    (simplify((s**2 + s) / (s**3 + s**2 + s + 1)), {
        "valid": False, 
        "fType": "INVALID-ORDER", 
        "parameters": None
    }),
    (simplify((s + 1) / (s**2 + s + 1)), {
        "valid": False, 
        "fType": "INVALID-NUMER", 
        "parameters": None
    }),

    # Edge cases
    (simplify(0 / (s**2 + s + 1)), {
        "valid": False, 
        "fType": "INVALID-ORDER", 
        "parameters": None
    }),
    (simplify((s + 1) / (1)), {
        "valid": False, 
        "fType": "INVALID-ORDER", 
        "parameters": None
    }),
    (simplify((s**2 + s) / (s**2 + s + 1)), {
        "valid": False, 
        "fType": "INVALID-NUMER", 
        "parameters": None
    }),

    # Corner cases
    (simplify((5 * s**2 + 10) / (2 * s**2 + 4 * s + 8)), {
        "valid": True, 
        "fType": "BS", 
        "parameters": {"wo": 2, "Q": 1, "bandwidth": 2, "K_LP": 0.625, "K_HP": 2.5, "K_BP": 0, "Qz": None, "Wz": 2.236}
    }),
    (simplify((2 * s**2 + 3 * s + 1) / (2 * s**2 + 3 * s + 1)), {
        "valid": True, 
        "fType": "GE", 
        "parameters": {"wo": 0.707, "Q": 1, "bandwidth": 0.707, "K_LP": 1, "K_HP": 1, "K_BP": 1, "Qz": -1, "Wz": 0.707}
    }),
])
def test_getBiQuadParameters(tf, expected):
    # Create an instance of the Filter_Classifier
    logging.debug(f"tf = {tf}")
    classifier = Filter_Classifier()
    
    # Call the method
    result = classifier._getBiQuadParameters(tf)
    
    # Assert validity
    assert result["valid"] == expected["valid"]
    
    # Assert filter type
    assert result["fType"] == expected["fType"]
    logging.debug(f"type: {result["fType"]} vs {expected["fType"]}")

    
    # Assert parameters (if valid)
    if result["valid"]:
        for param, value in expected["parameters"].items():
            logging.debug(f"param = {param} : {value}, actual = {result["parameters"][param]}")
            if value is not None:
                assert result["parameters"][param] == pytest.approx(value)
            else:
                assert result["parameters"][param] is None
    else:
        assert result["parameters"] == expected["parameters"]
