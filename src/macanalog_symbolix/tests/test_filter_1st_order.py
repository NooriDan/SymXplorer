import pytest
from sympy import symbols, simplify
from macanalog_symbolix.filter import Filter_Classifier  # Replace `your_module` with the actual module name

import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level to DEBUG
    format="%(asctime)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Output logs to the console
    ]
)


# Define symbolic variable
s = symbols('s')

@pytest.mark.parametrize("tf, expected", [
    # Valid cases
    (simplify((s) / (s + 1)), {"valid": True, "fType": "HP", "parameters": {"wo": 1, "wz": 0, "K": 1}}),
    (simplify(1 / (s + 1)), {"valid": True, "fType": "LP", "parameters": {"wo": 1, "wz": 0, "K": 1}}),
    (simplify((s + 1) / (s + 2)), {"valid": True, "fType": "BP", "parameters": {"wo": 2, "wz": 1, "K": 1}}),
    (simplify((2 * s + 3) / (s + 4)), {"valid": True, "fType": "BP", "parameters": {"wo": 4, "wz": 1.5, "K": 2}}),
    (simplify(5 / (s + 5)), {"valid": True, "fType": "LP", "parameters": {"wo": 5, "wz": 0, "K": 5}}),

    # Invalid cases
    (simplify(s**2 / (s + 1)), {"valid": False, "fType": "INVALID-ORDER", "parameters": None}),
    (simplify(1 / (s**2 + s + 1)), {"valid": False, "fType": "INVALID-ORDER", "parameters": None}),
    (simplify((s**2 + 1) / (s + 1)), {"valid": False, "fType": "INVALID-ORDER", "parameters": None}),
    (simplify((s + 1) / (s**2 + s + 1)), {"valid": False, "fType": "INVALID-ORDER", "parameters": None}),
    (simplify((s + 2) / (s + 2) * (s + 3)), {"valid": False, "fType": "INVALID-ORDER", "parameters": None}),
    
    # Edge cases
    (simplify((s + 1) / (1)), {"valid": False, "fType": "INVALID-ORDER", "parameters": None}),
    (simplify((0) / (s + 1)), {"valid": False, "fType": "INVALID-ORDER", "parameters": None}),
    (simplify((s + 0) / (s + 0)), {"valid": False, "fType": "INVALID-ORDER", "parameters": None}),
    
    # Corner cases
    (simplify((3 * s) / (4 * s + 2)), {"valid": True, "fType": "HP", "parameters": {"wo": 0.5, "wz": 0, "K": 0.75}}),
    (simplify((10 * (s + 2)) / (5 * (s + 4))), {"valid": True, "fType": "BP", "parameters": {"wo": 4, "wz": 2, "K": 2}})
])
def test_getFirstOrderParameters(tf, expected):
    # Create an instance of the Filter_Classifier
    classifier = Filter_Classifier()
    
    # Call the method
    result = classifier._getFirstOrderParameters(tf)
    
    # Assert validity
    assert result["valid"] == expected["valid"]
    
    # Assert filter type
    assert result["fType"] == expected["fType"]
    
    # Assert parameters (if valid)
    if result["valid"]:
        for param in expected["parameters"]:
            assert result["parameters"][param] == pytest.approx(expected["parameters"][param])
    else:
        assert result["parameters"] == expected["parameters"]


# Test Cases for addTFs
def test_addTFs():
    classifier = Filter_Classifier()
    transfer_functions = [simplify(s / (s + 1)), simplify(1 / (s + 1))]
    impedances = [{"R": 1, "C": 0.5}, {"R": 2, "L": 0.1}]
    
    # Add transfer functions and impedances
    result = classifier.addTFs(transfer_functions, impedances)
    
    # Validate result
    assert result is True
    assert len(classifier.transferFunctionsList) == 2
    assert len(classifier.impedanceList) == 2

