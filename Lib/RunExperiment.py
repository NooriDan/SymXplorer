from Global import *            # Global vairables (symbolic)
from CircuitSetUp import CircuitSetUp
from Experiment import SymbolixExperimentCG

_output = [Vop, Von]
_input  = [Iip, Iin]
T_types  = ["simple", "some_parasitic", "full_parasitic"]

for T_type in T_types:
    experimentName = "TIA"
    experimentName += "_" + T_type

    baseTF = CircuitSetUp(_output, _input,
                                transmissionMatrixType=T_type)
    baseTF.solve()

    impedanceKeys = [
                "Z1_ZL",
                "Z2_ZL",
                "Z3_ZL",
                "Z4_ZL",
                "Z5_ZL",
                "Z1_Z2_ZL",
                "Z1_Z3_ZL",
                "Z1_Z4_ZL",
                "Z1_Z5_ZL",
                "Z2_Z3_ZL",
                "Z2_Z4_ZL",
                "Z2_Z5_ZL",
                "Z3_Z4_ZL",
                "Z3_Z5_ZL",
                "Z4_Z5_ZL"
                ]

    # Z_arr_list = Z_arr_list[0:2]
    # impedanceKeys = ["Z5_ZL"]

    for i, key in enumerate(impedanceKeys, 1):
        print(f"==> Running the {experimentName} Experiment for {key} ({i}/{len(impedanceKeys)})\n")
        experiment = SymbolixExperimentCG(experimentName, baseTF)
        experiment.computeTFs(comboKey=key)
        #
        experiment.classifier.classifyBiQuadFilters()
        experiment.classifier.summarizeFilterType()

        experiment.reportSummary(experimentName, key)
        experiment.compilePDF()