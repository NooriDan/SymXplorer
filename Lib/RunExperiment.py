from Global import *            # Global vairables (symbolic)
from BaseTF import BaseTransferFunction
from Experiment import SymbolixExperimentCG


_output = [Vop, Von]
_input  = [Iip, Iin]
T_type  = "simple"
experimentName = "TIA"
experimentName += "_" + T_type

baseTF = BaseTransferFunction(_output, _input,
                              transmissionMatrixType=T_type)
print("Solving th")
baseTF.solve()

Z_arr_list = [
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
# Z_arr_list = ["Z1_ZL"]

for i, Z_arr in enumerate(Z_arr_list, 1):
    print(f"==> Running the {experimentName} Experiment for {Z_arr} ({i}/{len(Z_arr_list)})\n")
    experiment = SymbolixExperimentCG(experimentName, baseTF)
    experiment.computeTFs(comboKey=Z_arr)
    #
    experiment.classifier.classifyBiQuadFilters()
    experiment.classifier.summarizeFilterType()

    experiment.reportSummary(experimentName, Z_arr)
    experiment.compilePDF()

    experiment.export(f"{T_type}_{Z_arr}")
