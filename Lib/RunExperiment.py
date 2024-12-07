from Global import *            # Global vairables (symbolic)
from CircuitSetUp import CircuitSetUp
from Experiment import SymbolixExperimentCG

_output = [Vop, Von]
_input  = [Iip, Iin]
T_types  = ["simple", "some_parasitic", "full_parasitic"]

for T_type in T_types:
    experimentName = "TIA"
    experimentName += "_" + T_type

    circuit = CircuitSetUp(_output, _input,
                                transmissionMatrixType=T_type)
    circuit.solve()

    impedanceKeys = circuit.baseHsDict.keys()

    for i, key in enumerate(impedanceKeys, 1):
        print(f"==> Running the {experimentName} Experiment for {key} ({i}/{len(impedanceKeys)})\n")
        experiment = SymbolixExperimentCG(experimentName, circuit)
        experiment.computeTFs(comboKey=key)
        #
        experiment.classifier.classifyBiQuadFilters()
        experiment.classifier.summarizeFilterType()

        experiment.reportSummary(experimentName, key)
        experiment.compilePDF()