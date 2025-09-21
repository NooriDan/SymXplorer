import yaml

with open("/foss/designs/eda/SymXplorer/examples/tunable-tia/ihp-sg13g2/spice/project_setup.yaml") as f:
    data = yaml.safe_load(f)

max_cap = data["globals"]["MAX_CAP_SIZE"]
data["params"]["dut"][0]["range"][1] = max_cap