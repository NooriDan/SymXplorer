import subprocess
command = ['/foss/tools/bin/ngspice', '-D', 'ngbehavior=a', '-b', '-o', '/foss/designs/eda/SymXplorer/examples/tunable-tia/scripts/sizing/runs/simple/tia-bpf-1/tb_ac_4.log', '-r', '/foss/designs/eda/SymXplorer/examples/tunable-tia/scripts/sizing/runs/simple/tia-bpf-1/tb_ac_4.raw', '/foss/designs/eda/SymXplorer/examples/tunable-tia/scripts/sizing/runs/simple/tia-bpf-1/tb_ac_4.spice']

result = subprocess.run(command)
