from symxplorer.designer_tools.domains import Project_Setup
from symxplorer.logging import setup_loggers

if __name__ == "__main__":
    _ = setup_loggers()
    ws_root = "/foss/designs/eda/SymXplorer"
    project_setup_yaml = f"{ws_root}/examples/tunable-tia/ihp-sg13g2/spice/project_setup.yaml"

    # ----------------------------
    # Instantiation
    # ----------------------------
    project = Project_Setup.from_yaml(project_setup_yaml)

    # ----------------------------
    # Getter Methods
    # ----------------------------
    project.summary()