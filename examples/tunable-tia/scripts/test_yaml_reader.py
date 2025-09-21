from symxplorer.designer_tools.domains import Project_Setup

if __name__ == "__main__":
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