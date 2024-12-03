from Global import *
from Filter import FilterClassification

class FileSave:
    """
    A utility class for generating LaTeX reports.
    """
    def __init__(self, outputDirectory="Outputs"):
        self.outputDirectory = outputDirectory
        os.makedirs(outputDirectory, exist_ok=True)

    def generateLaTeXReport(self, filter_classifications: List[FilterClassification], output_filename="Report", newpage=False, subFolder="NONE"):
        """
        Generates a LaTeX report from the filter classifications and saves it to a file.

        :param filter_classifications: List of FilterClassification objects.
        :param output_filename: The name of the output LaTeX file (without extension).
        :param newpage: If True, adds a page break after each filter.
        """
        header = r"""
        \documentclass{article}
        \usepackage{amsmath}
        \usepackage{geometry}
        \geometry{landscape, a1paper, margin=1in}  % Adjust paper size and margins
        \begin{document}
        """

        footer = r"\end{document}"

        # LaTeX filename
        targetFolder = self.outputDirectory
        if subFolder!= "NONE":
            targetFolder += f"/{subFolder}"
            os.makedirs(targetFolder, exist_ok=True)
        
        output_filename = os.path.join(targetFolder, output_filename + ".tex")

        with open(output_filename, "w") as latex_file:
            latex_file.write(header)
            latex_file.write(f"\\section*{{Experiment: {subFolder.replace("_", " ")}}}\n")

            for i, classification in enumerate(filter_classifications, 1):
                latex_file.write(f"\\subsection*{{Filter {i}}}\n")
                if classification.valid:
                    latex_file.write(f"\\textbf{{Filter Type:}} {classification.fType} \\\\ \n")

                    latex_file.write(f"\\textbf{{$Z(s)$:}} ${latex(classification.zCombo)}$ \\\\ \n")
                    # Transfer function
                    latex_file.write(f"\\textbf{{$H(s)$:}} ${latex(classification.transferFunc)}$ \\\\ \n")
                    
                    # Ensure parameters are written in math mode
                    latex_file.write(f"\\textbf{{Q:}} ${latex(classification.parameters['Q'])}$ \\\\ \n")
                    latex_file.write(f"\\textbf{{$\\omega_0$:}} ${latex(classification.parameters['wo'])}$ \\\\ \n")
                    latex_file.write(f"\\textbf{{Bandwidth:}} ${latex(classification.parameters['bandwidth'])}$ \\\\ \n")

                    if (classification.fType == "GE"):
                        latex_file.write(f"\\textbf{{Qz:}} ${latex(classification.parameters['Qz'])}$ \\\\ \n")

                else:
                    latex_file.write("Invalid filter \\\\ \n")
                    latex_file.write(f"\\textbf{{$Z(s)$:}} ${latex(classification.zCombo)}$ \\\\ \n")
                    latex_file.write(f"\\textbf{{$H(s)$:}} ${latex(classification.transferFunc)}$ \\\\ \n")

                # Add a page break if newpage is requested
                if newpage:
                    latex_file.write("\\newpage\n")

            latex_file.write(footer)

        print(f"LaTeX report saved to: {output_filename}")
