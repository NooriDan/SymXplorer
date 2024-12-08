# from Global import *
import sympy
from sympy import latex, symbols, Matrix
from Filter import FilterClassification
from typing import Dict, List
import subprocess
import os
import platform
import pickle

# UTIL CLASSES
class TransmissionMatrix:
    def __init__(self, defaultType="Symbolic"):
        self.defaultType = defaultType

        # Variables global to the class
        gm, ro, Cgd, Cgs    = symbols('g_m r_o C_gd C_gs')
        a11, a12, a21, a22  = symbols('a11 a12 a21 a22')
        s = symbols("s")

        self.transmissionMatrix ={
        "simple"          : Matrix([[0, -1/gm],[0, 0]]),
        "symbolic"        : Matrix([[a11, a12],[a21, a22]]),
        "some_parasitic"  : Matrix([[-1/(gm*ro), -1/gm],[0, 0]]),
        "full_parasitic"  : Matrix([[(1/ro + s*Cgd)/(s*Cgd - gm), 1/(s*Cgd - gm)],[(Cgd*Cgs*ro*s + Cgd*gm*ro + Cgs + Cgd)*s/(s*Cgd - gm), (Cgs+Cgd)*s/(s*Cgd - gm)]])
        }

    def getTranmissionMatrix(self, transmissionMatrixType = "symbolic"):
            if self.transmissionMatrix.get(transmissionMatrixType) is None:
                 print("Invalide Transmission Matrix Selected")
                 raise KeyError
            return self.transmissionMatrix.get(transmissionMatrixType)

class Impedance:
    def __init__(self, name: str):
        self.name = str(name)
        self.Z: sympy.Basic = sympy.Symbol(f"Z_{name}")

        s = sympy.symbols("s")
        self.Z_R: sympy.Basic = sympy.symbols(f"R_{name}")
        self.Z_L: sympy.Basic = s * sympy.symbols(f"L_{name}")
        self.Z_C: sympy.Basic = 1 / (s * sympy.symbols(f"C_{name}"))

        # to be computed
        self.allowedConnections: List[sympy.Basic] = []

        # control variables
        self.zDictionary: Dict[str, sympy.Basic] = {
            "R" : self.Z_R,
            "L" : self.Z_L,
            "C" : self.Z_C
        }
        self.conectionSymbols: Dict[str, function] = {
            "|" : self.parallel,
            "+" : self.series
        }
        self.startOfFunctionToken: str  = "*START*"
        self.endOfFunctionToken:   str  = "*END*"

    def simplify(self):
        for i, _impedance in enumerate(self.allowedConnections):
            self.allowedConnections[i] = sympy.simplify(_impedance)
    
    def series(self, list_of_impedances: List[sympy.Basic]):
        
        equivalentZ = list_of_impedances[0]
        for impedance in list_of_impedances[1:]:
            equivalentZ += impedance
        
        return sympy.simplify(equivalentZ)
    
    def parallel(self,list_of_impedances: List[sympy.Basic]):
        
        equivalentG = 1/list_of_impedances[0]
        for impedance in list_of_impedances[1:]:
            equivalentG += 1/impedance
        
        return sympy.simplify(1/equivalentG)
    
    def setAllowedImpedanceConnections(self, allowedConnections_texts: List[str]):
        """
        Reads from allowedConnections_texts and converts each string representation
        of the impedance connections to its symbolic expression.
        """
        for conn_text in allowedConnections_texts:
            parsed = self.parse_expression(conn_text)
            self.allowedConnections.append(parsed)

    def parse_expression(self, expression: str):
        """
        Parse a string expression to build the symbolic impedance representation.
        """
        # print(f"Original Expression: {expression}")

        # Replace component symbols with their symbolic equivalents
        for key, value in self.zDictionary.items():
            expression = expression.replace(key, f"self.zDictionary['{key}']")
        # print(f"1 - After Replacing Symbols: {expression}")

        # Handle nested parentheses
        while "(" in expression:
            start = expression.rfind("(")
            end = expression.find(")", start)
            if end == -1:
                raise ValueError("Unmatched parentheses in expression.")
            inner = expression[start + 1:end]
            inner_parsed = self._replace_operators(inner)
            expression = expression[:start] + inner_parsed + expression[end + 1:]
            # print(f" 2 - After Parsing Parentheses: {expression}")

        # Final replacement for top-level operators
        expression = self._replace_operators(expression)
        # print(f"Final Parsed Expression: {expression}")

        # Safely evaluate the expression
        try:
            expression = expression.replace(self.startOfFunctionToken, "(")
            expression = expression.replace(self.endOfFunctionToken, ")")
            result = sympy.simplify(eval(expression))
        except Exception as e:
            raise ValueError(f"Failed to parse expression: {expression}. Error: {e}")

        return result

    def _replace_operators(self, expression: str):
        """
        Replace connection operators in the expression:
        - "|" -> "self.parallel([...])"
        - "+" -> "self.series([...])"
        & -> '('
        """
        if "+" in expression:
            terms = expression.split("+")
            replaced = ", ".join(terms)
            return f"self.series{self.startOfFunctionToken} [{replaced}] {self.endOfFunctionToken}"  # Corrected with parentheses

        if "|" in expression:
            terms = expression.split("|")
            replaced = ", ".join(terms)
            return f"self.parallel {self.startOfFunctionToken} [{replaced}] {self.endOfFunctionToken}"  # Corrected with parentheses

        # If no operators are found, return the expression as is
        return expression

class FileSave:
    """
    A utility class for generating LaTeX reports.
    """
    def __init__(self, outputDirectory="Runs"):
        self.outputDirectory = outputDirectory
        os.makedirs(outputDirectory, exist_ok=True)

        self.header = r"""
        \documentclass{article}
        \usepackage{amsmath}
        \usepackage{geometry}
        \usepackage{longtable}
        \geometry{landscape, a3paper, margin=0.5in}  % Adjust paper size and margins
        \begin{document}
        """
        self.texFiles: List[str] = []
    
        self.footer = r"\end{document}"

    # Saving the object
    def export(self, object, file: str) -> None:
        """Exports the current object to a file using pickle."""
        file = f"{self.outputDirectory}/{file}_object"
        try:
            with open(file, 'wb') as f:
                pickle.dump(object, f)
            print(f"Object exported to {file}")
        except (IOError, pickle.PickleError) as e:
            print(f"Failed to export object to {file}: {e}")

    def import_from(self, file: str):
        """Imports an object from a file."""
        file = f"{self.outputDirectory}/{file}_object"
        try:
            with open(file, 'rb') as f:
                obj = pickle.load(f)
            print(f"Object imported from {file}")
            return obj
        except (IOError, pickle.PickleError) as e:
            print(f"Failed to import object from {file}: {e}")
            return None

    # Creating LaTeX Reports
    def compile(self):
        print("\n=== Compiling the reports to PDF ===")
        for fileName in self.texFiles:
            print(f"----** compiling {fileName} **----")
            self.compileToPDF(fileName)
        print("=== Compiling DONE ===\n")

    @staticmethod
    def compileToPDF(latex_filepath, cleanup=True):
        # Get the directory and filename without the extension
        directory, filename = os.path.split(latex_filepath)
        basename, _ = os.path.splitext(filename)

        # Save the current working directory
        original_directory = os.getcwd()

        # Ensure the directory exists before trying to change into it
        if directory and not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it doesn't exist

        # Change to the directory of the LaTeX file
        if directory:
            os.chdir(directory)

        try:
            # Compile the LaTeX file into a PDF, suppressing stdout and stderr
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", filename],
                stdout=subprocess.DEVNULL,  # Suppress stdout
                stderr=subprocess.DEVNULL,  # Suppress stderr
                check=True
            )
            print(f"PDF generated: {os.path.join(directory, basename + '.pdf')}")
        except subprocess.CalledProcessError as e:
            print(f"Error during LaTeX compilation: {e}")
        finally:
            # Optionally clean up auxiliary files generated during compilation
            if cleanup:
                aux_files = [basename + ext for ext in [".aux", ".log", ".out"]]
                for aux_file in aux_files:
                    if os.path.exists(aux_file):
                        os.remove(aux_file)

            # Change back to the original directory after the LaTeX compilation
            os.chdir(original_directory)


    def createSubFolder(self, filename, subFolder, fileType=".tex"):
        # LaTeX filename
        targetFolder = self.outputDirectory
        if subFolder!= "NONE":
            targetFolder += f"/{subFolder}"
            os.makedirs(targetFolder, exist_ok=True)

        return os.path.join(targetFolder, filename + fileType)

    def saveLatexFile(self, latex_content: str, output_filename: str):
        # LaTeX footer
        latex_content += self.footer
        # Saving the LaTeX content to a .tex file
        with open(output_filename, "w") as tex_file:
            tex_file.write(latex_content)

        print(f"LaTeX report generated and saved to: {output_filename}")
        self.texFiles.append(output_filename)

    def generateLaTeXSummary(
        self,
        baseHs: sympy.Basic,
        filterClusters: Dict[str, List[FilterClassification]],
        output_filename="Report",
        newpage=False,
        subFolder="NONE"
    ):
        output_filename = self.createSubFolder(output_filename, subFolder)

        # LaTeX header and document setup
        latex_content = self.header
        latex_content += r"""
                        \title{Filter Summary Report: """ + f"{subFolder.replace('_', ',')}" + r"""}
                        \author{Generated by MacAnalog-Symbolix}
                        \maketitle
                        % Table of contents
                        \tableofcontents
                        \newpage
                        """
        
        latex_content += f"\\section{{Examined $H(z)$ for {subFolder.replace('_', ' ')}: ${str(latex(baseHs))}$ }}\\ \n"  # Use fType as the section name
        latex_content += f"\\textbf{{\[H(z) = {str(latex(baseHs))}\] }}\\ \n"  # Use fType as the section name

        # Iterate over the filter clusters
        for fType, filters in filterClusters.items():
            latex_content += f"\\section{{{fType}}}\\ \n"  # Use fType as the section name
            
            # Iterate over each filter classification in the current cluster
            for i, _filter in enumerate(filters, 1):

                latex_content += f"\\subsection{{{_filter.fType}-{i} $Z(s) = {str(latex(_filter.zCombo))}$ }} \\ \n"
                
                # Transfer function
                latex_content += f"\\textbf{{\[H(s) = {str(latex(_filter.transferFunc))}\] }} \\ \n"
                
                # Parameters
                if _filter.parameters:
                    latex_content += "\\textbf{Parameters:}\\\\ \n\n"
                    for param, value in _filter.parameters.items():
                        latex_content += f"{param.replace('_', '-')}: ${str(latex(value))}$\\ \n\n"
                    latex_content += "\\ \n\n"
                
                if newpage:
                    latex_content += "\\newpage\n"  # Optional new page after each filter

        self.saveLatexFile(latex_content, output_filename)

    def generateSummaryTable(self, clusterByType: Dict[str, List[FilterClassification]], output_filename, subFolder = "None"):

        output_filename = self.createSubFolder(output_filename, subFolder)

        # LaTeX header and document setup
        latex_content = self.header
        latex_content += r"""
                        \title{Filter Table: """ + f"{subFolder.replace('_', ',')}" + r"""}
                        \author{Generated by MacAnalog-Symbolix}
                        \maketitle
                        \newpage
                        """
                
        orderedFilters = [filter_instance for filters in clusterByType.values() for filter_instance in filters]

        latex_content += self._generate_latex_section("Filter Summary")
        latex_content += self._generate_latex_table(orderedFilters, "Filters by Type")

        self.saveLatexFile(latex_content, output_filename)

    def _generate_latex_table(self, filter_list: List[FilterClassification], caption):
        # Table header
        latex_table = r"""\centering
        \begin{longtable}{|c|c|c|c|c|c|}
        \hline
        Filter Order & Z Combo & Transfer Function & Valid & Filter Type & Parameters \\ \hline
        """

        # Populate table rows
        for f in filter_list:
            # Parameters
            parameters_i = "NONE"
            if f.parameters:
                parameters_i = ""
                for param, value in f.parameters.items():
                    # Replace underscores with hyphens and escape LaTeX special characters
                    param_safe = param.replace('_', '-')
                    value_safe = str(latex(value))  # Ensure value is correctly formatted for LaTeX
                    parameters_i += f"{param_safe}: ${value_safe}$; "

            # Format row with appropriate escaping for LaTeX math mode
            row = (
                f"{f.filterOrder} & "
                f"${latex(f.zCombo)}$ & "
                f"${latex(f.transferFunc)}$ & "
                f"{'Yes' if f.valid else 'No'} & "
                f"${f.fType}$ & "
                f"{parameters_i} "
            )
            latex_table += row + r" \\ \hline" + "\n"
        
        # End table
        latex_table += r"""
        \end{longtable}"""
        
        return latex_table

    def _generate_latex_section(self, sectionName: str):
        latex_content = f"\\section{{{sectionName}}}\\ \n"  # Use fType as the section name
        return latex_content

    def generateLaTeXReport(self, filter_classifications: List[FilterClassification], output_filename="Report", newpage=False, subFolder="NONE"):

        output_filename = self.createSubFolder(output_filename, subFolder)

        with open(output_filename, "w") as latex_file:
            latex_file.write(self.header)
            latex_file.write(f"\\section*{{Experiment: {subFolder.replace('_', ',')}}}\n")

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

            latex_file.write(self.footer)
        print(f"LaTeX report saved to: {output_filename}")
        self.texFiles.append(output_filename)


# UTIL FUNCTIONS
def clear_terminal():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


# def simplify_array(array_of_impedances: List[sympy.Mul]) -> List[sympy.Mul]:
#     array =[]
#     for _impedance in array_of_impedances:
#         array.append(sympy.simplify(_impedance))
#     return array

# def seriesZ(list_of_impedances: List[sympy.Basic]) -> sympy.Mul:
    
#     equivalentZ = list_of_impedances[0]
#     for impedance in list_of_impedances[1:]:
#         equivalentZ += impedance
    
#     return sympy.simplify(equivalentZ)

# def parallelZ(list_of_impedances: List[sympy.Basic]) -> sympy.Mul:
    
#     equivalentG = 1/list_of_impedances[0]
#     for impedance in list_of_impedances[1:]:
#         equivalentG += 1/impedance
    
#     return sympy.simplify(1/equivalentG)
