import sympy
from   sympy import latex
from   typing import Dict, List, TYPE_CHECKING
import subprocess
import os, datetime
import platform
import psutil
import pickle
# Custom Imports
if TYPE_CHECKING:
    # Imports for type checking only
    from .domains import Filter_Classification
    from .solver import Circuit_Solver



# UTIL CLASSES
class FilterDatabase:
    def __init__(self, name: str, filterOrder: int, z_arr_size: int = 6):
        self.name: str   = name
        self.filterOrder = filterOrder
        self.z_arr_size  = z_arr_size
        self.circuitSolver: Circuit_Solver
        self.results: Dict[str, List['Filter_Classification']]
        # Filter types
        self.bandpass: List['Filter_Classification'] = [] 
        self.lowpass:  List['Filter_Classification'] = []
        self.highpass: List['Filter_Classification'] = []
        self.bandstop: List['Filter_Classification'] = []
        self.ge_ae:    List['Filter_Classification'] = [] 

        self.invalid_numer: List['Filter_Classification'] = [] 
        self.invalid_wz:   List['Filter_Classification'] = []
        self.invalid_order: List['Filter_Classification'] = []
        self.error:       List['Filter_Classification'] = []

        self.mapList: Dict[str, List['Filter_Classification']] = {
                "BP" : self.bandpass,
                "HP" : self.highpass,
                "LP" : self.lowpass,
                "BS" : self.bandstop,
                "GE" : self.ge_ae,
                "INVALID-NUMER" : self.invalid_numer,
                "INVALID-WZ"    : self.invalid_wz,
                "INVALID-ORDER" : self.invalid_order,
                "PolynomialError" : self.error
        }

        self.unrecognized: List['Filter_Classification'] = []

        self.fileSave = FileSave()

    def add(self, classifications: List['Filter_Classification']):
        for classification in classifications:
            if (classification.fType in self.mapList.keys()) and (len(classification.zCombo) == self.z_arr_size):
                self.mapList[classification.fType].append(classification)
            else:
                self.unrecognized.append(classification)

    def filterByZ(self, fType, z_arr: List[sympy.Symbol]):
        if (len(z_arr != self.z_arr_size)):
            print("FilterDatabase: INVALID Z SIZE")
            raise ValueError
        
        # for i, z in enumerate(z_arr, 0):  
        #     z_arr[i] = (z == sympy.oo)

        if self.mapList.get(fType) == None:
            print("FilterDatabase: filter type not valid")
            raise KeyError

        # _z1, _z2, _z3, _z4, _z5, _zL = z_arr

        results: List['Filter_Classification'] = []
        for classification in self.mapList[fType]:
            for z in z_arr: 
                if z == sympy.oo:
                    results.append(classification)

        return results

    def printFilter(self, fType):
        if fType in self.mapList.keys():

            self.fileSave.generateLaTeXReport(self.mapList[fType], f"{self.name}_{fType}", subFolder="Database")

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
        filterClusters: Dict[str, List['Filter_Classification']],
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

    def generateSummaryTable(self, clusterByType: Dict[str, List['Filter_Classification']], output_filename, subFolder = "None"):

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

    def _generate_latex_table(self, filter_list: List['Filter_Classification'], caption):
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

    def generateLaTeXReport(self, filter_classifications: List['Filter_Classification'], output_filename="Report", newpage=False, subFolder="NONE"):

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
                    
                    # # Ensure parameters are written in math mode
                    # latex_file.write(f"\\textbf{{Q:}} ${latex(classification.parameters['Q'])}$ \\\\ \n")
                    # latex_file.write(f"\\textbf{{$\\omega_0$:}} ${latex(classification.parameters['wo'])}$ \\\\ \n")
                    # latex_file.write(f"\\textbf{{Bandwidth:}} ${latex(classification.parameters['bandwidth'])}$ \\\\ \n")

                    # Print parameters dynamically
                    for param_key, param_value in classification.parameters.items():
                        latex_file.write(f"\\textbf{{{param_key}:}} ${latex(param_value)}$ \\\\ \n")


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

def get_system_specs():
    # Basic system details
    specs = {
        "OS": platform.system(),
        "Node Name": platform.node(),
        "Release": platform.release(),
        "Version": platform.version(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "CPU Cores (Physical)": psutil.cpu_count(logical=False),
        "Logical CPUs": psutil.cpu_count(logical=True),
        "Total RAM (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "Python Version": platform.python_version()
    }
    
    # Extended details based on OS
    if specs["OS"] == "Darwin":  # macOS
        specs.update(get_macos_specs())
    elif specs["OS"] == "Linux":  # Linux
        specs.update(get_linux_specs())
    elif specs["OS"] == "Windows":  # Windows
        specs.update(get_windows_specs())

    return specs

def get_macos_specs():
    try:
        system_profiler = subprocess.check_output(["system_profiler", "SPHardwareDataType"], text=True)
        details = {}
        for line in system_profiler.splitlines():
            if "Chip:" in line or "Memory:" in line or "Model Name:" in line:
                key, value = line.split(":", 1)
                details[key.strip()] = value.strip()
        return details
    except Exception as e:
        return {"Mac Specific Info": f"Error: {e}"}

def get_linux_specs():
    try:
        cpu_info = subprocess.check_output("lscpu", shell=True, text=True)
        memory_info = subprocess.check_output("free -h", shell=True, text=True)
        return {
            "CPU Info": cpu_info,
            "Memory Info": memory_info
        }
    except Exception as e:
        return {"Linux Specific Info": f"Error: {e}"}

def get_windows_specs():
    try:
        cpu_info = subprocess.check_output("wmic cpu get caption,deviceid,maxclockspeed,numberofcores", shell=True, text=True)
        memory_info = subprocess.check_output("wmic memorychip get capacity", shell=True, text=True)
        return {
            "CPU Info": cpu_info,
            "Memory Info": memory_info
        }
    except Exception as e:
        return {"Windows Specific Info": f"Error: {e}"}

def print_specs():
    specs = get_system_specs()
    print(f"Run Datetime: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    print("\n====> Computer Specifications: <====")
    for key, value in specs.items():
        print(f"{key}: {value}")
    print("-------------------------------------\n\n")    


if __name__ == "__main__":
    clear_terminal()
    print("You are running the Utils.py file!")
    print_specs()