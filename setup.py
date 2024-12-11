from setuptools import setup, find_packages

setup(name           = "MacAnalog_Symbolix",
    description      = "A python app to perform symbolic analysis on analog circuits", 
    version          = "0.1",
    author           = "Danial Noori Zadeh",
    author_email     = "dnoorizadeh@gmail.com",
    url              = "https://github.com/NooriDan/MacAnalog-Symbolix",
    long_description = open("README.md").read(),
    packages         = find_packages(),
    python_requires  = ">=3.8",  # Minimum Python version required
    install_requires = [
        "numpy>=1.21.0",    # Dependencies
        "pandas>=1.3.0",
        "sympy>=1.13.3"
    ],    
    classifiers     = [
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",  # GPL License
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    )