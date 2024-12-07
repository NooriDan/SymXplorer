#!/bin/bash
# Script to delete files with specified extensions and show only the last two folders of the path

# Define the file extensions to delete (space-separated)
EXTENSIONS=("fdb_latexmk" "fls" "synctex.gz" "log" "aux" "toc")

# Get the last two folders of the current path
SHORT_PWD=$(pwd | awk -F/ '{print $(NF-1)"/"$NF}')

# Welcome message
echo
echo "========================================="
echo "          File Cleanup Utility           "
echo "========================================="
echo
echo "Current directory: $SHORT_PWD"
echo
echo "File extensions to delete: ${EXTENSIONS[*]}"
echo
echo "Proceed to delete all these files? (y/n)"
echo "-----------------------------------------"
read -r response
echo

# Process user response
if [[ $response == "y" ]]; then
    echo "Starting cleanup..."
    echo
    for ext in "${EXTENSIONS[@]}"; do
        echo "Deleting files with extension: .$ext"
        find . -type f -name "*.$ext" -delete
    done
    echo
    echo "========================================="
    echo "           Cleanup Completed             "
    echo "========================================="
    echo
else
    echo "-----------------------------------------"
    echo "           Operation Cancelled           "
    echo "-----------------------------------------"
    echo
fi
