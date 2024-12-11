#!/bin/bash
# Interactive script to move files of a specified extension
# Navigate directories to select source and destination

FILE_EXT="pdf"  # File extension to move (change as needed)

# Function to display and navigate directories
select_directory() {
    local current_dir="$1"  # Starting directory
    while true; do
        echo
        echo "========================================="
        echo "        ~ Select a Directory              "
        echo "========================================="
        echo "Current Directory: $current_dir"
        echo

        # List directories and provide navigation options
        local entries=()
        entries+=(".. (Go Up)")  # Option to go up one level
        index=1

        # Add directories to the list
        for entry in "$current_dir"*/; do
            echo "[$index] $(basename "$entry")/"
            entries+=("$entry")
            ((index++))
        done

        # Exit if there are no directories to navigate
        if [[ ${#entries[@]} -eq 1 ]]; then
            echo "No subdirectories found. Returning to the previous menu."
            return 1
        fi

        # Prompt user to choose a directory
        echo
        echo "Enter the line number of the directory, or '0' to confirm this directory:"
        read -r choice
        echo

        # Handle user input
        if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 0 || choice >= ${#entries[@]} )); then
            echo "Invalid choice. Please select a valid option."
        elif [[ "$choice" -eq 0 ]]; then
            echo "Directory selected: $current_dir"
            echo
            echo "========================================="
            return 0  # Confirm the current directory
        elif [[ "${entries[$choice]}" == ".. (Go Up)" ]]; then
            # Navigate up one level
            current_dir="$(dirname "$current_dir")/"
        else
            # Navigate into the selected directory
            current_dir="${entries[$choice]}"
        fi
    done
}

# Step 1: Select the source directory
echo "Step 1: Select the source directory to search for .$FILE_EXT files."
select_directory "./"
SOURCE_DIR="$current_dir"  # Save the selected source directory
echo "Selected source directory: $SOURCE_DIR"  # Debug print

# Convert to absolute path if it's a relative path
SOURCE_DIR=$(realpath "$SOURCE_DIR")
echo "Absolute source directory: $SOURCE_DIR"  # Debug print

# Step 2: Select the destination directory
echo "Step 2: Select the destination directory to move .$FILE_EXT files into."
select_directory "./"
DEST_DIR="$current_dir"  # Save the selected destination directory
echo "Selected destination directory: $DEST_DIR"  # Debug print

# Convert to absolute path if it's a relative path
DEST_DIR=$(realpath "$DEST_DIR")
echo "Absolute destination directory: $DEST_DIR"  # Debug print

# Step 3: Confirm operation
echo
echo "Source directory: $SOURCE_DIR"
echo "Destination directory: $DEST_DIR"
echo "All .$FILE_EXT files in $SOURCE_DIR and its subdirectories will be moved to $DEST_DIR."
echo "Proceed? (y/n)"
read -r confirm
echo

if [[ $confirm != "y" ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Step 4: Move files and count them
echo "Moving .$FILE_EXT files from $SOURCE_DIR to $DEST_DIR..."

file_count=0  # Initialize counter for moved files
# Use find to recursively search for files and move them
find "$SOURCE_DIR" -type f -iname "*.$FILE_EXT" | while read -r file; do
    mv "$file" "$DEST_DIR"
    ((file_count++))
done

echo
echo "========================================="
echo "         File Move Completed             "
echo "========================================="
echo "$file_count .$FILE_EXT files from $SOURCE_DIR have been moved to $DEST_DIR."
