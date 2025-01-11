#!/usr/bin/env bash

# Usage:
#   ./script.sh <main_directory> [additional_path...]
# where each 'additional_path' can be either a file or a directory.

# We need at least one argument (the main directory).
if [ $# -lt 1 ]; then
    echo "Usage: $0 <main_directory> [additional_path...]"
    exit 1
fi

MAIN_DIRECTORY="$1"
shift  # Shift so that $@ now holds the additional paths (if any)

OUTPUT="output.txt"

# Clear or create the output file
> "$OUTPUT"

############################################
# 1. Process all .py files in MAIN_DIRECTORY
############################################
find "$MAIN_DIRECTORY" -type f -name '*.py' | while IFS= read -r FILE; do
    echo "$FILE" >> "$OUTPUT"
    echo '```' >> "$OUTPUT"
    cat "$FILE" >> "$OUTPUT"
    echo '```' >> "$OUTPUT"
    echo "" >> "$OUTPUT"
done

##########################################
# 2. Process each additional path in $@
##########################################
for ITEM in "$@"; do
    if [ -d "$ITEM" ]; then
        # ITEM is a directory; process *.py files in it
        find "$ITEM" -type f -name '*.py' | while IFS= read -r FILE; do
            echo "$FILE" >> "$OUTPUT"
            echo '```' >> "$OUTPUT"
            cat "$FILE" >> "$OUTPUT"
            echo '```' >> "$OUTPUT"
            echo "" >> "$OUTPUT"
        done
    elif [ -f "$ITEM" ]; then
        # ITEM is a regular file; just output its content
        echo "$ITEM" >> "$OUTPUT"
        echo '```' >> "$OUTPUT"
        cat "$ITEM" >> "$OUTPUT"
        echo '```' >> "$OUTPUT"
        echo "" >> "$OUTPUT"
    else
        # ITEM is neither a directory nor a file
        echo "Warning: '$ITEM' does not exist or is not valid." >&2
    fi
done

echo "All requested files have been written to '$OUTPUT'."
