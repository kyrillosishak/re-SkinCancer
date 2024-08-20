#!/bin/zsh

# Find all markdown files in the markdowns directory and store them in an array
SOURCES=($(find markdowns -type f -name "*.md"))

# Create the notebooks directory if it doesn't exist
mkdir -p notebooks

# Loop over the array of source files
for SOURCE in "${SOURCES[@]}"; do
  # Derive the target notebook filename
  TARGET="notebooks/$(basename "${SOURCE%.md}.ipynb")"

  # Convert markdown to ipynb using pandoc
  if pandoc --resource-path=assets/ --embed-resources --standalone --from markdown-yaml_metadata_block --wrap=none "$SOURCE" -o "$TARGET"; then
    echo "Converted $SOURCE to $TARGET"
  else
    echo "Failed to convert $SOURCE"
  fi
done
