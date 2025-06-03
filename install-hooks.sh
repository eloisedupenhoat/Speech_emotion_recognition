#!/bin/bash
# install-hooks.sh

echo "Installing Git hooks..."

# Source directory for hooks
source_dir="hooks"
# Destination directory for hooks
dest_dir=".git/hooks"

# Check if the source directory exists
if [ ! -d "$source_dir" ]; then
    echo "Source hooks directory does not exist!"
    exit 1
fi

# Check if the destination directory exists
if [ ! -d "$dest_dir" ]; then
    echo "Destination hooks directory does not exist!"
    exit 1
fi

# Copy each hook from the source directory to the destination directory
for hook in "$source_dir"/*; do
    hook_name=$(basename "$hook")
    dest_hook="$dest_dir/$hook_name"

    # Check if the hook file already exists in the destination directory
    if [ -f "$dest_hook" ]; then
        echo "Hook '$hook_name' already exists. Skipping..."
    else
        cp "$hook" "$dest_hook"
        chmod +x "$dest_hook"
        echo "Installed new hook: $hook_name"
    fi
done

echo "Git hooks installation process completed!"
