#!/bin/bash

# Load .env from the current working directory (where the script is run from)
if [ -f "$(pwd)/.env" ]; then
    source "$(pwd)/.env"
else
    echo "Error: .env file not found in the current directory!"
    exit 1
fi

# Convert comma-separated SUBDIRECTORIES into space-separated array
IFS=',' read -ra SUBDIRECTORIES_ARRAY <<< "$SUBDIRECTORIES"

# If repo exists, update it
if [ -d "$REPO_NAME" ]; then
    cd "$REPO_NAME"
    git pull origin main  # Pull latest changes
    git sparse-checkout set "${SUBDIRECTORIES_ARRAY[@]}"  # Update sparse checkout
    git checkout  # Apply changes
    echo "Updated subdirectories: ${SUBDIRECTORIES_ARRAY[*]}"
else
    # Clone repository without checking out files
    git clone --no-checkout "$REPO_URL"
    cd "$REPO_NAME"

    # Enable sparse checkout
    git sparse-checkout init --cone
    git sparse-checkout set "${SUBDIRECTORIES_ARRAY[@]}"

    # Checkout the specified subdirectories
    git checkout
    echo "Cloned subdirectories: ${SUBDIRECTORIES_ARRAY[*]}"
fi
