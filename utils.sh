#!/bin/bash
# utils.sh - Shared utility functions and variables

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
RED="\033[0;31m"
NC="\033[0m" # No Color

# Function to print a formatted step message
print_step() {
    echo -e "\n${BOLD}${GREEN}=== $1 ===${NC}\n"
}

# Function to print a formatted informational message
print_info() {
    echo -e "${BLUE}$1${NC}"
}

# Function to print a formatted warning message
print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

# Function to print a formatted error message
print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Ensure the script is sourced, not executed directly, if needed for variable export
# (though for functions and simple variables, direct execution before sourcing works)
# If this script were only to define functions, it wouldn't need a shebang
# or execute permissions, but it's good practice to have them.

