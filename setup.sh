#!/bin/bash
# setup.sh - Complete environment setup script
# ----------------------------------------------------------
# This script sets up the Python environment.

# Source utility functions
# Ensure utils.sh is in the same directory or provide a correct path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=utils.sh
source "${SCRIPT_DIR}/utils.sh"

# Make script exit on first error
# set -e # Temporarily remove set -e for read prompts and conditional exits

# Set target Python version and environment name
PYTHON_VERSION="3.12.5"
VENV_NAME="venv-${PYTHON_VERSION}"
LOCAL_PYTHON_DIR="local-python" # Relative to PWD where script is run

# Python version parts for URLs and package names
PYTHON_VERSION_MAJOR_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f1,2) # e.g., 3.12
PYTHON_VERSION_URL_PART=$(echo "$PYTHON_VERSION" | tr -d '.')      # e.g., 3125

# Check if running on Windows
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    print_error "This script is for macOS/Linux. Please use setup.bat on Windows."
    exit 1
fi

# Function to set up pyenv (checks if pyenv is usable for target version)
use_pyenv_python() {
    # (This function remains largely the same as before, ensuring it correctly
    #  identifies if pyenv is present and has/can install the target Python version,
    #  and then sets pyenv shell to use it. Returns 0 on success, 1 on failure.)

    print_info "Checking for Python ${PYTHON_VERSION} via pyenv..."
    if ! command_exists pyenv; then
        print_info "pyenv command not found."
        return 1
    fi

    eval "$(pyenv init -)"
    eval "$(pyenv init --path 2>/dev/null || true)"

    if pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
        print_info "Python ${PYTHON_VERSION} found in pyenv."
        # To ensure the command_exists python check works, we'd need to set pyenv shell or similar
        # This function is more about detection and potential setup
        return 0 # pyenv has the version
    else
        print_warning "Python ${PYTHON_VERSION} not found in pyenv."
        read -p "Install Python ${PYTHON_VERSION} using pyenv? (y/N): " install_with_pyenv
        if [[ $install_with_pyenv == "y" || $install_with_pyenv == "Y" ]]; then
            print_info "Installing Python ${PYTHON_VERSION} with pyenv (this may take several minutes)..."
            if pyenv install "${PYTHON_VERSION}"; then
                print_info "Python ${PYTHON_VERSION} installed successfully with pyenv."
                return 0 # pyenv now has the version
            else
                print_error "Failed to install Python ${PYTHON_VERSION} with pyenv."
                return 1
            fi
        else
            print_info "Skipping pyenv installation of Python ${PYTHON_VERSION}."
            return 1
        fi
    fi
}

# Function to use system Python if available at correct version
use_system_python() {
    print_info "Checking for system Python ${PYTHON_VERSION}..."
    if command_exists python3; then
        system_python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_info "Found system python3 version: ${system_python_version}"
        if [[ "$system_python_version" == "$PYTHON_VERSION"* ]]; then # Allow for 3.12.5 and 3.12.5+
            print_info "System python3 matches required version ${PYTHON_VERSION}."
            PYTHON_CMD_FOUND="python3"
            return 0
        fi
    fi
    # Check for version-specific python command e.g. python3.12
    PYTHON_MM_CMD="python${PYTHON_VERSION_MAJOR_MINOR}" # e.g. python3.12
    if command_exists "$PYTHON_MM_CMD"; then
        system_python_version=$($PYTHON_MM_CMD --version 2>&1 | cut -d' ' -f2)
        print_info "Found system $PYTHON_MM_CMD version: ${system_python_version}"
        if [[ "$system_python_version" == "$PYTHON_VERSION"* ]]; then
             print_info "System $PYTHON_MM_CMD matches required version ${PYTHON_VERSION}."
             PYTHON_CMD_FOUND="$PYTHON_MM_CMD"
             return 0
        fi
    fi
    print_warning "No system Python found matching ${PYTHON_VERSION}."
    return 1
}


# Function to create a local Python installation in ./local-python/
# This function now relies on python-build or pyenv to do the actual building.
create_local_python() {
    print_step "Attempting to create local Python ${PYTHON_VERSION} in ./${LOCAL_PYTHON_DIR}/"
    ABSOLUTE_LOCAL_PYTHON_DIR="${PWD}/${LOCAL_PYTHON_DIR}"

    if [ -x "${ABSOLUTE_LOCAL_PYTHON_DIR}/bin/python3" ]; then
        local_python_version=$("${ABSOLUTE_LOCAL_PYTHON_DIR}/bin/python3" --version 2>&1 | cut -d' ' -f2)
        if [[ "$local_python_version" == "$PYTHON_VERSION"* ]]; then
            print_info "Python ${PYTHON_VERSION} already found in local directory: ${ABSOLUTE_LOCAL_PYTHON_DIR}"
            PYTHON_CMD_FOUND="${ABSOLUTE_LOCAL_PYTHON_DIR}/bin/python3"
            return 0
        else
            print_warning "Existing local Python version (${local_python_version}) does not match. Will attempt to reinstall."
            rm -rf "${ABSOLUTE_LOCAL_PYTHON_DIR}" # Clean up old version before reinstall
        fi
    fi

    mkdir -p "${ABSOLUTE_LOCAL_PYTHON_DIR}"

    if command_exists python-build; then
        print_info "Using python-build to install Python ${PYTHON_VERSION} into ${ABSOLUTE_LOCAL_PYTHON_DIR}..."
        if python-build "${PYTHON_VERSION}" "${ABSOLUTE_LOCAL_PYTHON_DIR}"; then
            if [ -x "${ABSOLUTE_LOCAL_PYTHON_DIR}/bin/python3" ]; then
                print_info "Python ${PYTHON_VERSION} installed successfully by python-build."
                PYTHON_CMD_FOUND="${ABSOLUTE_LOCAL_PYTHON_DIR}/bin/python3"
                return 0
            else
                print_error "python-build completed but Python executable not found."
                return 1
            fi
        else
            print_error "python-build failed to install Python ${PYTHON_VERSION}."
            return 1
        fi
    elif command_exists pyenv; then
        print_info "Using pyenv to install Python ${PYTHON_VERSION}, then copying to ${ABSOLUTE_LOCAL_PYTHON_DIR}..."
        eval "$(pyenv init -)" # Ensure pyenv is active for this sub-process
        eval "$(pyenv init --path 2>/dev/null || true)"

        # Check if pyenv already has this version; if not, install it
        if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
            print_info "pyenv does not have Python ${PYTHON_VERSION}. Attempting 'pyenv install ${PYTHON_VERSION}'..."
            if ! pyenv install "${PYTHON_VERSION}"; then
                print_error "pyenv failed to install Python ${PYTHON_VERSION}."
                return 1
            fi
            print_info "pyenv successfully installed Python ${PYTHON_VERSION}."
        else
            print_info "Python ${PYTHON_VERSION} already available in pyenv."
        fi
        
        PYENV_VERSION_PATH="$(pyenv root)/versions/${PYTHON_VERSION}"
        if [ -d "$PYENV_VERSION_PATH" ]; then
            print_info "Copying Python from ${PYENV_VERSION_PATH} to ${ABSOLUTE_LOCAL_PYTHON_DIR}..."
            # Using rsync for better copying if available, else cp
            # Ensure the destination directory is empty or handle appropriately
            if command_exists rsync; then
                rsync -a --delete "${PYENV_VERSION_PATH}/" "${ABSOLUTE_LOCAL_PYTHON_DIR}/"
            else
                # Simple cp; might need to clean dir first if cp doesn't overwrite well
                rm -rf "${ABSOLUTE_LOCAL_PYTHON_DIR:?}"/* # Clear before copy
                cp -RpH "${PYENV_VERSION_PATH}/." "${ABSOLUTE_LOCAL_PYTHON_DIR}/"
            fi

            if [ -x "${ABSOLUTE_LOCAL_PYTHON_DIR}/bin/python3" ]; then
                print_info "Python ${PYTHON_VERSION} copied successfully to local directory."
                PYTHON_CMD_FOUND="${ABSOLUTE_LOCAL_PYTHON_DIR}/bin/python3"
                return 0
            else
                print_error "Copied Python from pyenv, but executable not found in local directory."
                return 1
            fi
        else
            print_error "pyenv version path ${PYENV_VERSION_PATH} not found after (potential) install."
            return 1
        fi
    else
        print_error "Cannot create local Python: 'python-build' or 'pyenv' (for building) is required but not found on your system."
        # Specific error code or message to indicate prerequisites missing for this automated option
        return 2 # Special return code for "prerequisites missing for automated local build"
    fi
}

# New function for detailed manual installation guidance
print_manual_python_installation_guidance() {
    print_info "\nTo use this project, Python ${PYTHON_VERSION} is required."
    print_info "Please install it using one of the following methods and then re-run './setup.sh':"
    print_info "------------------------------------------------------------------------------------------"
    print_info "${BOLD}Option A: Install pyenv (Recommended for managing multiple Python versions)${NC}"
    print_info "  1. Install pyenv:"
    print_info "     - On macOS with Homebrew: 'brew install pyenv'"
    print_info "     - On Linux: Follow instructions at https://github.com/pyenv/pyenv-installer or your package manager."
    print_info "  2. Configure your shell for pyenv (it will provide instructions during/after install)."
    print_info "  3. Install Python ${PYTHON_VERSION} with pyenv: 'pyenv install ${PYTHON_VERSION}'"
    print_info "  4. Re-run this setup script. It should then automatically use this pyenv version."
    print_info "------------------------------------------------------------------------------------------"
    print_info "${BOLD}Option B: Install Python ${PYTHON_VERSION} directly using a system package manager${NC}"
    print_info "  - On macOS with Homebrew: 'brew install python@${PYTHON_VERSION_MAJOR_MINOR}'"
    print_info "    (Ensure Homebrew's Python is in your PATH, e.g., by following Homebrew's post-install caveats)"
    print_info "  - On Debian/Ubuntu (check availability): 'sudo apt update && sudo apt install python${PYTHON_VERSION_MAJOR_MINOR}'"
    print_info "  - On Fedora (check availability): 'sudo dnf install python${PYTHON_VERSION_MAJOR_MINOR}'"
    print_info "  After installation, re-run this setup script. It may detect it as 'system Python'."
    print_info "------------------------------------------------------------------------------------------"
    print_info "${BOLD}Option C: Download and install Python ${PYTHON_VERSION} from python.org${NC}"
    print_info "  1. Visit https://www.python.org/downloads/release/python-${PYTHON_VERSION_URL_PART}/"
    print_info "  2. Download the appropriate installer for your OS and run it."
    print_info "  3. Ensure the installed Python's 'bin' directory is in your system PATH."
    print_info "  4. Re-run this setup script."
    print_info "------------------------------------------------------------------------------------------"
}


# Main function to set up the Python environment
setup_python_environment() {
    PYTHON_CMD_FOUND="" # Will be set by one of the python finding/installing functions

    # 1. Try to use Python from an existing local project install
    if [ -x "${PWD}/${LOCAL_PYTHON_DIR}/bin/python3" ]; then
        local_python_version=$("${PWD}/${LOCAL_PYTHON_DIR}/bin/python3" --version 2>&1 | cut -d' ' -f2)
        if [[ "$local_python_version" == "$PYTHON_VERSION"* ]]; then
            print_info "Found existing local Python ${PYTHON_VERSION} in ./${LOCAL_PYTHON_DIR}/"
            PYTHON_CMD_FOUND="${PWD}/${LOCAL_PYTHON_DIR}/bin/python3"
        fi
    fi

    # 2. If not found locally, try pyenv (if available and configured for the version)
    if [ -z "$PYTHON_CMD_FOUND" ] && command_exists pyenv; then
        eval "$(pyenv init -)"
        eval "$(pyenv init --path 2>/dev/null || true)"
        if pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
            # If pyenv has it, ensure it's the one Python will use.
            # We might need to set `pyenv shell ${PYTHON_VERSION}` or rely on .python-version
            # For now, let's assume if pyenv has it, 'python' command (after pyenv init) will be it.
            # This needs to be robust.
            # A simple way: if pyenv has it, use `pyenv which python` that corresponds to the version.
            PYENV_PYTHON_PATH=$(pyenv which python || true) # Get path to current pyenv python
            if [ -n "$PYENV_PYTHON_PATH" ]; then
                 pyenv_python_ver=$($PYENV_PYTHON_PATH --version 2>&1 | cut -d' ' -f2)
                 if [[ "$pyenv_python_ver" == "$PYTHON_VERSION"* ]]; then
                    print_info "Using Python ${PYTHON_VERSION} from pyenv."
                    PYTHON_CMD_FOUND="$PYENV_PYTHON_PATH"
                 fi
            fi
        fi
    fi
    
    # 3. If not found yet, try system python
    if [ -z "$PYTHON_CMD_FOUND" ]; then
        if use_system_python; then # This sets PYTHON_CMD_FOUND internally on success
            print_info "Using system Python ${PYTHON_VERSION} found at ${PYTHON_CMD_FOUND}."
        fi
    fi

    # 4. If still no suitable Python found, interact with user
    if [ -z "$PYTHON_CMD_FOUND" ]; then
        print_warning "Python ${PYTHON_VERSION} was not found through existing local installations, pyenv, or system paths."
        echo ""
        print_info "How would you like to proceed to get Python ${PYTHON_VERSION}?"
        echo "1. Attempt to install Python ${PYTHON_VERSION} into a dedicated local project directory (./${LOCAL_PYTHON_DIR}/)."
        echo "   (This automated option requires 'pyenv' or 'python-build' to be installed on your system)."
        echo "2. Exit now. You can then install Python ${PYTHON_VERSION} manually (see guidance below) and re-run this script."
        
        local choice
        read -p "Choose an option (1/2): " choice

        if [ "$choice" == "1" ]; then
            create_local_python # This will attempt to use python-build or pyenv as a builder
            local_build_status=$?
            if [ $local_build_status -eq 0 ]; then # PYTHON_CMD_FOUND is set by create_local_python on success
                print_info "Successfully set up local Python in ./${LOCAL_PYTHON_DIR}/."
            elif [ $local_build_status -eq 2 ]; then # Prerequisite missing for automated local build
                print_error "Automated local installation failed because 'pyenv' or 'python-build' was not found."
                print_manual_python_installation_guidance
                exit 1
            else # Other failure during local build
                print_error "Failed to create a local Python installation."
                print_manual_python_installation_guidance
                exit 1
            fi
        else
            print_manual_python_installation_guidance
            exit 0 # User chose to exit and install manually
        fi
    fi

    # Verify Python command is accessible and correct version
    if ! command_exists "$PYTHON_CMD_FOUND"; then
        print_error "Python command ($PYTHON_CMD_FOUND) is not accessible or not set after attempting setup."
        print_manual_python_installation_guidance
        exit 1
    fi
    
    current_python_version=$("$PYTHON_CMD_FOUND" --version 2>&1 | cut -d' ' -f2)
    if [[ "$current_python_version" != "$PYTHON_VERSION"* ]]; then
        print_error "Python version mismatch: active Python is $current_python_version ($PYTHON_CMD_FOUND), but expected $PYTHON_VERSION."
        print_manual_python_installation_guidance
        exit 1
    fi
    
    print_info "Using Python: $($PYTHON_CMD_FOUND --version) (from $PYTHON_CMD_FOUND)"

    # --- Virtual Environment Setup (using $PYTHON_CMD_FOUND) ---
    print_step "Creating virtual environment ./${VENV_NAME}/"
    if [ -d "$VENV_NAME" ]; then
        read -p "Virtual environment './${VENV_NAME}/' already exists. Recreate? (y/N): " recreate
        if [[ $recreate == "y" || $recreate == "Y" ]]; then
            print_info "Removing existing virtual environment."
            rm -rf "$VENV_NAME"
        else
            print_info "Using existing virtual environment."
        fi
    fi
    
    if [ ! -d "$VENV_NAME" ]; then
        print_info "Creating new virtual environment with $($PYTHON_CMD_FOUND --version)..."
        "$PYTHON_CMD_FOUND" -m venv "$VENV_NAME"
        if [ $? -ne 0 ]; then
            print_error "Failed to create virtual environment. Check your Python installation."
            exit 1
        fi
    fi
    
    # Activate virtual environment for this script's operations
    print_step "Activating virtual environment for setup"
    # shellcheck source=./venv-3.12.5/bin/activate
    source "${PWD}/${VENV_NAME}/bin/activate"
    
    # Verify Python version in virtual environment
    VENV_PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
    print_info "Virtual environment Python version: ${VENV_PYTHON_VERSION}"
    if [[ "$VENV_PYTHON_VERSION" != "$PYTHON_VERSION"* ]]; then
        print_warning "Virtual environment Python version mismatch: got $VENV_PYTHON_VERSION, expected $PYTHON_VERSION."
        print_warning "This might indicate an issue with venv creation or activation. Proceeding cautiously."
    fi
    
    # Upgrade pip
    print_step "Upgrading pip"
    python -m pip install --upgrade pip
    
    # Install packages
    if [ -f "requirements.txt" ]; then
        print_step "Installing packages from requirements.txt"
        python -m pip install -r requirements.txt
    else
        print_warning "requirements.txt not found. Skipping package installation from file."
        print_step "Installing essential packages (jupyterlab, ipykernel, python-dotenv, google-cloud-storage, torch)"
        python -m pip install jupyterlab ipykernel python-dotenv google-cloud-storage
    fi
    
    # PyTorch installation (could be conditional or already in requirements.txt)
    # This logic can be kept or simplified if PyTorch is always in requirements.txt
    print_step "Handling PyTorch installation (if not already installed by requirements.txt)"
    if ! python -m pip show torch > /dev/null 2>&1; then
        print_info "PyTorch not detected from requirements.txt, attempting specific install..."
        if command_exists nvidia-smi; then
            print_info "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
            python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 # Adjust CUDA version as needed
        elif [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
            print_info "Apple Silicon (arm64) detected. Installing PyTorch with MPS support..."
            python -m pip install torch torchvision torchaudio # Default should get MPS version
        else
            print_info "No specialized GPU detected or PyTorch already installed. Installing/checking PyTorch CPU version..."
            python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    else
        print_info "PyTorch seems to be already installed (likely from requirements.txt)."
    fi

    # Setup Jupyter kernel
    print_step "Setting up Jupyter kernel"
    PROJECT_NAME=$(basename "$PWD")
    KERNEL_NAME="${PROJECT_NAME}_kernel_py${PYTHON_VERSION_MAJOR_MINOR}"
    DISPLAY_NAME="${PROJECT_NAME} (Python ${PYTHON_VERSION})"
    
    python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "${DISPLAY_NAME}"
    print_info "Jupyter kernel installed as '${DISPLAY_NAME}'"
    
    # Create activation script
    print_step "Creating activation script (activate.sh)"
    cat > activate.sh << EOL
#!/bin/bash
# Activate the virtual environment
# shellcheck source=./${VENV_NAME}/bin/activate
source "${PWD}/${VENV_NAME}/bin/activate"

# Set environment variables from .env if it exists
if [ -f .env ]; then
    set -a
    # shellcheck source=.env
    source .env
    set +a
fi

echo "Virtual environment activated with Python ${PYTHON_VERSION}. Run 'deactivate' to exit."
EOL
    
    chmod +x activate.sh
    print_info "Created activate.sh - Run 'source activate.sh' to activate the environment."

    # Deactivate at the end of setup script's use of venv
    if declare -f deactivate > /dev/null && [[ "$(type -t deactivate)" == "function" ]]; then
      deactivate
    fi
}


# Main function
main() {
    print_step "Starting Research Environment Setup"
    
    setup_python_environment # This function will exit if Python setup isn't resolved
    
    print_step "Environment Setup Complete!"
    print_info "Your Python environment and Jupyter kernel are ready."
    print_info "To activate the environment in a new terminal: source activate.sh"
    print_info "Then, to start Jupyter Lab: jupyter-lab"
    print_info "\nNext, please run './download_data.sh' to download the research data if you haven't already."
}

# Run the main function
main