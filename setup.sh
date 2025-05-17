#!/bin/bash
# setup.sh - Complete environment setup script
# ----------------------------------------------------------
# This script sets up the Python environment.

# Source utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=utils.sh
source "${SCRIPT_DIR}/utils.sh"

# Set target Python version and environment name
PYTHON_VERSION="3.12.5"
VENV_NAME="venv-${PYTHON_VERSION}"
LOCAL_PYTHON_DIR="local-python" # Relative to PWD where script is run

# Python version parts for URLs and package names
PYTHON_VERSION_MAJOR_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f1,2) # e.g., 3.12
PYTHON_VERSION_URL_PART=$(echo "$PYTHON_VERSION" | tr -d '.')      # e.g., 3125 for python.org URL
PYTHON_DOWNLOAD_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
PYTHON_SOURCE_DIR_NAME="Python-${PYTHON_VERSION}" # e.g. Python-3.12.5

# Global variable to store the found/installed Python command
PYTHON_CMD_FOUND=""

# --- Python Detection & Installation Functions ---

check_existing_local_python() {
    ABSOLUTE_LOCAL_PYTHON_DIR="${PWD}/${LOCAL_PYTHON_DIR}"
    if [ -x "${ABSOLUTE_LOCAL_PYTHON_DIR}/bin/python3" ]; then
        local local_py_ver
        local_py_ver=$("${ABSOLUTE_LOCAL_PYTHON_DIR}/bin/python3" --version 2>&1 | cut -d' ' -f2)
        if [[ "$local_py_ver" == "$PYTHON_VERSION"* ]]; then
            print_info "Found compatible Python ${PYTHON_VERSION} in local directory: ${ABSOLUTE_LOCAL_PYTHON_DIR}"
            PYTHON_CMD_FOUND="${ABSOLUTE_LOCAL_PYTHON_DIR}/bin/python3"
            return 0
        fi
    fi
    return 1
}

check_pyenv_python() {
    if ! command_exists pyenv; then
        print_info "pyenv command not found."
        return 1
    fi
    print_info "Checking for Python ${PYTHON_VERSION} via pyenv..."
    eval "$(pyenv init -)"
    eval "$(pyenv init --path 2>/dev/null || true)"

    if pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
        # If pyenv has it, we'll use pyenv's shim/path for the python command
        # This assumes user has pyenv configured to make 'python' point to the pyenv version
        # or we use `pyenv exec python` or `$(pyenv root)/versions/$PYTHON_VERSION/bin/python`
        PYENV_PYTHON_EXECUTABLE="$(pyenv root)/versions/${PYTHON_VERSION}/bin/python3"
        if [ -x "$PYENV_PYTHON_EXECUTABLE" ]; then
            pyenv_ver=$($PYENV_PYTHON_EXECUTABLE --version 2>&1 | cut -d' ' -f2)
            if [[ "$pyenv_ver" == "$PYTHON_VERSION"* ]]; then
                print_info "Using Python ${PYTHON_VERSION} from pyenv installation: ${PYENV_PYTHON_EXECUTABLE}"
                PYTHON_CMD_FOUND="$PYENV_PYTHON_EXECUTABLE"
                return 0
            fi
        fi
    fi
    print_info "Python ${PYTHON_VERSION} not actively configured or found via pyenv global/shell."
    return 1
}

check_system_python() {
    print_info "Checking for system Python ${PYTHON_VERSION}..."
    local try_cmds=("python3" "python${PYTHON_VERSION_MAJOR_MINOR}")
    for cmd_to_try in "${try_cmds[@]}"; do
        if command_exists "$cmd_to_try"; then
            local system_py_ver
            system_py_ver=$("$cmd_to_try" --version 2>&1 | cut -d' ' -f2)
            print_info "Found system '$cmd_to_try' version: ${system_py_ver}"
            if [[ "$system_py_ver" == "$PYTHON_VERSION"* ]]; then
                print_info "System '$cmd_to_try' matches required version ${PYTHON_VERSION}."
                PYTHON_CMD_FOUND="$cmd_to_try"
                return 0
            fi
        fi
    done
    print_warning "No system Python found matching ${PYTHON_VERSION}."
    return 1
}

# Function to build Python from source (NEW and COMPLEX)
build_python_from_source() {
    print_step "Attempting to download and build Python ${PYTHON_VERSION} from python.org"
    ABSOLUTE_LOCAL_PYTHON_DIR="${PWD}/${LOCAL_PYTHON_DIR}"

    print_info "This requires build tools (make, C compiler) and development libraries (OpenSSL, zlib, readline, etc.)."
    print_info "Checking for some essential tools..."
    if ! command_exists curl && ! command_exists wget; then
        print_error "curl or wget is required to download Python source. Please install and re-run."
        return 1
    fi
    if ! command_exists make; then
        print_error "'make' is required to build Python. Please install build-essential/make and re-run."
        return 1
    fi
    if ! command_exists gcc && ! command_exists clang; then
        print_error "A C compiler (gcc or clang) is required. Please install and re-run."
        return 1
    fi
    print_info "Basic build tools check passed. Further dependencies (libraries) will be checked by Python's configure script."

    # Clean up previous attempts if any
    rm -rf "${PYTHON_SOURCE_DIR_NAME}" "${PYTHON_SOURCE_DIR_NAME}.tgz" "${ABSOLUTE_LOCAL_PYTHON_DIR}"
    mkdir -p "${ABSOLUTE_LOCAL_PYTHON_DIR}"

    print_info "Downloading Python source from ${PYTHON_DOWNLOAD_URL}..."
    if command_exists curl; then
        curl -L -o "${PYTHON_SOURCE_DIR_NAME}.tgz" "${PYTHON_DOWNLOAD_URL}"
    else
        wget -O "${PYTHON_SOURCE_DIR_NAME}.tgz" "${PYTHON_DOWNLOAD_URL}"
    fi

    if [ ! -f "${PYTHON_SOURCE_DIR_NAME}.tgz" ]; then
        print_error "Failed to download Python source."
        return 1
    fi
    print_info "Download complete. Extracting..."
    tar -xzf "${PYTHON_SOURCE_DIR_NAME}.tgz"
    if [ ! -d "${PYTHON_SOURCE_DIR_NAME}" ]; then
        print_error "Failed to extract Python source."
        rm -f "${PYTHON_SOURCE_DIR_NAME}.tgz"
        return 1
    fi

    cd "${PYTHON_SOURCE_DIR_NAME}" || return 1

    print_info "Configuring Python build (prefix: ${ABSOLUTE_LOCAL_PYTHON_DIR})..."
    # Add --with-openssl if custom OpenSSL path needed, common on macOS
    # Add other --with-xxx flags if specific library locations are known/needed
    # For macOS, explicitly pointing to Homebrew's OpenSSL might be necessary if system one is old
    # e.g. LDFLAGS="-L/opt/homebrew/opt/openssl@3/lib" CPPFLAGS="-I/opt/homebrew/opt/openssl@3/include"
    # This gets very OS-specific quickly.
    ./configure --prefix="${ABSOLUTE_LOCAL_PYTHON_DIR}" --enable-optimizations --with-ensurepip=install
    if [ $? -ne 0 ]; then
        print_error "Python configure failed. Check for missing development libraries (e.g., libssl-dev, zlib1g-dev, libreadline-dev, libffi-dev, sqlite3-dev, etc.)."
        cd ..
        rm -rf "${PYTHON_SOURCE_DIR_NAME}" "${PYTHON_SOURCE_DIR_NAME}.tgz"
        return 1
    fi

    print_info "Building Python (make -j)... This may take several minutes."
    local core_count
    core_count=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
    make -j"${core_count}"
    if [ $? -ne 0 ]; then
        print_error "Python 'make' failed."
        cd ..
        rm -rf "${PYTHON_SOURCE_DIR_NAME}" "${PYTHON_SOURCE_DIR_NAME}.tgz"
        return 1
    fi

    print_info "Installing Python (make install)..."
    make install
    if [ $? -ne 0 ]; then
        print_error "Python 'make install' failed."
        cd ..
        rm -rf "${PYTHON_SOURCE_DIR_NAME}" "${PYTHON_SOURCE_DIR_NAME}.tgz"
        return 1
    fi

    cd ..
    print_info "Cleaning up downloaded source..."
    rm -rf "${PYTHON_SOURCE_DIR_NAME}" "${PYTHON_SOURCE_DIR_NAME}.tgz"

    if [ -x "${ABSOLUTE_LOCAL_PYTHON_DIR}/bin/python3" ]; then
        print_info "Python ${PYTHON_VERSION} successfully built and installed to ./${LOCAL_PYTHON_DIR}/"
        PYTHON_CMD_FOUND="${ABSOLUTE_LOCAL_PYTHON_DIR}/bin/python3"
        return 0
    else
        print_error "Build successful, but Python executable not found at expected location."
        return 1
    fi
}


# Function to attempt automated local Python installation
create_local_python_automatically() {
    print_step "Attempting automated local Python ${PYTHON_VERSION} installation"
    ABSOLUTE_LOCAL_PYTHON_DIR="${PWD}/${LOCAL_PYTHON_DIR}"

    # Try 1: python-build (if available)
    if command_exists python-build; then
        print_info "Found 'python-build'. Attempting to install Python ${PYTHON_VERSION} into ${ABSOLUTE_LOCAL_PYTHON_DIR}..."
        rm -rf "${ABSOLUTE_LOCAL_PYTHON_DIR}" # Clean slate
        mkdir -p "${ABSOLUTE_LOCAL_PYTHON_DIR}"
        if python-build "${PYTHON_VERSION}" "${ABSOLUTE_LOCAL_PYTHON_DIR}"; then
            if [ -x "${ABSOLUTE_LOCAL_PYTHON_DIR}/bin/python3" ]; then
                print_info "Python ${PYTHON_VERSION} installed successfully by python-build."
                PYTHON_CMD_FOUND="${ABSOLUTE_LOCAL_PYTHON_DIR}/bin/python3"
                return 0
            fi
        fi
        print_warning "python-build failed or did not produce an executable. Trying next method."
    else
        print_info "'python-build' not found."
    fi

    # Try 2: pyenv (if available, use it to install then copy)
    if command_exists pyenv; then
        print_info "Found 'pyenv'. Attempting to install Python ${PYTHON_VERSION} via pyenv and copy locally..."
        eval "$(pyenv init -)"
        eval "$(pyenv init --path 2>/dev/null || true)"
        if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
            print_info "pyenv installing Python ${PYTHON_VERSION} (this may take time)..."
            if ! pyenv install "${PYTHON_VERSION}"; then
                print_warning "pyenv install ${PYTHON_VERSION} failed. Trying next method."
                return 1 # pyenv install itself failed
            fi
        fi
        PYENV_VERSION_PATH="$(pyenv root)/versions/${PYTHON_VERSION}"
        if [ -d "$PYENV_VERSION_PATH" ]; then
            print_info "Copying Python from ${PYENV_VERSION_PATH} to ${ABSOLUTE_LOCAL_PYTHON_DIR}..."
            rm -rf "${ABSOLUTE_LOCAL_PYTHON_DIR}" # Clean slate for copy
            mkdir -p "${ABSOLUTE_LOCAL_PYTHON_DIR}"
            if command_exists rsync; then
                rsync -a --delete "${PYENV_VERSION_PATH}/" "${ABSOLUTE_LOCAL_PYTHON_DIR}/"
            else
                cp -RpH "${PYENV_VERSION_PATH}/." "${ABSOLUTE_LOCAL_PYTHON_DIR}/"
            fi
            if [ -x "${ABSOLUTE_LOCAL_PYTHON_DIR}/bin/python3" ]; then
                print_info "Python ${PYTHON_VERSION} copied successfully from pyenv installation."
                PYTHON_CMD_FOUND="${ABSOLUTE_LOCAL_PYTHON_DIR}/bin/python3"
                return 0
            fi
        fi
        print_warning "Failed to copy Python from pyenv. Trying next method."
    else
        print_info "'pyenv' not found."
    fi

    # Try 3: Build Python from source directly (NEW)
    print_info "As a last resort for automated local install, attempting to build Python ${PYTHON_VERSION} from source..."
    if build_python_from_source; then # This sets PYTHON_CMD_FOUND on success
        return 0
    else
        print_error "Building Python ${PYTHON_VERSION} from source failed."
        return 1
    fi
}

# Function to present manual installation options and exit
present_manual_install_options_and_exit() {
    print_error "Automated local installation of Python ${PYTHON_VERSION} failed or was not attempted."
    print_info "\nPlease choose a manual method to install Python ${PYTHON_VERSION} and then re-run './setup.sh':"
    echo "1. Manually install pyenv: https://github.com/pyenv/pyenv#installation."
    echo "   (Then use 'pyenv install ${PYTHON_VERSION}'. Re-run this script when you finish.)"
    echo "2. Manually install using your system's package manager."
    echo "   (e.g., 'brew install python@${PYTHON_VERSION_MAJOR_MINOR}' on macOS; 'sudo apt install python${PYTHON_VERSION_MAJOR_MINOR}' on Debian/Ubuntu)."
    echo "   (Re-run this script when you finish.)"
    echo "3. Manually download and install from python.org: https://www.python.org/downloads/release/python-${PYTHON_VERSION_URL_PART}/"
    echo "   (Ensure it's added to your PATH. Re-run this script when you finish.)"
    echo "4. Exit setup now."

    local manual_choice
    read -p "Choose an option (1/2/3/4 to acknowledge and exit, or just exit): " manual_choice
    case "$manual_choice" in
        1|2|3) print_info "Exiting. Please perform the chosen manual installation and then re-run ./setup.sh." ;;
        *) print_info "Exiting setup." ;;
    esac
    exit 1 # Exit for user to take action
}


# --- Main Python Environment Setup Logic ---
setup_python_environment() {
    print_step "Locating/Setting up Python ${PYTHON_VERSION}"

    # Priority: 1. Existing local, 2. Pyenv, 3. System
    if check_existing_local_python || check_pyenv_python || check_system_python; then
        print_info "Using compatible Python found at: ${PYTHON_CMD_FOUND}"
    else
        print_warning "\nPython ${PYTHON_VERSION} was not found through existing installations (local, pyenv, or system)."
        echo ""
        print_info "How would you like to proceed?"
        echo "1. Attempt to automatically install Python ${PYTHON_VERSION} into a local project directory (./${LOCAL_PYTHON_DIR}/)."
        echo "   (Tries pyenv/python-build if available, then attempts to download & compile from python.org.)"
        echo "   (Compilation requires build tools like a C compiler, 'make', and dev libraries like OpenSSL.)"
        echo "2. Exit setup. You will be guided to install Python ${PYTHON_VERSION} manually."

        local initial_choice
        read -p "Choose an option (1 or 2): " initial_choice

        if [ "$initial_choice" == "1" ]; then
            if ! create_local_python_automatically; then # This tries all automated methods including build from source
                # If create_local_python_automatically fails, it would have printed specific errors.
                # Now, offer the specific manual install guidance menu.
                present_manual_install_options_and_exit
            fi
            # If successful, PYTHON_CMD_FOUND is set by create_local_python_automatically
        else
            print_info "You chose to exit and install Python manually."
            present_manual_install_options_and_exit
        fi
    fi

    # Verify Python command is accessible and correct version
    if [ -z "$PYTHON_CMD_FOUND" ] || ! command_exists "$PYTHON_CMD_FOUND"; then
        print_error "Critical: Python command ($PYTHON_CMD_FOUND) is not accessible or not set after all attempts."
        print_info "Please ensure Python ${PYTHON_VERSION} is correctly installed and accessible, then re-run."
        exit 1
    fi
    
    current_python_version=$("$PYTHON_CMD_FOUND" --version 2>&1 | cut -d' ' -f2)
    if [[ "$current_python_version" != "$PYTHON_VERSION"* ]]; then
        print_error "Python version mismatch: Python command '$PYTHON_CMD_FOUND' provides $current_python_version, but expected $PYTHON_VERSION."
        exit 1
    fi
    
    print_info "\nSuccessfully configured to use Python: $($PYTHON_CMD_FOUND --version) (from $PYTHON_CMD_FOUND)"

    # --- Virtual Environment Setup ---
    print_step "Creating virtual environment ./${VENV_NAME}/"
    # (Virtual env setup logic remains the same as your last version, using $PYTHON_CMD_FOUND)
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
            print_error "Failed to create virtual environment. Check your Python installation ($PYTHON_CMD_FOUND)."
            exit 1
        fi
    fi
    
    print_step "Activating virtual environment for setup"
    source "${PWD}/${VENV_NAME}/bin/activate"
    
    VENV_PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
    print_info "Virtual environment Python version: ${VENV_PYTHON_VERSION}"
    if [[ "$VENV_PYTHON_VERSION" != "$PYTHON_VERSION"* ]]; then
        print_warning "Virtual env Python version ($VENV_PYTHON_VERSION) doesn't match target ($PYTHON_VERSION)."
    fi
    
    print_step "Upgrading pip"
    python -m pip install --upgrade pip
    
    if [ -f "requirements.txt" ]; then
        print_step "Installing packages from requirements.txt"
        python -m pip install -r requirements.txt
    else
        print_warning "requirements.txt not found. Skipping package installation from file."
    fi
    
    print_step "Handling PyTorch installation (if not already installed)"
    # (PyTorch logic as before)
    if ! python -m pip show torch > /dev/null 2>&1; then
        print_info "PyTorch not detected, attempting specific install..."
        if command_exists nvidia-smi; then
            print_info "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
            python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        elif [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
            print_info "Apple Silicon (arm64) detected. Installing PyTorch with MPS support..."
            python -m pip install torch torchvision torchaudio
        else
            print_info "Installing PyTorch CPU version..."
            python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    else
        print_info "PyTorch seems to be already installed."
    fi
    
    print_step "Setting up Jupyter kernel"
    # (Jupyter kernel setup as before)
    PROJECT_NAME=$(basename "$PWD")
    KERNEL_NAME="${PROJECT_NAME}_kernel_py${PYTHON_VERSION_MAJOR_MINOR}"
    DISPLAY_NAME="${PROJECT_NAME} (Python ${PYTHON_VERSION})"
    python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "${DISPLAY_NAME}"
    print_info "Jupyter kernel installed as '${DISPLAY_NAME}'"
    
    print_step "Creating activation script (activate.sh)"
    # (activate.sh creation as before)
    cat > activate.sh << EOL
#!/bin/bash
source "${PWD}/${VENV_NAME}/bin/activate"
if [ -f .env ]; then set -a; source .env; set +a; fi
echo "Virtual environment activated with Python ${PYTHON_VERSION}. Run 'deactivate' to exit."
EOL
    chmod +x activate.sh
    print_info "Created activate.sh - Run 'source activate.sh' to activate."

    if declare -f deactivate > /dev/null && [[ "$(type -t deactivate)" == "function" ]]; then
      deactivate
    fi
}

# Main function
main() {
    print_step "Starting Research Environment Setup"
    setup_python_environment
    print_step "Environment Setup Complete!"
    print_info "Your Python environment and Jupyter kernel are ready."
    print_info "To activate: source activate.sh"
    print_info "Then, to start Jupyter Lab: jupyter-lab"
    print_info "\nNext, run './download_data.sh' for research data."
}

# Run main
main