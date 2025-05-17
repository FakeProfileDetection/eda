#!/bin/bash
# setup.sh - Fixed environment setup script with proper pyenv handling
# This script accounts for different Python setups and ensures Python 3.12.5 is used

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
RED="\033[0;31m"
NC="\033[0m" # No Color

print_step() {
    echo -e "\n${BOLD}${GREEN}=== $1 ===${NC}\n"
}

print_info() {
    echo -e "${BLUE}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

# Set target Python version and environment name
PYTHON_VERSION="3.12.5"
VENV_NAME="venv-${PYTHON_VERSION}"
LOCAL_PYTHON_DIR="local-python"

# Check if running on Windows
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    print_error "This script is for macOS/Linux. Please use setup.bat on Windows."
    exit 1
fi

print_step "Checking Python Environment"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to set up pyenv (separated from the main script for clarity)
use_pyenv_python() {
    print_info "Setting up Python ${PYTHON_VERSION} using pyenv..."
    
    # Initialize pyenv
    if [[ -z "${PYENV_ROOT}" ]]; then
        export PYENV_ROOT="${HOME}/.pyenv"
    fi
    
    if command_exists pyenv; then
        eval "$(pyenv init -)"
        eval "$(pyenv init --path 2>/dev/null || true)"
        
        # Check if pyenv version exists
        if pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
            print_info "Python ${PYTHON_VERSION} is installed in pyenv."
            pyenv shell "${PYTHON_VERSION}"
            
            # Verify Python is accessible
            if command_exists python; then
                python_version=$(python --version 2>&1 | cut -d' ' -f2)
                if [ "$python_version" = "$PYTHON_VERSION" ]; then
                    print_info "Successfully activated Python ${PYTHON_VERSION} with pyenv."
                    return 0
                else
                    print_warning "pyenv shell did not activate the correct Python version."
                fi
            else
                print_warning "Python command not available after pyenv shell."
            fi
        else
            print_warning "Python ${PYTHON_VERSION} not found in pyenv."
            
            # Offer to install with pyenv
            read -p "Install Python ${PYTHON_VERSION} using pyenv? (y/N): " install_with_pyenv
            if [[ $install_with_pyenv == "y" || $install_with_pyenv == "Y" ]]; then
                print_info "Installing Python ${PYTHON_VERSION} with pyenv..."
                pyenv install ${PYTHON_VERSION}
                if [ $? -eq 0 ]; then
                    pyenv shell ${PYTHON_VERSION}
                    print_info "Python ${PYTHON_VERSION} installed and activated with pyenv."
                    return 0
                else
                    print_error "Failed to install Python ${PYTHON_VERSION} with pyenv."
                    return 1
                fi
            else
                return 1
            fi
        fi
    else
        print_warning "pyenv detected but not properly configured."
        return 1
    fi
    
    return 1
}

# Function to use system Python if available at correct version
use_system_python() {
    print_info "Checking for system Python..."
    
    if command_exists python3; then
        system_python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_info "System Python version: ${system_python_version}"
        
        if [ "$system_python_version" = "$PYTHON_VERSION" ]; then
            print_info "System Python matches required version ${PYTHON_VERSION}."
            return 0
        else
            print_warning "System Python (${system_python_version}) does not match required version (${PYTHON_VERSION})."
            return 1
        fi
    fi
    
    return 1
}

# Function to create a local Python installation
create_local_python() {
    print_step "Setting up local Python ${PYTHON_VERSION}"
    
    # Check if Python is already installed locally
    if [ -x "${LOCAL_PYTHON_DIR}/bin/python3" ]; then
        local_python_version=$(${LOCAL_PYTHON_DIR}/bin/python3 --version 2>&1 | cut -d' ' -f2)
        if [ "$local_python_version" = "$PYTHON_VERSION" ]; then
            print_info "Local Python ${PYTHON_VERSION} is already installed."
            return 0
        else
            print_warning "Local Python installation found but version is ${local_python_version}, not ${PYTHON_VERSION}."
        fi
    fi
    
    # Check for python-build
    if command_exists python-build; then
        # Build Python locally
        print_info "Building Python ${PYTHON_VERSION} locally (this may take several minutes)..."
        mkdir -p "${LOCAL_PYTHON_DIR}"
        python-build "${PYTHON_VERSION}" "${LOCAL_PYTHON_DIR}"
        
        # Verify installation
        if [ -x "${LOCAL_PYTHON_DIR}/bin/python3" ]; then
            local_python_version=$(${LOCAL_PYTHON_DIR}/bin/python3 --version 2>&1 | cut -d' ' -f2)
            if [ "$local_python_version" = "$PYTHON_VERSION" ]; then
                print_info "Local Python ${PYTHON_VERSION} installed successfully."
                return 0
            else
                print_error "Local Python installation failed. Version mismatch: ${local_python_version}"
                return 1
            fi
        else
            print_error "Local Python installation failed."
            return 1
        fi
    else
        print_warning "python-build not found. Attempting to use pyenv to build Python..."
        
        if command_exists pyenv; then
            mkdir -p "${LOCAL_PYTHON_DIR}"
            PYENV_VERSION=${PYTHON_VERSION} PYTHON_BUILD_CACHE_PATH="${PWD}/.python-build-cache" \
            pyenv install --verbose ${PYTHON_VERSION}
            
            # Copy the built Python to our local directory
            cp -r "${HOME}/.pyenv/versions/${PYTHON_VERSION}" "${LOCAL_PYTHON_DIR}"
            
            # Verify installation
            if [ -x "${LOCAL_PYTHON_DIR}/bin/python3" ]; then
                print_info "Local Python ${PYTHON_VERSION} installed successfully using pyenv."
                return 0
            else
                print_error "Failed to install local Python using pyenv."
                return 1
            fi
        else
            print_error "Neither python-build nor pyenv is available. Cannot install local Python."
            print_info "Please install Python ${PYTHON_VERSION} using one of these methods:"
            print_info "1. Download from python.org"
            print_info "2. Install pyenv: https://github.com/pyenv/pyenv#installation"
            print_info "3. Use your system's package manager"
            return 1
        fi
    fi
}

# Main function to set up the environment
main() {
    # Try to use Python from various sources
    if command_exists pyenv; then
        print_info "pyenv detected."
        
        # Try using pyenv Python
        if use_pyenv_python; then
            PYTHON_CMD="python"
        else
            print_warning "Failed to use pyenv Python ${PYTHON_VERSION}."
            PYTHON_CMD=""
        fi
    else
        print_info "pyenv not detected."
        PYTHON_CMD=""
    fi
    
    # If pyenv didn't work, try system Python
    if [ -z "$PYTHON_CMD" ] && use_system_python; then
        PYTHON_CMD="python3"
    fi
    
    # If still no Python, offer to create a local installation
    if [ -z "$PYTHON_CMD" ]; then
        print_warning "Python ${PYTHON_VERSION} not found in pyenv or system."
        
        # Offer options to the user
        echo "Options:"
        echo "1. Install Python ${PYTHON_VERSION} locally in this project directory"
        echo "2. Exit and install Python ${PYTHON_VERSION} manually"
        
        read -p "Choose an option (1/2): " option
        
        case $option in
            1)
                if create_local_python; then
                    PYTHON_CMD="${LOCAL_PYTHON_DIR}/bin/python3"
                else
                    print_error "Failed to create local Python installation."
                    exit 1
                fi
                ;;
            *)
                print_info "Exiting. Please install Python ${PYTHON_VERSION} and run this script again."
                exit 0
                ;;
        esac
    fi
    
    # Verify Python command is accessible
    if ! command_exists "$PYTHON_CMD"; then
        print_error "Python command ($PYTHON_CMD) is not accessible."
        exit 1
    fi
    
    # Verify Python version
    python_version=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    if [ "$python_version" != "$PYTHON_VERSION" ]; then
        print_error "Python version mismatch: got $python_version, expected $PYTHON_VERSION"
        exit 1
    fi
    
    print_info "Using Python: $($PYTHON_CMD --version)"
    print_info "Python path: $(command -v $PYTHON_CMD)"
    
    # Create virtual environment
    print_step "Creating virtual environment"
    if [ -d "$VENV_NAME" ]; then
        read -p "Virtual environment already exists. Recreate? (y/N): " recreate
        if [[ $recreate == "y" || $recreate == "Y" ]]; then
            rm -rf "$VENV_NAME"
        else
            print_info "Using existing virtual environment."
        fi
    fi
    
    if [ ! -d "$VENV_NAME" ]; then
        print_info "Creating new virtual environment with Python ${PYTHON_VERSION}..."
        $PYTHON_CMD -m venv "$VENV_NAME"
    fi
    
    # Activate virtual environment
    print_step "Activating virtual environment"
    source "${VENV_NAME}/bin/activate"
    
    # Verify Python version in virtual environment
    VENV_PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
    print_info "Virtual environment Python version: ${VENV_PYTHON_VERSION}"
    
    # Check if virtual environment is properly activated
    if [ "$VENV_PYTHON_VERSION" != "$PYTHON_VERSION" ]; then
        print_warning "Virtual environment Python version mismatch: got $VENV_PYTHON_VERSION, expected $PYTHON_VERSION"
        print_warning "Proceeding anyway, but there might be issues."
    fi
    
    # Upgrade pip
    print_step "Upgrading pip"
    python -m pip install --upgrade pip
    
    # Install packages
    print_step "Installing required packages"
    python -m pip install polars pandas numpy scikit-learn matplotlib seaborn jupyterlab ipykernel tqdm python-dotenv pydrive2
    
    # Check for GPU and install appropriate PyTorch version
    print_step "Detecting GPU for PyTorch installation"
    
    # Check for NVIDIA GPU
    if command_exists nvidia-smi; then
        print_info "NVIDIA GPU detected! Installing PyTorch with CUDA support..."
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
        # Check for Apple Silicon
        print_info "Apple Silicon detected! Installing PyTorch with MPS support..."
        python -m pip install torch torchvision torchaudio
    else
        print_info "No specialized GPU detected. Installing PyTorch CPU version..."
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Setup Jupyter kernel
    print_step "Setting up Jupyter kernel"
    
    # Get project name from directory
    PROJECT_NAME=$(basename "$(pwd)")
    KERNEL_NAME="${PROJECT_NAME}_kernel"
    DISPLAY_NAME="${PROJECT_NAME} (Python ${PYTHON_VERSION})"
    
    python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "${DISPLAY_NAME}"
    print_info "Jupyter kernel installed as '${DISPLAY_NAME}'"
    
    # Create activation script
    print_step "Creating activation script"
    cat > activate.sh << EOL
#!/bin/bash
# Activate the virtual environment
source "${VENV_NAME}/bin/activate"

# Set environment variables from .env if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

echo "Virtual environment activated with Python ${PYTHON_VERSION}. Run 'deactivate' to exit."
EOL
    
    chmod +x activate.sh
    print_info "Created activate.sh - Run 'source activate.sh' to activate the environment"
    
    # Download team data
    if [ -f "simple_setup.py" ]; then
        print_step "Downloading team data"
        python simple_setup.py
    else
        print_warning "simple_setup.py not found. Skipping data download."
    fi
    
    print_step "Setup complete!"
    print_info "To activate the environment: source activate.sh"
    print_info "To start Jupyter Lab: jupyter-lab"
    
    # Deactivate virtual environment
    deactivate
}

# Run the main function
main
