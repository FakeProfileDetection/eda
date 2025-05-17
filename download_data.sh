#!/bin/bash
# download_data.sh - Research data download script
# ----------------------------------------------------------
# Downloads research data from Google Cloud Storage bucket
# and sets up environment variables for data path.

# Source utility functions
# Ensure utils.sh is in the same directory or provide a correct path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=utils.sh
source "${SCRIPT_DIR}/utils.sh"

# Make script exit on first error
set -e

# Google Cloud Storage settings
PROJECT_ID="fake-profile-detection-460117"
BUCKET_NAME="fake-profile-detection-eda-bucket"
FILE_PATH="loadable_data/loadable_Combined_HU_HT.tar.gz"
DATA_DIR_NAME="data_dump"
ENV_FILE=".env"
ENV_VAR_NAME="DATA_PATH"

# Function to download and extract data from Google Cloud Storage
download_and_extract_data() {
    print_step "Checking Google Cloud Authentication"

    # Check if gcloud is installed
    if ! command_exists gcloud; then
        print_error "gcloud CLI not found. Please install Google Cloud SDK first."
        echo "Visit: https://cloud.google.com/sdk/docs/install"
        return 1
    fi

    # Check if user is logged in
    gcloud_account=$(gcloud config get-value account 2>/dev/null)
    if [ -z "$gcloud_account" ]; then
        print_warning "Not logged in to gcloud. Attempting login now."
        gcloud auth login
        if [ $? -ne 0 ]; then
            print_error "Login failed. Please run 'gcloud auth login' manually."
            return 1
        fi
    fi

    # Get current logged in account
    gcloud_account=$(gcloud config get-value account 2>/dev/null)
    print_info "Logged in to gcloud as: $gcloud_account"

    # Set the project
    current_project=$(gcloud config get-value project 2>/dev/null)
    if [ "$current_project" != "$PROJECT_ID" ]; then
        print_info "Setting project to $PROJECT_ID..."
        gcloud config set project "$PROJECT_ID"
        if [ $? -ne 0 ]; then
            print_error "Failed to set project to $PROJECT_ID. Please check permissions or set manually."
            return 1
        fi
    else
        print_info "Current gcloud project is already set to $PROJECT_ID."
    fi

    # Check for application default credentials
    print_info "Checking for application default credentials..."
    if ! gcloud auth application-default print-access-token &>/dev/null; then
        print_warning "Application default credentials not found. Attempting to set up now..."
        gcloud auth application-default login
        if [ $? -ne 0 ]; then
            print_error "Failed to set up application default credentials. Please run 'gcloud auth application-default login' manually."
            return 1
        fi
        gcloud auth application-default set-quota-project "$PROJECT_ID"
        if [ $? -ne 0 ]; then
            print_warning "Failed to set quota project for application default credentials. This might cause issues."
        fi
    else
        print_info "Application default credentials found."
    fi

    print_step "Checking for existing data"

    # Check if data already exists and determine if we need to download
    NEED_DOWNLOAD=true
    DATA_FILE_DEST_PATH="${PWD}/${DATA_DIR_NAME}/$(basename "$FILE_PATH")"

    if [ -d "${PWD}/${DATA_DIR_NAME}" ]; then
        CSV_COUNT=$(find "${PWD}/${DATA_DIR_NAME}" -name "*.csv" -type f 2>/dev/null | wc -l)
        if [ "$CSV_COUNT" -gt 0 ]; then
            print_info "Found $CSV_COUNT CSV files in ${PWD}/${DATA_DIR_NAME} directory."
            read -p "Data files appear to exist. Do you want to re-download and overwrite? (y/N): " REDOWNLOAD
            if [[ $REDOWNLOAD != "y" && $REDOWNLOAD != "Y" ]]; then
                print_info "Skipping download. Using existing data."
                NEED_DOWNLOAD=false
            else
                print_info "Proceeding to re-download data."
                # Consider cleaning up old data if re-downloading
                # print_warning "Removing old data from ${PWD}/${DATA_DIR_NAME}..."
                # rm -rf "${PWD}/${DATA_DIR_NAME:?}/"* # Added :? for safety
            fi
        else
            print_info "Data directory ${PWD}/${DATA_DIR_NAME} exists but no CSV files found. Will download data."
        fi
    else
        print_info "Data directory ${PWD}/${DATA_DIR_NAME} not found. Creating it and will download data."
        mkdir -p "${PWD}/${DATA_DIR_NAME}"
    fi

    if $NEED_DOWNLOAD; then
        print_step "Downloading Data from Google Cloud Storage"

        # Calculate the source and destination paths
        SOURCE_PATH="gs://${BUCKET_NAME}/${FILE_PATH}"

        print_info "Downloading from: $SOURCE_PATH"
        print_info "Saving to: $DATA_FILE_DEST_PATH"

        # Download the file
        gcloud storage cp "$SOURCE_PATH" "$DATA_FILE_DEST_PATH"

        # Check if download was successful
        if [ $? -ne 0 ]; then
            print_error "Download failed. Please check error messages above."
            return 1
        fi

        print_info "Download complete!"

        print_step "Extracting Archive"

        # Extract the TAR.GZ file
        print_info "Extracting $DATA_FILE_DEST_PATH to: ${PWD}/${DATA_DIR_NAME}"

        # Extract the file
        tar -xzf "$DATA_FILE_DEST_PATH" -C "${PWD}/${DATA_DIR_NAME}"

        # Check if extraction was successful
        if [ $? -ne 0 ]; then
            print_error "Extraction failed. Please check error messages above."
            return 1
        fi

        print_info "Extraction complete!"

        # Optionally remove the archive to save space
        read -p "Remove the downloaded archive ($DATA_FILE_DEST_PATH) to save space? (Y/n): " REMOVE_ARCHIVE
        if [[ $REMOVE_ARCHIVE != "n" && $REMOVE_ARCHIVE != "N" ]]; then
            rm "$DATA_FILE_DEST_PATH"
            print_info "Archive removed."
        fi
    fi

    # Set up environment variable for DATA_PATH
    print_step "Setting Up Data Path Environment Variable"

    ABSOLUTE_DATA_PATH=$(realpath "${PWD}/${DATA_DIR_NAME}")

    # Create or update .env file
    if [ -f "$ENV_FILE" ] && grep -q "^${ENV_VAR_NAME}=" "$ENV_FILE" 2>/dev/null; then
        # Update existing variable
        # Using a temporary file for sed -i compatibility on macOS and Linux
        tmp_env_file=$(mktemp)
        sed "s|^${ENV_VAR_NAME}=.*|${ENV_VAR_NAME}=${ABSOLUTE_DATA_PATH}|" "$ENV_FILE" > "$tmp_env_file" && mv "$tmp_env_file" "$ENV_FILE"
        print_info "Updated $ENV_VAR_NAME in $ENV_FILE."
    else
        # Create new variable or new file
        echo "${ENV_VAR_NAME}=${ABSOLUTE_DATA_PATH}" >> "$ENV_FILE"
        print_info "Added $ENV_VAR_NAME to $ENV_FILE."
    fi

    print_info "Environment variable $ENV_VAR_NAME set to $ABSOLUTE_DATA_PATH"
    print_info "This will be available in your shell after sourcing 'activate.sh'."

    return 0
}

# Main function for data download script
main_data() {
    print_step "Starting Research Data Download & Setup"

    if download_and_extract_data; then
        print_step "Data Setup Complete!"
        print_info "Your research data has been downloaded/verified, and the DATA_PATH environment variable is set in '.env'."
        print_info "If your environment is active (via 'source activate.sh'), the DATA_PATH variable should be available."
        print_info "\nYou can access the data in your code using:"
        echo -e "\nimport os"
        echo -e "data_path = os.environ.get('$ENV_VAR_NAME')"
        echo -e "if data_path:"
        echo -e "    print(f'Data path: {data_path}')"
        echo -e "    # Example usage:"
        echo -e "    # import pandas as pd"
        echo -e "    # file_to_load = os.path.join(data_path, 'your_sub_folder', 'your_file.csv')"
        echo -e "    # data = pd.read_csv(file_to_load)"
        echo -e "else:"
        echo -e "    print(f'$ENV_VAR_NAME not set. Ensure .env is sourced by your activation script or shell.')\n"
    else
        print_error "Data setup failed. Please check the error messages above."
    fi
}

# Run the main data download function
main_data