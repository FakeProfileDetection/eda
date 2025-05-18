#!/usr/bin/env bash
# ------------------------------------------------------------------
# download_data.sh – research data bootstrap (headless‑aware)
# ------------------------------------------------------------------
# * Downloads a .tar.gz dataset from Google Cloud Storage
# * Creates/updates .env with DATA_PATH
# * Works on macOS (Homebrew) & Linux without global config changes
# * Flags:
#     --non-interactive   run unattended (skip y/N prompts)
#     --headless          use --no-launch-browser for gcloud logins
# ------------------------------------------------------------------

set -euo pipefail

# ------------------------------------------------- config ---------
PROJECT_ID="fake-profile-detection-460117"
BUCKET_NAME="fake-profile-detection-eda-bucket"
FILE_PATH="loadable_data/loadable_Combined_HU_HT.tar.gz"

DATA_DIR_NAME="data_dump"
ENV_FILE=".env"
ENV_VAR_NAME="DATA_PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=utils.sh
source "${SCRIPT_DIR}/utils.sh"

# ------------------------------------------------- flags ----------
NON_INTERACTIVE=false
HEADLESS=false
for arg in "$@"; do
  case "$arg" in
    --non-interactive) NON_INTERACTIVE=true ;;
    --headless)        HEADLESS=true        ;;
  esac
done

# ------------------------------------------------- helpers --------
install_gcloud_macos() {
  [[ $(uname) == Darwin ]] || return 1
  command_exists gcloud && return 0
  command_exists brew   || return 1
  print_info "Installing Google Cloud SDK via Homebrew (one‑time)…"
  brew install --quiet --cask google-cloud-sdk
}

ensure_gcloud() {
  command_exists gcloud && return 0
  install_gcloud_macos   || {
      print_error "gcloud CLI not found. Install manually:"
      echo "  • macOS: brew install --cask google-cloud-sdk"
      echo "  • Linux: https://cloud.google.com/sdk/docs/install"
      return 1
  }
}

confirm_or_skip() {            # $1 = prompt, true → proceed
  $NON_INTERACTIVE && return 0
  read -r -p "$1 (y/N): " ans
  [[ "$ans" =~ ^[Yy]$ ]]
}

maybe_gcloud_login() {
  if [[ -z $(gcloud config get-value account 2>/dev/null) ]]; then
      print_info "No gcloud account; starting login (headless=$HEADLESS)…"
      if $HEADLESS; then
          gcloud auth login --no-launch-browser || { print_error "Login failed"; exit 1; }
      else
          gcloud auth login || { print_error "Login failed"; exit 1; }
      fi
  fi
  print_info "Logged in as: $(gcloud config get-value account)"
}

maybe_adc_login() {
  if ! gcloud auth application-default print-access-token &>/dev/null; then
      print_info "Creating application‑default credentials…"
      if $HEADLESS; then
          gcloud auth application-default login --no-launch-browser
      else
          gcloud auth application-default login
      fi
      gcloud auth application-default set-quota-project "$PROJECT_ID" || true
  fi
}

# ------------------------------------------------ main ------------
main() {
  print_step "Checking Google Cloud CLI"
  ensure_gcloud || exit 1

  maybe_gcloud_login

  if [[ $(gcloud config get-value project 2>/dev/null) != "$PROJECT_ID" ]]; then
      gcloud config set project "$PROJECT_ID" >/dev/null
  fi
  maybe_adc_login

  # ----- data directory & (re)download? -----
  mkdir -p "$DATA_DIR_NAME"
  CSV_COUNT=$(find "$DATA_DIR_NAME" -name '*.csv' -type f 2>/dev/null | wc -l)

  NEED_DL=true
  if (( CSV_COUNT > 0 )); then
      if confirm_or_skip "Found $CSV_COUNT CSV files; re-download and overwrite?"; then
          NEED_DL=true
      else
          NEED_DL=false
      fi
  fi

  if $NEED_DL; then
      print_step "Downloading dataset"
      SRC="gs://${BUCKET_NAME}/${FILE_PATH}"
      DEST="${DATA_DIR_NAME}/$(basename "$FILE_PATH")"
      gcloud storage cp "$SRC" "$DEST"

      print_step "Extracting"
      tar -xzf "$DEST" -C "$DATA_DIR_NAME"

      if confirm_or_skip "Remove the downloaded archive to save space?"; then
          rm -f "$DEST"
      fi
  fi

  # ----- .env update -----
  print_step "Setting DATA_PATH in ${ENV_FILE}"
  ABS_PATH="$(realpath "$DATA_DIR_NAME")"

  if grep -q "^${ENV_VAR_NAME}=" "$ENV_FILE" 2>/dev/null; then
      sed -i.bak "s|^${ENV_VAR_NAME}=.*|${ENV_VAR_NAME}=${ABS_PATH}|" "$ENV_FILE"
      rm -f "${ENV_FILE}.bak"
  else
      echo "${ENV_VAR_NAME}=${ABS_PATH}" >> "$ENV_FILE"
  fi
  print_info "DATA_PATH set to ${ABS_PATH}"

  print_step "Done!"
  print_info "Activate your venv and run:  source activate.sh"
  print_info "Then notebooks will see os.environ['${ENV_VAR_NAME}']"
}

main "$@"
