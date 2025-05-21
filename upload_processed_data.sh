#!/usr/bin/env bash
# upload_processed_data.sh
# Usage:
#   ./upload_processed_data.sh [--no-archive] <dir-or-archive>
#
# By default, it will tar.gz the given directory, upload it,
# then delete the archive. If you pass --no-archive, you must
# supply an existing .tar.gz and it will be uploaded as-is.

set -euo pipefail

BUCKET="fake-profile-detection-eda-bucket"
PREFIX="processed_data"

usage() {
  cat <<EOF
Usage: $0 [--no-archive] <processed_data-<timestamp>-<hostname> | archive.tar.gz>

Options:
  --no-archive   Don’t create a .tar.gz; treat the final argument as an existing archive
EOF
  exit 1
}

# parse flag
SKIP_ARCHIVE=false
if [[ "${1-}" == "--no-archive" ]]; then
  SKIP_ARCHIVE=true
  shift
fi

# verify argument
if [[ $# -ne 1 ]]; then
  usage
fi

INPUT="$1"

if [[ "$SKIP_ARCHIVE" == false ]]; then
  # must be a directory
  if [[ ! -d "$INPUT" ]]; then
    echo "❌ Directory not found: $INPUT"
    exit 1
  fi
  ARCHIVE="${INPUT}.tar.gz"
  echo "🗜  Creating archive $ARCHIVE …"
  tar czf "$ARCHIVE" "$INPUT"
else
  # must be a .tar.gz file
  ARCHIVE="$INPUT"
  if [[ ! -f "$ARCHIVE" || "${ARCHIVE##*.}" != "gz" ]]; then
    echo "❌ File not found or not a .tar.gz: $ARCHIVE"
    exit 1
  fi
fi

echo "☁️  Uploading $ARCHIVE → gs://$BUCKET/$PREFIX/ …"
gsutil cp "$ARCHIVE" "gs://$BUCKET/$PREFIX/"

if [[ "$SKIP_ARCHIVE" == false ]]; then
  echo "🧹 Removing local archive $ARCHIVE …"
  rm "$ARCHIVE"
fi

echo "✅ Upload complete."
