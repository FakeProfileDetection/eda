#!/usr/bin/env bash
# download_processed_data.sh
# Pull and unpack .tar.gz snapshots from GCS → ./processed_data/
# Default: fetch only the latest snapshot.

set -euo pipefail

BUCKET="fake-profile-detection-eda-bucket"
PREFIX="processed_data"
DEST_DIR="processed_data"

usage() {
  cat <<EOF
Usage: $0 [--all] [--latest] [--hostname [NAME]] [--interactive]

  (no args)             Download only the latest snapshot
  --all                 Download every snapshot
  --latest              Only the most recent by timestamp
  --hostname [NAME]     Only snapshots from host “NAME”
                        If NAME is omitted, pick from a list
  --interactive         Pick a single snapshot by filename
EOF
  exit 1
}

# default → latest
MODE="latest"
HOST_PROVIDED=false
HOST=""

# parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --all)
      MODE="all"; shift
      ;;
    --latest)
      MODE="latest"; shift
      ;;
    --hostname)
      MODE="hostname"; shift
      if [[ $# -gt 0 && ! $1 =~ ^-- ]]; then
        HOST="$1"; HOST_PROVIDED=true
        shift
      fi
      ;;
    --interactive)
      MODE="interactive"; shift
      ;;
    *)
      usage
      ;;
  esac
done

# 1) gather remote .tar.gz URLs, sorted
remote_objs=()
while IFS= read -r uri; do
  remote_objs+=("$uri")
done < <(gsutil ls "gs://$BUCKET/$PREFIX/"*.tar.gz 2>/dev/null | sort)

if (( ${#remote_objs[@]} == 0 )); then
  echo "❌ No .tar.gz found in gs://$BUCKET/$PREFIX/"
  exit 1
fi

# 2) extract filenames
remote_names=()
for uri in "${remote_objs[@]}"; do
  remote_names+=( "${uri##*/}" )
done

# 3) pick which to fetch
to_fetch=()

case $MODE in
  all)
    to_fetch=( "${remote_objs[@]}" )
    ;;

  latest)
    idx=$((${#remote_objs[@]} - 1))
    to_fetch=( "${remote_objs[idx]}" )
    ;;

  interactive)
    echo "Select a snapshot to download:"
    PS3="Your choice: "
    select fname in "${remote_names[@]}"; do
      [[ -n "$fname" ]] || { echo "Invalid."; continue; }
      for i in "${!remote_names[@]}"; do
        if [[ "${remote_names[i]}" == "$fname" ]]; then
          to_fetch=( "${remote_objs[i]}" )
          break 2
        fi
      done
    done
    ;;

  hostname)
    if $HOST_PROVIDED; then
      # filter by given host
      for i in "${!remote_names[@]}"; do
        if [[ "${remote_names[i]}" == *"-${HOST}.tar.gz" ]]; then
          to_fetch+=( "${remote_objs[i]}" )
        fi
      done
      if (( ${#to_fetch[@]} == 0 )); then
        echo "❌ No snapshots found for host ‘$HOST’"
        exit 1
      fi

    else
      # no host given → build list of hosts + odd filenames
      host_list=()
      for name in "${remote_names[@]}"; do
        if [[ $name =~ ^processed_data-[0-9T-]+-([^-]+)\.tar\.gz$ ]]; then
          h="${BASH_REMATCH[1]}"
          if ! printf '%s\n' "${host_list[@]}" | grep -qx "$h"; then
            host_list+=( "$h" )
          fi
        fi
      done

      options=( "${host_list[@]}" )
      for name in "${remote_names[@]}"; do
        if ! [[ $name =~ ^processed_data-[0-9T-]+-[^-]+\.tar\.gz$ ]]; then
          options+=( "$name" )
        fi
      done

      echo "Select a host or specific snapshot:"
      PS3="Choice: "
      select choice in "${options[@]}"; do
        [[ -n "$choice" ]] || { echo "Invalid."; continue; }
        selected="$choice"
        break
      done

      # if they picked a host
      if printf '%s\n' "${host_list[@]}" | grep -qx "$selected"; then
        for i in "${!remote_names[@]}"; do
          if [[ "${remote_names[i]}" == *"-${selected}.tar.gz" ]]; then
            to_fetch+=( "${remote_objs[i]}" )
          fi
        done
      else
        # they picked exact filename
        for i in "${!remote_names[@]}"; do
          if [[ "${remote_names[i]}" == "$selected" ]]; then
            to_fetch=( "${remote_objs[i]}" )
            break
          fi
        done
      fi
    fi
    ;;
esac

# 4) download & extract
mkdir -p "$DEST_DIR"
for uri in "${to_fetch[@]}"; do
  fname="${uri##*/}"
  echo "⬇️  Downloading $fname …"
  gsutil cp "$uri" "$DEST_DIR/"

  echo "🗜  Extracting $fname …"
  tar -xzf "$DEST_DIR/$fname" -C "$DEST_DIR"

  echo "🧹 Removing $fname …"
  rm "$DEST_DIR/$fname"
done

echo "✅ Done. Snapshots are under $DEST_DIR/"
