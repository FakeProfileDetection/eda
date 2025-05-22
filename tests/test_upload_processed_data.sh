#!/bin/sh

# Determine the absolute path to the repository root (where upload_processed_data.sh is)
# and the tests directory.
# THIS_SCRIPT_PATH is the path to this test script itself.
THIS_SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)" # Absolute path to tests directory
REPO_ROOT="$(cd "${THIS_SCRIPT_PATH}/.." && pwd)" # Absolute path to repo root (/app)
SCRIPT_UNDER_TEST="${REPO_ROOT}/upload_processed_data.sh"
GSUTIL_MOCK_SCRIPT="${THIS_SCRIPT_PATH}/gsutil_mock.sh"
GSUTIL_MOCK_LOG="${THIS_SCRIPT_PATH}/gsutil_mock_calls.log"

# Global setup for all tests
original_path=""

oneTimeSetUp() {
  # Create the mock gsutil script
  cat > "${GSUTIL_MOCK_SCRIPT}" <<EOF
#!/bin/sh
# Log all arguments to a file
echo "\$@" >> "${GSUTIL_MOCK_LOG}"
# Check if the first argument is 'cp'
if [ "\$1" = "cp" ]; then
  # Check if there are at least two more arguments (source and destination)
  if [ "\$#" -ge 3 ]; then
    exit 0
  else
    # Incorrect cp arguments
    exit 1 # Should cause gsutil to error, and script to error due to set -e
  fi
else
  # Not a cp command, or not a recognized command for this mock
  exit 0 # Behave neutrally for other commands
fi
EOF
  chmod +x "${GSUTIL_MOCK_SCRIPT}"

  # Create a symlink named 'gsutil' in the tests directory pointing to the mock script
  ln -sf "${GSUTIL_MOCK_SCRIPT}" "${THIS_SCRIPT_PATH}/gsutil"

  # Ensure the script under test is executable
  if [ -f "${SCRIPT_UNDER_TEST}" ]; then
    chmod +x "${SCRIPT_UNDER_TEST}"
  else
    echo "FATAL: Script ${SCRIPT_UNDER_TEST} not found from $(pwd)" >&2
    exit 1 # Fail fast if script under test is missing
  fi

  original_path=$PATH
  # Prepend the directory of gsutil_mock.sh to PATH
  export PATH="${THIS_SCRIPT_PATH}:$PATH"
}

# Per-test setup
setUp() {
  # Clear the mock gsutil calls log and ensure it exists
  rm -f "${GSUTIL_MOCK_LOG}"
  touch "${GSUTIL_MOCK_LOG}"
  # Test artifacts (dummy dirs, dummy archives) are created in THIS_SCRIPT_PATH.
  # Archives created by upload_processed_data.sh (when it does archiving)
  # will also be in THIS_SCRIPT_PATH based on its logic: ARCHIVE="${INPUT}.tar.gz"
  # where INPUT is an absolute path to something in THIS_SCRIPT_PATH.
}

# Per-test cleanup
tearDown() {
  # Cleanup for testUploadDirectorySuccess
  rm -rf "${THIS_SCRIPT_PATH}/test_dir_upload"
  rm -f "${THIS_SCRIPT_PATH}/test_dir_upload.tar.gz" # Archive created by script

  # Cleanup for testUploadArchiveSuccess
  rm -f "${THIS_SCRIPT_PATH}/test_archive.tar.gz" # Original archive provided to script

  # Cleanup for testErrorNotTarGzArchive
  rm -f "${THIS_SCRIPT_PATH}/dummy_file.txt"
}

# Global cleanup after all tests
oneTimeTearDown() {
  export PATH=$original_path
  rm -f "${THIS_SCRIPT_PATH}/gsutil" # Remove the symlink
  rm -f "${GSUTIL_MOCK_SCRIPT}"
  rm -f "${GSUTIL_MOCK_LOG}"
}

# Test cases
testUploadDirectorySuccess() {
  local test_dir="${THIS_SCRIPT_PATH}/test_dir_upload"
  local archive_path="${test_dir}.tar.gz" # Based on script logic: ARCHIVE="${INPUT}.tar.gz"

  # 1. Create dummy directory and file
  mkdir "${test_dir}"
  echo "dummy content" > "${test_dir}/dummy.txt"

  # 2. Run the script targeting this directory
  #    Script CWD is REPO_ROOT. INPUT to script is absolute path to test_dir.
  #    Archive is created at archive_path (absolute).
  "${SCRIPT_UNDER_TEST}" "${test_dir}"
  local exit_code=$?

  # 3. Assert exit code
  assertEquals "testUploadDirectorySuccess: Script exit code" 0 ${exit_code}

  # 4. Assert archive was created (implicitly by script not erroring, explicit check not required by prompt for now)
  #    assertTrue "Archive ${archive_path} should have been created." "[ -f \"${archive_path}\" ]"

  # 5. Assert gsutil mock was called correctly (matches buggy script destination)
  local expected_gsutil_call="cp ${archive_path} gs://fake-profile-detection-eda-bucket/processed_data/"
  local actual_gsutil_call=$(cat "${GSUTIL_MOCK_LOG}" | tr -d '\n')
  assertEquals "testUploadDirectorySuccess: gsutil cp command" "${expected_gsutil_call}" "${actual_gsutil_call}"

  # 6. Assert local archive was removed by the script
  assertFalse "testUploadDirectorySuccess: Local archive ${archive_path} should be removed." \
    "[ -f \"${archive_path}\" ]"
}

testUploadArchiveSuccess() {
  local test_archive="${THIS_SCRIPT_PATH}/test_archive.tar.gz"

  # 1. Create dummy archive file
  touch "${test_archive}"

  # 2. Run the script with --no-archive
  "${SCRIPT_UNDER_TEST}" --no-archive "${test_archive}"
  local exit_code=$?

  # 3. Assert exit code
  assertEquals "testUploadArchiveSuccess: Script exit code" 0 ${exit_code}

  # 4. Assert gsutil mock was called correctly (matches buggy script destination)
  local expected_gsutil_call="cp ${test_archive} gs://fake-profile-detection-eda-bucket/processed_data/"
  local actual_gsutil_call=$(cat "${GSUTIL_MOCK_LOG}" | tr -d '\n')
  assertEquals "testUploadArchiveSuccess: gsutil cp command" "${expected_gsutil_call}" "${actual_gsutil_call}"

  # 5. Assert local archive was NOT removed
  assertTrue "testUploadArchiveSuccess: Local archive ${test_archive} should not be removed." \
    "[ -f \"${test_archive}\" ]"
}

testErrorNonExistentDirectory() {
  local dir_name="non_existent_dir_for_test"
  
  local all_output_plus_rc=$("${SCRIPT_UNDER_TEST}" "${dir_name}" 2>&1; echo $?)
  local exit_code=$(echo "${all_output_plus_rc}" | tail -n 1)
  local output=$(echo "${all_output_plus_rc}" | sed '$d')

  assertEquals "testErrorNonExistentDirectory: Script exit code" 1 "${exit_code}"
  assertContains "testErrorNonExistentDirectory: Error message" \
    "${output}" "❌ Directory not found: ${dir_name}"
}

testErrorNonExistentArchive() {
  local archive_name="non_existent_archive_for_test.tar.gz"

  local all_output_plus_rc=$("${SCRIPT_UNDER_TEST}" --no-archive "${archive_name}" 2>&1; echo $?)
  local exit_code=$(echo "${all_output_plus_rc}" | tail -n 1)
  local output=$(echo "${all_output_plus_rc}" | sed '$d')

  assertEquals "testErrorNonExistentArchive: Script exit code" 1 "${exit_code}"
  assertContains "testErrorNonExistentArchive: Error message" \
    "${output}" "❌ File not found or not a .tar.gz: ${archive_name}"
}

testErrorNotTarGzArchive() {
  local dummy_file="${THIS_SCRIPT_PATH}/dummy_file.txt"
  touch "${dummy_file}"

  local all_output_plus_rc=$("${SCRIPT_UNDER_TEST}" --no-archive "${dummy_file}" 2>&1; echo $?)
  local exit_code=$(echo "${all_output_plus_rc}" | tail -n 1)
  local output=$(echo "${all_output_plus_rc}" | sed '$d')

  assertEquals "testErrorNotTarGzArchive: Script exit code" 1 "${exit_code}"
  assertContains "testErrorNotTarGzArchive: Error message" \
    "${output}" "❌ File not found or not a .tar.gz: ${dummy_file}"
  # tearDown will clean up dummy_file.txt
}

testErrorIncorrectArgCount() {
  # $0 in the script will be the full path to SCRIPT_UNDER_TEST
  local usage_hint="Usage: ${SCRIPT_UNDER_TEST} [--no-archive]"

  # Sub-case 1: No arguments
  local all_output_no_args_plus_rc=$("${SCRIPT_UNDER_TEST}" 2>&1; echo $?)
  local exit_code_no_args=$(echo "${all_output_no_args_plus_rc}" | tail -n 1)
  local output_no_args=$(echo "${all_output_no_args_plus_rc}" | sed '$d')
  
  assertEquals "testErrorIncorrectArgCount (no args): Script exit code" 1 "${exit_code_no_args}"
  assertContains "testErrorIncorrectArgCount (no args): Usage message" \
    "${output_no_args}" "${usage_hint}"

  # Sub-case 2: Too many arguments (after potential --no-archive shift)
  # The script expects exactly one argument after flags are processed.
  # So, SCRIPT --flag arg1 arg2 -> too many
  # SCRIPT arg1 arg2 -> too many
  local all_output_many_args_plus_rc=$("${SCRIPT_UNDER_TEST}" "arg1" "arg2" 2>&1; echo $?)
  local exit_code_many_args=$(echo "${all_output_many_args_plus_rc}" | tail -n 1)
  local output_many_args=$(echo "${all_output_many_args_plus_rc}" | sed '$d')

  assertEquals "testErrorIncorrectArgCount (2 args): Script exit code" 1 "${exit_code_many_args}"
  assertContains "testErrorIncorrectArgCount (2 args): Usage message" \
    "${output_many_args}" "${usage_hint}"

  # Sub-case 3: --no-archive with no path
  # The script itself handles this: if [[ $# -ne 1 ]]; then usage; fi after shift
  # This is covered by "No arguments" if --no-archive is the only thing.
  # Let's test --no-archive with too many arguments
  local all_output_no_archive_many_plus_rc=$("${SCRIPT_UNDER_TEST}" --no-archive "arg1" "arg2" 2>&1; echo $?)
  local exit_code_no_archive_many=$(echo "${all_output_no_archive_many_plus_rc}" | tail -n 1)
  local output_no_archive_many=$(echo "${all_output_no_archive_many_plus_rc}" | sed '$d')

  assertEquals "testErrorIncorrectArgCount (--no-archive + 2 args): Script exit code" 1 "${exit_code_no_archive_many}"
  assertContains "testErrorIncorrectArgCount (--no-archive + 2 args): Usage message" \
    "${output_no_archive_many}" "${usage_hint}"
}

# Call shunit2 to run the tests (must be the last thing in the script)
. "${THIS_SCRIPT_PATH}/shunit2" # Source shunit2 using absolute path
