#!/usr/bin/env bash
# pipeline_control.conf - Configuration to control which pipeline steps to run
# Set to "true" to run a step, "false" to skip it

# Step 1: Check GCloud authentication
RUN_STEP1_AUTH=true

# Step 2: Download web app data from GCS
# Set to false if you already have the data locally
RUN_STEP2_DOWNLOAD=false

# Step 3: Map web app data to user directories
RUN_STEP3_MAP_DATA=true

# Step 4: Upload raw data to GCS
RUN_STEP4_UPLOAD_RAW=true

# Step 5: Extract TypeNet features
RUN_STEP5_EXTRACT_FEATURES=true

# Step 6: Upload TypeNet features to GCS
RUN_STEP6_UPLOAD_FEATURES=true

# Step 7: Extract ML features
RUN_STEP7_EXTRACT_ML=true

# Step 8: Upload ML features to GCS
RUN_STEP8_UPLOAD_ML=true

# Step 9: Generate summary report
RUN_STEP9_SUMMARY=true

# Additional options
CLEANUP_TEMP_FILES=false  # Set to true to remove intermediate files after upload
VERBOSE_OUTPUT=true       # Set to true for detailed output