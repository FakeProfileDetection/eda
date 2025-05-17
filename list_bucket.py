#!/usr/bin/env python3
"""
list_buckets.py - List all buckets in your Google Cloud projects
---------------------------------------------------------------
Uses Application Default Credentials to list all accessible buckets.

Requirements:
    pip install google-cloud-storage
"""

import os
import sys
from google.cloud import storage

def main():
    print("=== Listing Google Cloud Storage Buckets ===")
    
    # Check if authenticated
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        cred_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if not cred_file:
            print("Note: No explicit credentials file set.")
        
    print("Attempting to authenticate...")
    
    try:
        # Try to authenticate with default credentials
        client = storage.Client()
        project = client.project
        print(f"Successfully authenticated to project: {project}")
        
        print("\nListing all accessible buckets:")
        buckets = list(client.list_buckets())
        
        if not buckets:
            print("No buckets found in this project.")
        else:
            print(f"Found {len(buckets)} buckets:")
            for bucket in buckets:
                print(f"  - {bucket.name} (project: {bucket.project_number})")
        
    except Exception as e:
        print(f"Error: {e}")
        
        print("\nTrying anonymous client to list public buckets...")
        try:
            anon_client = storage.Client.create_anonymous_client()
            # Try to access a known public bucket
            public_bucket = anon_client.bucket("gcp-public-data-landsat")
            blobs = list(public_bucket.list_blobs(max_results=1))
            print("Anonymous access successful (can access public buckets)")
            
            # If we get here, anonymous access works
            print("\nTo access your buckets, please verify:")
            print("1. You're authenticated with gcloud: run 'gcloud auth application-default login'")
            print("2. You're using the correct project ID")
            print("3. Your account has permission to list buckets in the project")
            
        except Exception as anon_e:
            print(f"Anonymous access also failed: {anon_e}")
            print("\nPlease ensure you're authenticated:")
            print("1. Install Google Cloud CLI: https://cloud.google.com/sdk/docs/install")
            print("2. Run: gcloud auth application-default login")
            print("3. Run this script again")

if __name__ == "__main__":
    main()