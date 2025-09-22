# scripts/run_schema.py
"""
Simple script to verify DATABASE_URL connectivity and run db/schema.sql once.
Usage:
  - Locally: set DATABASE_URL env var (or create a .env file) then run:
      python scripts/run_schema.py
  - In CI: set DATABASE_URL as a secret and run the script (example workflow step).
"""

import os
import sys
from db import db_helpers

def main():
    # db_helpers will raise if DATABASE_URL is not set
    try:
        print("Using DATABASE_URL from environment.")
        # show a short sanity check (do not print credentials)
        db_url = os.getenv("DATABASE_URL")
        print("DATABASE_URL is set." if db_url else "DATABASE_URL not set.")
    except Exception as e:
        print("Error reading DATABASE_URL:", e)
        sys.exit(2)

    # Path to schema file
    schema_path = os.path.join(os.path.dirname(__file__), "..", "db", "schema.sql")
    schema_path = os.path.abspath(schema_path)
    if not os.path.exists(schema_path):
        print("Schema file not found at", schema_path)
        sys.exit(2)

    try:
        print("Running schema SQL (this will create extensions/tables if allowed)...")
        db_helpers.run_schema(schema_path)
        print("Schema applied successfully.")
    except Exception as e:
        print("Failed to run schema:", e)
        sys.exit(3)

if __name__ == "__main__":
    main()
