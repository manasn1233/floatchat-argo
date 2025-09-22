# scripts/run_once.py
"""
Debugging wrapper to inspect the downloader module at import time.
Runs a light check and prints module import exceptions and contents.
After we find the issue, we'll restore the functional orchestration.
"""

import importlib
import traceback
import sys

CI_MAX_YEARS = 1

ROOTS = [
    ("https://data-argo.ifremer.fr/geo/atlantic_ocean/", "atlantic"),
    ("https://data-argo.ifremer.fr/geo/indian_ocean/", "indian"),
]

def inspect_downloader_module():
    print("=== Inspecting downloader.downloader module import ===")
    try:
        mod = importlib.import_module("downloader.downloader")
        print("Imported module:", mod)
        print("Module file:", getattr(mod, "__file__", "<no __file__>"))
        names = sorted(name for name in dir(mod) if not name.startswith("_"))
        print("Public names in downloader.downloader:", names)
        # Print signature hints if available
        for key in ("download_batch","download_batch_sync","download_if_new"):
            if key in names:
                print(f"- {key} is present in module.")
            else:
                print(f"- {key} NOT FOUND in module.")
    except Exception as e:
        print("Import failed with exception:")
        traceback.print_exc()
        return False
    return True

def main():
    ok = inspect_downloader_module()
    if not ok:
        print("Downloader module failed to import — see traceback above.")
        sys.exit(2)
    else:
        print("Downloader module import appears OK. Next step would be to run crawling/downloading.")
        # Optionally, we can proceed to a minimal crawl test here, but stop for now.
        # This keeps CI short and focused on diagnosing the import issue.

if __name__ == "__main__":
    main()
