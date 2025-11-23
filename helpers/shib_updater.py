import os
import re
import shutil

from helpers.globals import cfg


def update_shib_threshold(file_path, new_threshold):
    """
    Updates the p:failureThreshold value inside the Shibboleth rba-beans.xml file.

    New functionality:
      - Optional backup creation before writing.
      - Backup suffix is configurable.
      - Atomic write: writes to a temp file, then replaces the original.
    """
    try:
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return

        backup_enabled = cfg("deployment.backup_before_update", True)
        backup_suffix = cfg("deployment.backup_suffix", ".bak")

        # Read original file
        with open(file_path, "r") as f:
            content = f.read()

        formatted_threshold = f"{new_threshold:.2f}"

        pattern = r'(p:failureThreshold=")[0-9\.]*(")'
        replacement = rf'\g<1>{formatted_threshold}\g<2>'

        new_content, count = re.subn(pattern, replacement, content)

        if count == 0:
            print(f"Warning: No 'p:failureThreshold' attribute found in {file_path}. No update performed.")
            return

        # Create backup if enabled
        if backup_enabled:
            backup_path = f"{file_path}{backup_suffix}"
            try:
                shutil.copy2(file_path, backup_path)
                print(f"Backup created: {backup_path}")
            except Exception as e:
                print(f"Warning: Could not create backup file: {e}")

        # Write atomically by using a temporary file
        temp_path = f"{file_path}.tmp"

        with open(temp_path, "w") as f:
            f.write(new_content)

        # Replace original with updated file
        os.replace(temp_path, file_path)

        print(f"Successfully updated Shibboleth threshold to {formatted_threshold} in {file_path}")

    except Exception as e:
        print(f"Unexpected error while updating Shibboleth file: {e}")
