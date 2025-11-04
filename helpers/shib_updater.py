import os
import re


def update_shib_threshold(file_path, new_threshold):
    """
    Finds and replaces the p:failureThreshold value in the specified XML file.
    """
    try:
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return

        with open(file_path, 'r') as f:
            content = f.read()

        # Format the threshold to two decimal places, matching the example "0.60"
        formatted_threshold = f"{new_threshold:.2f}"

        # Regex to find 'p:failureThreshold="<number>"'
        # It captures the parts before and after the number so we can put them back
        pattern = r'(p:failureThreshold=")[0-9\.]*(")'
        replacement = rf'\g<1>{formatted_threshold}\g<2>'

        # re.subn returns a tuple: (new_string, number_of_subs_made)
        new_content, count = re.subn(pattern, replacement, content)

        if count == 0:
            print(f"Warning: Could not find 'p:failureThreshold' attribute in {file_path}.")
            print("File was not modified.")
        else:
            with open(file_path, 'w') as f:
                f.write(new_content)
            print(f"Successfully updated {file_path} with new threshold: {formatted_threshold}")

    except IOError as e:
        print(f"Error reading or writing to {file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while updating shibboleth file: {e}")