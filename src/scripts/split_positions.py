import os
import re
import shutil


# Takes in a directory (hierarchy pos 1) of directories (hierarchy pos 2)
def process_directory(directory, output_directory=None):
    if output_directory is None:
        output_directory = directory
    # Create a new directory for position splits at hierarchy pos 1
    new_directory = os.path.join(
        output_directory, f"{os.path.basename(directory)}-position_splits"
    )
    os.makedirs(new_directory, exist_ok=True)

    # Iterate through all subdirectories at hierarchy pos 2
    for subdir in os.scandir(directory):
        if subdir.is_dir() and not subdir.name.endswith("-position_splits"):
            # Iterate through timepoints of each hierarchy pos 2 directory
            process_subdir(subdir, new_directory)


def process_subdir(subdir, new_directory):
    timepoint_pattern = re.compile(
        r"timepoint(\d+)_position(\d+)_(\d+)_burst(\d+)"
    )
    timepoint_directories = {}

    # For each of the three folders
    for folder in ["data", "metadata", "preview"]:
        # Get the path of the current folder
        folder_path = os.path.join(subdir.path, folder)

        if os.path.isdir(folder_path):
            # Iterate through each file in the current folder
            for file in os.listdir(folder_path):
                # Get the file's timepoint
                match = timepoint_pattern.match(file)
                if match:
                    timepoint = match.group(1)

                    # Create the directory if it hasn't been created yet
                    if timepoint not in timepoint_directories:
                        dst_dir = os.path.join(
                            new_directory, f"{subdir.name}-{timepoint}"
                        )
                        os.makedirs(
                            os.path.join(dst_dir, "data"), exist_ok=True
                        )
                        os.makedirs(
                            os.path.join(dst_dir, "metadata"), exist_ok=True
                        )
                        os.makedirs(
                            os.path.join(dst_dir, "preview"), exist_ok=True
                        )
                        timepoint_directories[timepoint] = dst_dir

                    # Copy the file to the destination directory
                    shutil.copy2(
                        os.path.join(folder_path, file),
                        os.path.join(
                            timepoint_directories[timepoint], folder, file
                        ),
                    )
