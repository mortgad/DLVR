def load_utkface():
    import tarfile
    import subprocess
    import sys
    def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    install('PyDrive')
    install('oauth2client')
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from google.colab import auth
    from oauth2client.client import GoogleCredentials

    # Authenticate PyDrive
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)

    # Folder ID from the shared Google Drive folder
    folder_id = '1HROmgviy4jUUUaCdvvrQ8PcqtNg2jn3G'
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

    # Loop through the files and download each .tar.gz file
    for file in file_list:
        if file['title'].endswith('.tar.gz'):
            print(f'Downloading {file["title"]}')
            file.GetContentFile(file['title'])

            # Extract the .tar.gz file
            tar = tarfile.open(file['title'])
            tar.extractall('/content/extracted')  # Extract all files to /content/extracted directory
            tar.close()

            print(f'Extracted {file["title"]}')

def preprocess_utkface():
    import os
    import cv2
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import re

    def delete_non_image_files(directories, allowed_extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        for directory in directories:
            for filename in os.listdir(directory):
                if not filename.lower().endswith(allowed_extensions):
                    file_path = os.path.join(directory, filename)
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

    def delete_invalid_images(directories):
        # Regular expression pattern to match the desired format
        pattern = re.compile(r'^\d+_\d+_\d+_\d+\.\w+$')

        for directory in directories:
            for filename in os.listdir(directory):
                # Check if the filename matches the desired pattern
                if not pattern.match(filename):
                    file_path = os.path.join(directory, filename)
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

    # Specify the directories
    directories = [
        '/content/extracted/part1',
        '/content/extracted/part2',
        '/content/extracted/part3'
    ]

    # Run the function to delete non-image files
    delete_non_image_files(directories)

    # Run the function to delete invalid images
    delete_invalid_images(directories)

    # Initialize lists to store images and labels
    images = []
    ages_utkface = []
    genders_utkface = []
    ethnicities_utkface = []
    files_utkface = []  # New list to store file paths

    # Process each directory
    for directory in directories:
        print(f"Processing from: {directory}")
        files = os.listdir(directory)

        for file in files:
            file_path = os.path.join(directory, file)  # Create the full file path
            
            # Load the image
            image = cv2.imread(file_path) 

            # Check if the image was loaded successfully
            if image is not None:  
                
                # Parse filename to extract labels
                split_var = file.split('_')
                ages_utkface.append(int(split_var[0]))
                genders_utkface.append(int(split_var[1]))
                ethnicities_utkface.append(int(split_var[2]))
                files_utkface.append(file_path)  # Store the file path in the list
            else:
                print(f"Failed to load image: {file_path}")

    # Create a dataframe from the lists
    df_utkface = pd.DataFrame({
        'age': ages_utkface,
        'gender': genders_utkface,
        'ethnicity': ethnicities_utkface,
        'file': files_utkface  # Add the file paths as a column
    })

    # Create a DataFrame from the lists
    df_utkface_raw = pd.DataFrame({
        'file': files_utkface,
        'age_raw': ages_utkface,
        'gender_code': genders_utkface,
        'race_code': ethnicities_utkface
    })

    # Return the DataFrame
    return df_utkface, df_utkface_raw
