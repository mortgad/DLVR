def download_and_extract(extract_dir):
    import os
    import tarfile
    import urllib.request

    # Define URLs within the function
    urls = [
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar",
        "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_meta.tar"
    ]

    # Create the destination directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)

    # Loop over each URL, download, and extract
    for url in urls:
        # File name based on URL
        file_name = url.split('/')[-1]
        file_path = os.path.join(extract_dir, file_name)

        # Download the file
        print(f"Downloading {file_name}...")
        urllib.request.urlretrieve(url, file_path)
        print(f"Downloaded {file_name}.")

        # Extract the tar file
        print(f"Extracting {file_name}...")
        with tarfile.open(file_path, "r") as tar:
            tar.extractall(extract_dir)
        print(f"Extracted {file_name}.")

        # Remove the tar file after extraction to save space
        os.remove(file_path)
        print(f"Removed {file_name}.")

    print("All datasets downloaded and extracted.")

def create_dataframe(extract_dir):
    from scipy.io import loadmat
    import os
    import datetime
    import numpy as np
    
    # Locate the .mat file and load it
    mat_file_path = os.path.join(extract_dir, 'imdb/imdb.mat')
    data = loadmat(mat_file_path)
    metadata = data['imdb'][0, 0]

    def matlab_datenum_to_year(matlab_datenum):
        """Convert MATLAB datenum to year, with error handling for extreme values."""
        try:
            python_base_date = datetime.datetime(1, 1, 1)
            days_offset = int(matlab_datenum) - 366  # Convert to integer
            converted_date = python_base_date + datetime.timedelta(days=days_offset)
            return converted_date.year
        except OverflowError:
            # print(f"OverflowError: Unable to convert MATLAB datenum {matlab_datenum}.")
            return None

    dob = metadata['dob'][0]
    dob_years = [matlab_datenum_to_year(date[0] if isinstance(date, np.ndarray) else date) for date in dob]
    photo_taken = metadata['photo_taken'][0]
    full_paths = metadata['full_path'][0]
    gender = metadata['gender'][0]
    face_score = metadata['face_score'][0]
    second_face_score = metadata['second_face_score'][0]

    # Ensure dob_years has np.nan for invalid entries
    dob_years = np.array([year if year is not None else np.nan for year in dob_years], dtype=float)

    # Calculate age while keeping alignment
    ages = np.array([photo - dob if dob is not None else np.nan for dob, photo in zip(dob_years, photo_taken)])

    # Filter valid indices based on face_score, second_face_score, and valid dob_years
    valid_indices = np.where((face_score > 1.0) & (np.isnan(second_face_score)) & (~np.isnan(dob_years)))[0]

    # Apply the filter to the arrays
    filtered_paths = full_paths[valid_indices]
    filtered_ages = ages[valid_indices]
    filtered_gender = gender[valid_indices]
    filtered_dob = dob_years[valid_indices]
    filtered_photo_taken = photo_taken[valid_indices]

    # Convert paths to strings and create a full path for each image
    base_image_path = "/content/datasets/imdb_crop/"
    image_paths = [os.path.join(base_image_path, path[0]) for path in filtered_paths]

    # Set minimum dimensions for ResNet input and initialize filtered lists
    min_width, min_height = 224, 224
    valid_image_paths = []
    valid_ages = []
    valid_gender = []
    valid_dob = []
    valid_photo_taken = []

    # Check each image for dimensions and RGB mode
    from PIL import Image
    for i, path in enumerate(image_paths):
        try:
            with Image.open(path) as img:
                # Check if image meets the minimum size and has RGB channels
                if img.width >= min_width and img.height >= min_height and img.mode == 'RGB':
                    valid_image_paths.append(path)
                    valid_ages.append(filtered_ages[i])
                    valid_gender.append(filtered_gender[i])
                    valid_dob.append(filtered_dob[i])
                    valid_photo_taken.append(filtered_photo_taken[i])
        except Exception as e:
            print(f"Error opening image {path}: {e}")
            continue  # Skip image if there's an error

    # Create final DataFrame with filtered images and metadata
    import pandas as pd
    df = pd.DataFrame({
        "image_path": valid_image_paths,
        "age": valid_ages,
        "gender": valid_gender,
        "dob": valid_dob,
        "photo_taken": valid_photo_taken
    })

    # Display the first 10 rows of the DataFrame
    df.head(10).style.set_properties(**{'text-align': 'center'})

    return df