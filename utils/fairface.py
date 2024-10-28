
def load_fairface():
    # Pip install PyDrive
    import subprocess
    import sys
    def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    install('PyDrive')
    install('oauth2client')
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDriveFile
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from google.colab import auth, drive
    from oauth2client.client import GoogleCredentials
    from pathlib import Path
    from tqdm.notebook import tqdm
    from PIL import Image
    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import zipfile

    # Authenticate and create the PyDrive client
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)

    # Specify the file ID from the link
    file_id = '1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86'

    # Download the .zip file
    downloaded = drive.CreateFile({'id': file_id})
    zip_filename = 'downloaded_file.zip'  # Name of the downloaded zip file
    downloaded.GetContentFile(zip_filename)

    # Create the extracted directory if it doesn't exist
    extract_dir = '/content/extracted'
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    # Extract the .zip file into the /content/extracted directory
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Function to download a file from Google Drive
    def download_csv(file_id, destination):
        downloaded = drive.CreateFile({'id': file_id})
        downloaded.GetContentFile(destination)
        print(f"Downloaded file saved as {destination}")

    # File IDs and destinations for Train and Validation labels
    train_file_id = '1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH'
    val_file_id = '1wOdja-ezstMEp81tX1a-EYkFebev4h7D'
    train_labels_path = '/content/train_labels.csv'
    val_labels_path = '/content/val_labels.csv'

    # Download Train and Validation label CSV files
    download_csv(train_file_id, train_labels_path)
    download_csv(val_file_id, val_labels_path)
    df_fairface_train = pd.read_csv(train_labels_path)
    df_fairface_val = pd.read_csv(val_labels_path)

    return df_fairface_train, df_fairface_val

def preprocess_fairface(df_fairface_train, df_fairface_val):
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDriveFile
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from google.colab import auth, drive
    from oauth2client.client import GoogleCredentials
    from pathlib import Path
    from tqdm.notebook import tqdm
    from PIL import Image
    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import zipfile
    from pathlib import Path
    from tqdm.notebook import tqdm  # For the progress bar (if not already installed, run: !pip install tqdm)

    tqdm.pandas()

    train_df = df_fairface_train.copy()
    val_df = df_fairface_val.copy()

    # Not sure what service_test is
    train_df.drop(columns=['service_test'],inplace=True)
    val_df.drop(columns=['service_test'],inplace=True)

    # Define the base image directory
    base_img = Path('/content/extracted')

    # Update labels in both train_df and val_df
    train_df['age'] = train_df['age'].replace('more than 70', '70+')
    val_df['age'] = val_df['age'].replace('more than 70', '70+')

    # Update the 'file' column for train and val dataframes
    train_df['file'] = train_df['file'].progress_map(lambda x: base_img / x)
    val_df['file'] = val_df['file'].progress_map(lambda x: base_img / x)

    ###### Mapping gender
    gender_map = {'Male': 0, 'Female': 1}

    train_df['gender_code'] = train_df['gender'].progress_map(lambda x:gender_map[x])
    val_df['gender_code'] = val_df['gender'].progress_map(lambda x:gender_map[x])

    ###### Mapping age
    age_order = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

    # Setting the desired order
    train_df['age'] = train_df['age'].astype('category')
    train_df['age'] = train_df['age'].cat.set_categories(age_order, ordered=True)

    # Encoding the age groups
    train_df['age_code'] = train_df['age'].cat.codes

    # Creating the age map
    age_map = dict(zip(train_df['age'].cat.categories, range(len(train_df['age'].cat.categories))))
    val_df['age_code'] = val_df['age'].progress_map(lambda x: age_map[x])

    ###### Mapping race
    train_df['race'] = train_df['race'].astype('category')
    train_df['race_code'] = train_df['race'].cat.codes

    race_map = dict(zip(train_df['race'].cat.categories,range(len(train_df['race'].cat.categories))))
    val_df['race_code'] = val_df['race'].progress_map(lambda x:race_map[x])

    # New desired mapping: White, Black, Asian, Indian, and Others
    # Create a function to map the current race categories to the target categories
    def map_race_to_target(race):
        if race in ['East Asian', 'Southeast Asian']:
            return 'Asian'
        elif race in ['Latino_Hispanic', 'Middle Eastern']:
            return 'Others'
        else:
            return race  # Retain 'White', 'Black', and 'Indian' as is

    # Define the desired category order
    desired_order = ['White', 'Black', 'Asian', 'Indian', 'Others']

    # Apply the mapping function to map the races to the target categories
    train_df['race'] = train_df['race'].map(map_race_to_target)
    val_df['race'] = val_df['race'].map(map_race_to_target)

    # Convert to categorical type and set the desired category order
    train_df['race'] = train_df['race'].astype('category')
    train_df['race'] = train_df['race'].cat.set_categories(desired_order, ordered=True)

    val_df['race'] = val_df['race'].astype('category')
    val_df['race'] = val_df['race'].cat.set_categories(desired_order, ordered=True)

    # Create 'race_code' using the ordered categories
    train_df['race_code'] = train_df['race'].cat.codes
    val_df['race_code'] = val_df['race'].cat.codes

    # Create the new race_map based on the new categories
    race_map_fair = dict(zip(range(len(desired_order)), desired_order))

    # Display the resulting mappings
    print("New Race Mapping:", race_map_fair)

    df_fairface_raw = pd.concat([train_df, val_df], ignore_index=True)

    from PIL import Image
    import numpy as np

    # Lists to store the extracted data
    images_fairface = []
    ages_fairface = df_fairface_raw['age_code'].tolist()  # List of age codes
    genders_fairface = df_fairface_raw['gender_code'].tolist()  # List of gender codes
    ethnicities_fairface = df_fairface_raw['race_code'].tolist()

    df_fairface_eda = df_fairface_raw[['age','gender','race']]
    df_fairface_code = df_fairface_raw[['file','age_code','gender_code','race_code']]

    return df_fairface_eda, df_fairface_code


def split_data(images, ages, races, genders):
  from sklearn.model_selection import train_test_split
  # Reduce sample size
  images_sample, hep_x, ages_sample, hep_y, races_sample, hep_z, genders_sample, hep_w = train_test_split(
      images, ages, races, genders, test_size=0.50, random_state=42, stratify=ages)

  # Split into train (75%) and temp (25%)
  X_train, X_temp, y_train_age, y_temp_age, y_train_races, y_temp_races, y_train_gender, y_temp_gender = train_test_split(
      images_sample, ages_sample, races_sample, genders_sample, test_size=0.25, random_state=42, stratify=ages_sample)

  # Split temp into validation (80% of temp) and test (20% of test)
  X_val, X_test, y_val_age, y_test_age, y_val_races, y_test_races, y_val_gender, y_test_gender = train_test_split(
      X_temp, y_temp_age, y_temp_races, y_temp_gender, test_size=0.20, random_state=42, stratify=y_temp_age)

  # Print output
  print(f"Training set: {X_train.shape}, Training labels: {y_train_age.shape}")
  print(f"Age validation set: {X_val.shape}, Age validation labels: {y_val_age.shape}")
  print(f"Age test set: {X_test.shape}, Age test labels: {y_test_age.shape}")
  print(f"Race validation labels: {y_val_races.shape}")
  print(f"Race test labels: {y_test_races.shape}")
  print(f"Gender validation labels: {y_val_gender.shape}")
  print(f"Gender test labels: {y_test_gender.shape}")

  return X_train, X_val, X_test, y_train_age, y_val_age, y_test_age, y_train_races, y_val_races, y_test_races, y_train_gender, y_val_gender, y_test_gender



def create_lists(df):
    !pip install PyDrive
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDriveFile
    import os
    import numpy as np
    import pandas as pd
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from google.colab import auth
    from oauth2client.client import GoogleCredentials
    from pathlib import Path
    from tqdm.notebook import tqdm
    from PIL import Image
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import seaborn as sns
    from google.colab import drive
    from sklearn.model_selection import train_test_split
    from keras.applications import mobilenet_v3
    import keras
    from keras import layers, models  
    import numpy as np
    import pandas as pd
    from PIL import Image
  # Initialize lists to store images and labels
  images = []
  ages = []
  races = []
  genders = []

  # Iterate over each row in the DataFrame
  for _, row in df.iterrows():  # Iterate over each row in the sampled dataframe
    img_path = row['file']           # Extract the file path
    age_code = row['age_code']       # Extract the age code
    race_code = row['race_code']     # Extract the race code
    gender_code = row['gender_code'] # Extract the gender code

    try:
        # Open the image
        img = Image.open(img_path)

        # Convert the image to RGB (if it's grayscale or RGBA)
        img = img.convert("RGB")

        # Resize the image to the target size (224, 224)
        img = img.resize((224, 224))

        # Convert the resized image to a numpy array
        img_array = np.array(img)

        # Ensure the image has the correct shape (224, 224, 3)
        if img_array.shape == (224, 224, 3):
            images.append(img_array)  # Append the image to the list
            ages.append(age_code)  # Append the age code to the 'ages' list
            races.append(race_code)
            genders.append(gender_code)

        else:
            print(f"Skipping image with incorrect shape: {img_path}")
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")

  # Convert lists to numpy arrays
  images = np.array(images)
  ages = np.array(ages)
  races = np.array(races)
  genders = np.array(genders)

  return images, ages, races, genders
