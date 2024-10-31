def create_lists(df):
    # Install necessary packages
    import subprocess
    import sys

    def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    install('PyDrive')
    install('oauth2client')
    
    # Import necessary libraries
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    import os
    import numpy as np
    import pandas as pd
    from google.colab import auth
    from oauth2client.client import GoogleCredentials
    from pathlib import Path
    from tqdm.notebook import tqdm
    from PIL import Image
    import cv2
    import matplotlib.pyplot as plt
    import seaborn as sns
    from google.colab import drive
    from sklearn.model_selection import train_test_split
    from keras.applications import mobilenet_v3
    import keras
    from keras import layers, models

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
                ages.append(age_code)     # Append the age code to the 'ages' list
                races.append(race_code)   # Append the race code to the 'races' list
                genders.append(gender_code) # Append the gender code to the 'genders' list
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



def evaluate_and_plot(history, accuracy_key = 'accuracy', val_accuracy_key = 'val_accuracy', loss_key = 'loss', val_loss_key = 'val_loss'):
  import matplotlib.pyplot as plt
  # Get the accuracy and loss data from the history object
  accuracy = history.history[accuracy_key]
  val_accuracy = history.history[val_accuracy_key]
  loss = history.history[loss_key]
  val_loss = history.history[val_loss_key]
  epochs = range(1, len(accuracy) + 1)

  # Find the best accuracy and loss values
  best_acc = max(accuracy)
  best_val_acc = max(val_accuracy)
  best_loss = min(loss)
  best_val_loss = min(val_loss)

  # Print the best values
  print(f"Best Training Accuracy: {best_acc:.4f}")
  print(f"Best Validation Accuracy: {best_val_acc:.4f}")
  print(f"Best Training Loss: {best_loss:.4f}")
  print(f"Best Validation Loss: {best_val_loss:.4f}")

  # Plotting accuracy
  plt.figure(figsize=(12, 5))

  plt.subplot(1, 2, 1)
  plt.plot(epochs, accuracy, label='Training Accuracy')
  plt.plot(epochs, val_accuracy, label='Validation Accuracy')
  plt.title('Training and Validation Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()

  # Plotting loss
  plt.subplot(1, 2, 2)
  plt.plot(epochs, loss, label='Training Loss')
  plt.plot(epochs, val_loss, label='Validation Loss')
  plt.title('Training and Validation Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  # Show the plots
  plt.tight_layout()
  plt.show()
