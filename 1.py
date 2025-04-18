import numpy as np
import os
import cv2
import pickle
import random
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

# 🔹 Dataset Directory
DATADIR = "train"
CATEGORIES = sorted(os.listdir(DATADIR))
IMG_SIZE = 50  

# ✅ Allow only image formats
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def is_image_file(filename):
    """Returns True if the file is a valid image."""
    return any(filename.lower().endswith(ext) for ext in VALID_EXTENSIONS)

def create_training_data():
    """Reads only valid images, applies preprocessing, and stores them in training_data."""
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        
        for img in os.listdir(path):
            if not is_image_file(img):
                print(f"⚠️ Skipping non-image file: {img}")
                continue  # Skip non-image files

            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img_array is None:
                    print(f"⚠️ Skipping unreadable image: {img_path}")
                    continue

                # ✅ Image Processing
                clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(8,8))
                img_array = clahe.apply(img_array)

                median = cv2.medianBlur(img_array.astype('uint8'), 5)
                median = 255 - median

                _, thresh = cv2.threshold(median, 165, 255, cv2.THRESH_BINARY_INV)
                new_array = cv2.resize(thresh, (IMG_SIZE, IMG_SIZE))

                training_data.append([new_array, class_num])

            except Exception as e:
                print(f"⚠️ Error processing {img}: {e}")
                continue

    return training_data

# 🔹 Run Preprocessing
training_data = create_training_data()
random.shuffle(training_data)

# 🔹 Convert to NumPy Arrays
X, y = zip(*training_data)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y).astype(int).reshape(-1,)

# ✅ Print Final Dataset Shape
print(f"✅ Final Dataset Shape: X={X.shape}, y={y.shape}")
print("Unique Labels:", np.unique(y))

# 🔹 Save Data
with open("X.pickle", "wb") as pickle_out:
    pickle.dump(X, pickle_out)

with open("y.pickle", "wb") as pickle_out:
    pickle.dump(y, pickle_out)

print("✅ X.pickle and y.pickle files created successfully!")
