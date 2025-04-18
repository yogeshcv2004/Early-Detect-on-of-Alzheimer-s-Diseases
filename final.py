import numpy as np
import cv2
import tensorflow as tf
from tkinter import *
from tkinter import filedialog
import PIL.Image
import PIL.ImageTk
import os

# -------------------------
# Load Model and Categories
# -------------------------
try:
    model = tf.keras.models.load_model("model.h5")
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    exit()

DATADIR = "train"
if not os.path.exists(DATADIR):
    print("Error: Training data directory not found!")
    exit()

CATEGORIES = os.listdir(DATADIR)
print("Categories:", CATEGORIES)
NUM_CLASSES = len(CATEGORIES)
print("Number of categories:", NUM_CLASSES)

# -------------------------
# Preprocess Function
# -------------------------
def prepare(file):
    IMG_SIZE = 50
    try:
        img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img_array is None:
            raise ValueError("Error reading image.")

        clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(8, 8))
        img_array = clahe.apply(img_array)

        median = cv2.medianBlur(img_array.astype('uint8'), 5)
        median = 255 - median
        _, thresh = cv2.threshold(median.astype('uint8'), 165, 255, cv2.THRESH_BINARY_INV)

        new_array = cv2.resize(thresh, (IMG_SIZE, IMG_SIZE))
        processed = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
        return processed
    except Exception as e:
        print("Preprocessing Error:", e)
        return None

# -------------------------
# Prediction Function
# -------------------------
def detect(filename):
    processed_image = prepare(filename)
    if processed_image is None:
        value.set("Error: Image preprocessing failed.")
        return

    try:
        prediction = model.predict(processed_image)
        print("Raw model output:", prediction)

        if len(prediction[0]) != NUM_CLASSES:
            value.set("Error: Model output size mismatch.")
            print(f"Expected {NUM_CLASSES}, but got {len(prediction[0])}")
            return

        pred_list = list(prediction[0])
        result = CATEGORIES[pred_list.index(max(pred_list))]
        value.set("Prediction: " + result)
        print("Prediction:", result)
    except Exception as e:
        value.set("Error: Prediction failed.")
        print("Prediction Error:", e)

# -------------------------
# File Selection Handler
# -------------------------
def ClickAction():
    filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if filename:
        img = PIL.Image.open(filename).resize((250, 250))
        img = PIL.ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img  
        detect(filename)

# -------------------------
# GUI Setup with Tkinter
# -------------------------
root = Tk()
root.title("Alzheimer's Detection")
root.state('zoomed')

# Load and set the background image
try:
    bg_image = PIL.Image.open("background.png")
    bg_photo = PIL.ImageTk.PhotoImage(bg_image)
    bg_label = Label(root, image=bg_photo)
    bg_label.place(relwidth=1, relheight=1)
    bg_label.lower()
except Exception as e:
    print("Background image not found:", e)

value = StringVar()
value.set("Prediction will appear here.")

header_label = Label(root, text="Alzheimer's Detection App", font=("Arial", 28, "bold"))
header_label.pack(pady=20)

select_button = Button(root, text="SELECT FILE", font=("Arial", 18), command=ClickAction, bg="#4CAF50", fg="white", padx=20, pady=10)
select_button.pack(pady=10)

panel = Label(root)
panel.pack(pady=20)

result_label = Label(root, textvariable=value, font=("Arial", 20))
result_label.pack(pady=20)

root.mainloop()



# import tensorflow as tf

# model = tf.keras.models.load_model("model.h5")
# print(model.output_shape)
