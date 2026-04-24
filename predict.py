import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model
model = load_model("brain_tumor_model.h5")   # or .keras

# Class labels (IMPORTANT - must match training folders)
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load image
img_path = "/Users/krititripathi/Desktop/brain-mri-project/dataset/Testing/glioma/Te-gl_1.jpg" # change image path
img = cv2.imread(img_path)

if img is None:
    print("Error: Image not found. Check path!")
    exit()

# Preprocess
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.reshape(img, (1, 224, 224, 3))

# Predict
prediction = model.predict(img)
predicted_class = classes[np.argmax(prediction)]

print("Prediction:", predicted_class)
print("Confidence:", np.max(prediction))