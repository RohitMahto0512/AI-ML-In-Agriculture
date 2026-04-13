import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os   # ✅ IMPORTANT

model = load_model("tomato_model.keras")

# Automatically get class names
class_names = sorted(os.listdir("dataset"))

print("Classes:", class_names)

img = cv2.imread("test.jpg")

if img is None:
    print("Image not found ❌")
    exit()

img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)[0]

for i, prob in enumerate(prediction):
    print(f"{class_names[i]}: {prob:.2f}")

class_index = np.argmax(prediction)
confidence = prediction[class_index]

print("\nFinal Prediction:", class_names[class_index])
print("Confidence:", confidence)
