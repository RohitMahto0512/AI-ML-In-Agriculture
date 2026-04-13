import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os   # ✅ correct import

model = load_model("tomato_model.keras")

# Automatically get class names
class_names = sorted(os.listdir("dataset"))

print("Classes:", class_names)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    img = cv2.resize(frame, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    label = class_names[class_index]

    cv2.putText(frame, label, (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Plant Disease Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()