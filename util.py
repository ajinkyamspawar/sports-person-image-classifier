import joblib
import json

import cv2
import numpy as np
import pywt

model = None
class_dict = None


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
def load_saved_artifacts():

    global model
    global class_dict

    print("Loading saved artifacts...")

    model = joblib.load("saved_model.pkl")

    with open("class_dictionary.json", "r") as f:
        class_dict = json.load(f)

    print("Artifacts loaded successfully")

def w2d(image, mode='haar', level=1):

    imArray = image

    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)

    imArray = np.float32(imArray)
    imArray /= 255

    coeffs = pywt.wavedec2(imArray, mode, level=level)

    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    imArray_H = pywt.waverec2(coeffs_H, mode)

    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H

def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) >= 2:
            return roi_color

def get_class_name(class_num):

    for name, number in class_dict.items():
        if number == class_num:
            return name

    return None


def classify_image(image_path):

    cropped_img = get_cropped_image_if_2_eyes(image_path)

    if cropped_img is None:
        return None

    scaled_raw_img = cv2.resize(cropped_img, (32, 32))

    img_har = w2d(cropped_img, 'db1', 5)

    scaled_img_har = cv2.resize(img_har, (32, 32))

    combined_img = np.vstack(
        (
            scaled_raw_img.reshape(32*32*3, 1),
            scaled_img_har.reshape(32*32, 1)
        )
    )

    X = combined_img.reshape(1, 4096).astype(float)

    prediction = model.predict(X)

    predicted_name = get_class_name(prediction[0])

    display_name = predicted_name.replace("_", " ").title()

    probabilities = model.predict_proba(X)

    
    confidence = round(
    np.max(probabilities) * 100,
    2
    )

    return {
    "player": display_name,
    "confidence": confidence
    }
