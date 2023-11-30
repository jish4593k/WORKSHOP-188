import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from PIL import Image
import PIL.ImageOps
import os, ssl, time
import tensorflow as tf
from tensorflow import keras

if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
    ssl._create_default_https_context = ssl._create_unverified_context

# Load data
X_data = np.load('alphabet/image.npz')['arr_0']
y_data = pd.read_csv("alphabet/labels.csv")["labels"]
print(pd.Series(y_data).value_counts())

# Define classes
alphabet_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
num_classes = len(alphabet_classes)

# Split data
X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X_data, y_data, random_state=9, train_size=3500, test_size=500)
X_train_scaled_data = X_train_data / 255.0
X_test_scaled_data = X_test_data / 255.0

# Train logistic regression model
logistic_regression_model = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled_data, y_train_data)

# Evaluate model
y_pred_data = logistic_regression_model.predict(X_test_scaled_data)
accuracy_data = accuracy_score(y_test_data, y_pred_data)
print("The accuracy is: ", accuracy_data)

# Load TensorFlow/Keras model
model_path = 'mnist_cnn_model.h5'
mnist_model = keras.models.load_model(model_path)

# Capture video
cap_video = cv2.VideoCapture(0)

# GUI initialization
cv2.namedWindow("Handwriting Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Handwriting Recognition", 800, 600)

while True:
    try:
        ret_video, frame_video = cap_video.read()

        gray_video = cv2.cvtColor(frame_video, cv2.COLOR_BGR2GRAY)

        height_video, width_video = gray_video.shape
        upper_left_video = (int(width_video / 2 - 56), int(height_video / 2 - 56))
        bottom_right_video = (int(width_video / 2 + 56), int(height_video / 2 + 56))
        cv2.rectangle(gray_video, upper_left_video, bottom_right_video, (0, 255, 0), 2)

        roi_video = gray_video[upper_left_video[1]:bottom_right_video[1],
                     upper_left_video[0]:bottom_right_video[0]]

        im_pil_video = Image.fromarray(roi_video)
        image_bw_video = im_pil_video.convert('L')
        image_bw_resized_video = image_bw_video.resize((28, 28), Image.ANTIALIAS)

        image_bw_resized_inverted_video = PIL.ImageOps.invert(image_bw_resized_video)
        pixel_filter_video = 20
        min_pixel_video = np.percentile(image_bw_resized_inverted_video, pixel_filter_video)
        image_bw_resized_inverted_scaled_video = np.clip(image_bw_resized_inverted_video - min_pixel_video, 0, 255)
        max_pixel_video = np.max(image_bw_resized_inverted_video)
        image_bw_resized_inverted_scaled_video = np.asarray(
            image_bw_resized_inverted_scaled_video) / max_pixel_video
        test_sample_video = np.array(image_bw_resized_inverted_scaled_video).reshape(1, 784)

        # Predict using TensorFlow/Keras model
        prediction = mnist_model.predict_classes(test_sample_video)

        # Predict using logistic regression model
        logistic_regression_prediction = logistic_regression_model.predict(test_sample_video)

        print("TensorFlow/Keras Prediction: ", prediction)
        print("Logistic Regression Prediction: ", logistic_regression_prediction)

        # Display prediction on the GUI
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(gray_video, f'TensorFlow/Keras: {prediction[0]}', (10, 50), font, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(gray_video, f'Logistic Regression: {logistic_regression_prediction[0]}', (10, 100), font, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Handwriting Recognition', gray_video)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass

cap_video.release()
cv2.destroyAllWindows()
