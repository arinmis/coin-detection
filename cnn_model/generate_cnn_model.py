import numpy as np
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
from keras.models import load_model

# Acquire RGB Coin Image
def acquire_coin_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    return image

# Convert RGB Coin Image to Grayscale
def convert_to_grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

# Crop and Trim the Image
def crop_and_trim_image(image):
    height, width = image.shape[:2]

    crop_height = min(height, 200)
    crop_width = min(width, 200)

    start_row = (height - crop_height) // 2
    end_row = start_row + crop_height
    start_col = (width - crop_width) // 2
    end_col = start_col + crop_width

    cropped_image = image[start_row:end_row, start_col:end_col]

    return cropped_image

def crop_circle(image):
    gray = cv2.medianBlur(image, 5)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            x1 = x - r
            y1 = y - r
            x2 = x + r
            y2 = y + r
            cropped_image = image[y1:y2, x1:x2]

            return cropped_image

    return image

def generate_feature_vector(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

def prepare_images():
    classes = ['5kr', '10kr', '25kr', '50kr', '1tl']

    current_directory = os.getcwd()
    image_dir = current_directory + '\cnn_model\data'

    images = []
    labels = []

    for class_index, class_name in enumerate(classes):
        class_dir = os.path.join(image_dir, class_name)
        for filename in os.listdir(class_dir):
            image_path = os.path.join(class_dir, filename)
            img = acquire_coin_image(image_path)
            gray = convert_to_grayscale(img)
            resized_img = cv2.resize(gray, (416, 416))
            images.append(resized_img)
            labels.append(class_index)
    images = np.array(images)
    labels = np.array(labels)
    return (images, labels, classes)


def train():
    (images,labels, classes) = prepare_images()
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Etiketleri one-hot kodlamasıyla kodlama
    num_classes = len(classes)
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    # Build CNN model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(416, 416, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train CNN model
    model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

    # Test CNN model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    # Save model
    model.save("coin_detection_model.h5")
    return (model)

def predict(img, model):
    new_gray = convert_to_grayscale(img)
    new_resized = cv2.resize(new_gray, (416, 416))
    new_input = np.expand_dims(new_resized, axis=0)
    new_input = np.expand_dims(new_input, axis=-1)
    cnn_model = model

    (images, labels, classes) = prepare_images()
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(y_train)
    label_encoder.transform(y_test)

    # Predict the label
    predictions = cnn_model.predict(new_input)
    predicted_class_index = np.argmax(predictions)
    predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
    if predicted_class == 0:
        return ('5kr', 0.05)
    elif predicted_class == 1:
        return ('10kr', 0.1)
    elif predicted_class == 2:
        return ('25kr', 0.25)
    elif predicted_class == 3:
        return ('50kr', 0.5)
    else:
        return ('1tl', 1.0)
