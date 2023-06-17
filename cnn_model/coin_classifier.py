from tensorflow.keras.models import load_model

# pass image to predict, read with cv2
'''
import cv2
img = cv2.imread('coins.jpg')
'''
def predict(img):
    model = load_model('coin_classifier.h5')
    resize = tf.image.resize(img, (256,256))
    result = model.predict(np.expand_dims(resize/255, 0))
    print("predicted result: ", result)
    return result
