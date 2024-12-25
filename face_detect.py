from tensorflow.keras.models import load_model #type: ignore
import numpy as np

class detect:
    def __init__(self):
        self.model = load_model('densenet201_binary_classification_softmax_final.h5')
    def find_face(self,img_array):
        pred=self.model.predict(img_array)
        return np.argmax(pred)