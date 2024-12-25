import cv2
import numpy as np
from tensorflow.keras.preprocessing import image    #type: ignore
import matplotlib.pyplot as plt
from datetime import datetime
from face_detect import detect
from full_face import face_points
import dlib
import imutils

a = ["face", "symmetry", "Eye", "mouth", "tilt", "verticle"]
print("[INFO] loading facial landmark predictor...")

sound_file = "beep.wav"
class_labels = ['class_0', 'class_1']
print("[INFO] initializing video capture...")
video_path = "D:/coding/python/project/nv.mp4"  # Change this to your video file path
cap = cv2.VideoCapture(video_path)

count_detection = 0
count_face = 0
count_symmetry = 0
COUNTER = 0
count_mouth = 0
count_tilt = 0
count_verticle = 0
tim = []
fig, axs = plt.subplots(6, 1, figsize=(10, 20), sharex=True)
counts = [[] for _ in range(7)]

def update_data():
    for i, count in enumerate([count_face, count_symmetry, countER, count_mouth, count_tilt, count_verticle]):
        binary_count = 1 if count > 0 else 0
        counts[i].append(binary_count)

def generate_y_data(index):
    return counts[index]

COUNTER = 0
face = detect()
detector = dlib.get_frontal_face_detector()
facepts = face_points()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tim.append(datetime.now())
        img = cv2.resize(frame, (224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predicted_class_label = class_labels[face.find_face(img_array)]
        if predicted_class_label == 'class_0':
            count_detection = 0
        else:
            count_detection = 1
        cv2.putText(frame, f'Prediction: {predicted_class_label}', (0, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        frame = imutils.resize(frame, width=1024, height=576)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Histogram equalization for better contrast

        rects = detector(gray, 0)
        frame, count_face, count_symmetry, countER, count_mouth, count_tilt, count_verticle = facepts.points(frame)
        count_symmetry = (count_symmetry and count_tilt and count_verticle)
        print(count_symmetry)

        update_data()
        for i in range(6):
            axs[i].cla()
            axs[i].plot(tim, generate_y_data(i))
            axs[i].set_ylabel(a[i])

        plt.xlabel('Frames')
        plt.tight_layout()
        plt.pause(0.01)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        result = any(variable >= 6 for variable in [count_detection, count_face, count_symmetry, countER, count_mouth, count_tilt, count_verticle])
        if key == ord("q"):
            break

except KeyboardInterrupt:
    plt.savefig('plot.png')
    print("Plot image saved.")
finally:
    cv2.destroyAllWindows()
    cap.release()
