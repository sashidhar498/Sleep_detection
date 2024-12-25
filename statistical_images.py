from imutils.video import VideoStream
import imutils
import dlib
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image #type: ignore
# import matplotlib.pyplot as plt
from datetime import datetime
from face_detect import detect
from full_face2 import face_points
import keyboard
import psutil
import os
import time

def memory_usage():
    # Get the process ID (PID) of the current Python script
    process = psutil.Process(os.getpid())
    # Return the memory usage in MB
    return process.memory_info().rss / (1024 * 1024)
print("memory usage before:",memory_usage())
a=["face","symmetry","Eye","mouth","tilt","verticle"]
# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")


sound_file = "beep.wav"  # Replace with the actual path to your sound file
class_labels = ['class_0', 'class_1']
# initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up
print("[INFO] initializing camera...")
# vs = VideoStream(src=0).start()
count_detection=0
count_face=0
count_symmetry=0
COUNTER=0
count_mouth=0
count_tilt=0
count_verticle=0
tim=[]
# Create subplots
# fig, axs = plt.subplots(6, 1, figsize=(10, 20), sharex=True)

# Initialize lists for counts
counts = [[] for _ in range(7)]

# Function to update counts
# def update_data():
#     for i, count in enumerate([count_face, count_symmetry, countER, count_mouth, count_tilt, count_verticle]):
#         binary_count = 1 if count > 0 else 0
#         counts[i].append(binary_count)



# Function to generate y-axis data
def generate_y_data(index):
    return counts[index]

# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
#time.sleep(2.0)

# loop over the frames from the video stream
# 2D image points. If you change the image, you need to change vector

COUNTER = 0
face=detect()
# grab the indexes of the facial landmarks for the mouth
detector = dlib.get_frontal_face_detector()
facepts=face_points()
folder="images"
try:
    for i in os.listdir(folder):
        # a=time.time()
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale  
        # frame = vs.read()
        frame=cv2.imread(os.path.join(folder,i))
        tim.append(datetime.now())
        img = cv2.resize(frame, (224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make predictions

        # Decode the class index to class label if you have class labels
        # Replace classes=['class_0', 'class_1'] with your actual class labels
        
        predicted_class_label = class_labels[face.find_face(img_array)]
        if predicted_class_label=='class_0':
            count_detection=0
        else:
            count_detection=1

        # Display the frame with prediction
        cv2.putText(frame, f'Prediction: {predicted_class_label}', (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        frame = imutils.resize(frame, width=1024, height=576)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape

        # loop over the face detections
        frame,count_face,count_symmetry,countER,count_mouth,count_tilt,count_verticle=facepts.points(frame)
        count_symmetry=(count_symmetry and count_tilt and count_verticle)
        print(count_symmetry)

        # Update data for plotting
        # update_data()

        # # Plotting
        # for i in range(6):
        #     axs[i].cla()  # Clear previous plot
        #     axs[i].plot(tim, generate_y_data(i))
        #     axs[i].set_ylabel(a[i])

        # plt.xlabel('Frames')
        # plt.tight_layout()
        # plt.pause(0.01)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        result = any(variable >= 6 for variable in [count_detection, count_face, count_symmetry, countER, count_mouth, count_tilt, count_verticle])
        if key == ord("q"):
            break
        # b=time.time()
        time.sleep(5)
        # keyboard.wait('space')
    
    # print("memory usage after:",memory_usage(),"\ntime taken:",b-a)

except KeyboardInterrupt:
    # Save the plotted image
    # plt.savefig('plot.png')
    print("Plot image saved.")
finally:
    # do a bit of cleanup
    cv2.destroyAllWindows()
    # vs.stop()