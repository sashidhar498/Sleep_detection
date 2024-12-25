import imutils
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import math
import numpy as np
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio

class face_points:
    def __init__(self):
        self.EYE_AR_THRESH = 0.25
        self.MOUTH_AR_THRESH = 0.75
        self.EYE_AR_CONSEC_FRAMES = 3
        (self.mStart, self.mEnd) = (49, 68)
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.count_face=0
        self.count_symmetry=0
        self.countER=0
        self.eye_res=0
        self.count_mouth=0
        self.count_tilt=0
        self.count_verticle=0
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.detector = dlib.get_frontal_face_detector()
        self.symmetry_check = True
        pass
    def points(self,frame):
        
        frame = imutils.resize(frame, width=1024, height=576)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape
        rects = self.detector(gray, 0)
        if len(rects) > 0:
            text = "{} face(s) found".format(len(rects))
            # cv2.putText(frame, text, (10, 20),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            self.count_face=1
        else:
            self.count_face=0
        for rect in rects:
            # compute the bounding box of the face and draw it on the
            # frame
            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Check face symmetry
            left_eye_pts = shape[self.lStart:self.lEnd]
            right_eye_pts = shape[self.rStart:self.rEnd]

            left_eye_center = np.mean(left_eye_pts, axis=0).astype(int)
            right_eye_center = np.mean(right_eye_pts, axis=0).astype(int)

            # Draw lines connecting the corresponding landmarks
            for i in range(len(left_eye_pts)):
                cv2.line(frame, tuple(left_eye_pts[i]), tuple(right_eye_pts[i]), (0, 255, 0), 1)

            # Calculate the distance between the eye centers
            eye_distance = dist.euclidean(left_eye_center, right_eye_center)

            # Draw a line connecting the eye centers
            cv2.line(frame, tuple(left_eye_center), tuple(right_eye_center), (0, 255, 0), 1)

            # Check symmetry by comparing the distances between corresponding landmarks
           

            # Extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)



            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame self.counter
            if ear < self.EYE_AR_THRESH:
                self.countER+= 1
                # if the eyes were closed for a sufficient number of times
                # then show the warning
                if self.countER >= self.EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "Eyes: Closed ", (0, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.eye_res=0
                # otherwise, the eye aspect ratio is not below the blink
                # threshold, so reset the self.counter and alarm

            else:
                cv2.putText(frame, "Eyes: Opened", (0, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self.countER = 0
                self.eye_res=1

            mouth = shape[self.mStart:self.mEnd]

            mouthMAR = mouth_aspect_ratio(mouth)
            mar = mouthMAR
            # compute the convex hull for the mouth, then
            # visualize the mouth
            mouthHull = cv2.convexHull(mouth)

            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            cv2.putText(frame, "Mouth: Closed", (0, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw text if mouth is open
            if mar > self.MOUTH_AR_THRESH:
                cv2.putText(frame, "Mouth: Opened", (0, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.count_mouth=0
            else:
                self.count_mouth=1
            

            # Calculate the head tilt angle
            nose_tip = shape[33]
            chin = shape[8]
            # Calculate the angle between the line connecting nose tip and chin and a horizontal line
            dy = chin[1] - nose_tip[1]
            dx = chin[0] - nose_tip[0]
            head_tilt_angle = np.degrees(np.arctan2(dy, dx))
            # Display the head tilt angle
            if (head_tilt_angle<=74.0) or (head_tilt_angle>=110.0):
                self.count_tilt=0
                cv2.putText(frame, "Head: Tilted", (0, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.count_tilt=1
                cv2.putText(frame, "Head: Straight", (0, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Calculate the vertical position of eyes and mouth in relation to the nose
            nose_y = nose_tip[1]
            eyes_and_mouth_y = (left_eye_center[1] + right_eye_center[1] + shape[self.mStart][1] + shape[self.mEnd - 1][1]) / 4

            # Calculate the vertical movement
            vertical_movement = eyes_and_mouth_y - nose_y

            # Display the vertical movement
            
            if (vertical_movement>=-12) or (vertical_movement<=-35):
                self.count_verticle=0
                cv2.putText(frame, "Vertical Movement: Yes", (0, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.count_verticle=1
                cv2.putText(frame, f"Vertical Movement: no", (0, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.symmetry_check=True
            for i in range(len(left_eye_pts)):
                left_eye_to_center = dist.euclidean(left_eye_center, tuple(left_eye_pts[i]))
                right_eye_to_center = dist.euclidean(right_eye_center, tuple(right_eye_pts[i]))

                # If the distances are not approximately equal, set self.symmetry_check to False
                if not math.isclose(left_eye_to_center, right_eye_to_center, abs_tol=4):
                    self.symmetry_check = False


            # Display the symmetry result
            if self.symmetry_check and self.count_verticle and self.count_tilt:
                cv2.putText(frame, "Symmetrical Face", (0, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self.count_symmetry=1
            else:
                cv2.putText(frame, "Asymmetrical Face", (0, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.count_symmetry=0

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw each of them
            for (i, (x, y)) in enumerate(shape):
                if i in [33, 8, 36, 45, 48, 54]:
                    # something to our key landmarks
                    # write on frame in Green
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                else:
                    # everything to all other landmarks
                    # write on frame in Red
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        return frame,self.count_face,self.count_symmetry,self.eye_res,self.count_mouth,self.count_tilt,self.count_verticle