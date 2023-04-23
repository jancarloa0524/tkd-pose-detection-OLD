# Capture & Render
import mediapipe as mp # holistic solutions
import cv2 # rendering
# Export landmarks to CSV
import csv # for working with CSV file
import os # for working with files
import numpy as np # works with array mathematics
# Making Detections
import pandas as pd # working with tabular data
import pickle # library for saving and opening models on disks

# open model for making detections
with open('tkd.pkl', 'rb') as f:
                model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils # drawing utilities
mp_pose = mp.solutions.pose # holistic solutions

cap = cv2.VideoCapture(0) # webcam capture


# Display & Render Holistic Model
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Holistic Detections
        results = pose.process(img)

        # Color back
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Pose Detections
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius = 4), mp_drawing.DrawingSpec(color=(0,0,0), thickness = 2, circle_radius = 2))
        
        # Export Testing Coords to CSV, OR run hollistics with detection algorithm
        try:
            class_name = "Jab"
            # Extract pose landmarks
            pose_results = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_results]).flatten())
            # Append class name for exporting
            # pose_row.insert(0, class_name)
            # # Export to CSV
            # with open('coords.csv', mode='a', newline='') as f:
            #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #     csv_writer.writerow(pose_row)

            # Making Detections
            X = pd.DataFrame([pose_row]) 
            tkd_class = model.predict(X)[0] # predict and extract first X values
            tkd_class_prob = model.predict_proba(X)[0]
            print(tkd_class, tkd_class_prob)

            # Get status box
            cv2.rectangle(img, (0,0), (250, 60), (0,0,0), -1)
            # Display Class
            cv2.putText(img, 'POSITION'
                        , (100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img, tkd_class.split(' ')[0]
                        , (110, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(img, 'PROB'
                        , (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img, str(round(tkd_class_prob[np.argmax(tkd_class_prob)],2))
                        , (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        except:
            pass

        cv2.imshow("Feed", img)
        
        k = cv2.waitKey(10)

        if k == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Capture landmarks to CSV file
# num_coords = len(results.pose_landmarks.landmark)

# landmarks = ['class']
# for val in range(1, num_coords + 1):
#     landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

# with open('coords.csv', mode='w', newline='') as f:
#     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     csv_writer.writerow(landmarks)