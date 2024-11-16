import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd 
import pickle

# Function to process the uploaded video
def processVideo(videoFile):
    mp_pose = mp.solutions.pose

    def calculateAngle(a,b,c): # a,b,c = (x,y)
        a = np.array(a) #First
        b = np.array(b) #Second
        c = np.array(c) #Third
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        return angle  

    def getCoor(results):
        landmarksResults = results.pose_landmarks.landmark
        landmarks = []
            
        # shoulderR 0, elbowR 1, wristR 2, hipR 3, kneeR 4, ankleR 5
        #Get Coordinates           
        landmarks.append([landmarksResults[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarksResults[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        landmarks.append([landmarksResults[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarksResults[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
        landmarks.append([landmarksResults[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarksResults[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])
        landmarks.append([landmarksResults[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarksResults[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
        landmarks.append([landmarksResults[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarksResults[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])
        landmarks.append([landmarksResults[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarksResults[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])

        # to check is it pushUp or not
        y_wrist = landmarks[2][1]
        y_knee = landmarks[4][1]
        if y_wrist > y_knee:
            # hip, shoulder, elbow    
            shoulderAngleR = calculateAngle(landmarks[3], landmarks[0], landmarks[1]) # to straight shoulder
            # shoulder, hip, knee
            hibAngleR = calculateAngle(landmarks[0], landmarks[3], landmarks[4]) # to straight back
            # hip, knee, ankle 
            kneeAngleR = calculateAngle(landmarks[3], landmarks[4], landmarks[5]) # to straight knee

            # Initialize keypoints with angles
            angles = [shoulderAngleR, hibAngleR, kneeAngleR]

            # wristR, elbowR, shoulderR, hipR, kneeR, ankleR
            return angles, landmarks[2], landmarks[1], landmarks[0], landmarks[3], landmarks[4], landmarks[5]
        else:
            angles = [0.123]
            return angles, landmarks[2], landmarks[1], landmarks[0], landmarks[3], landmarks[4], landmarks[5]


    def draw_visual(image, accuracy, bodyLanguageClass, wristR, elbowR, shoulderR, hipR, kneeR, ankleR):

        # Draw circles at elbow, shoulder, wrist, and ankle positions
        cv2.circle(image, (int(elbowR[0] * image.shape[1]), int(elbowR[1] * image.shape[0])), 8, (255, 0, 0), -1)
        cv2.circle(image, (int(shoulderR[0] * image.shape[1]), int(shoulderR[1] * image.shape[0])), 8, (255, 0, 0), -1)
        cv2.circle(image, (int(wristR[0] * image.shape[1]), int(wristR[1] * image.shape[0])), 8, (255, 0, 0), -1)
        cv2.circle(image, (int(ankleR[0] * image.shape[1]), int(ankleR[1] * image.shape[0])), 8, (255, 0, 0), -1)
        cv2.circle(image, (int(hipR[0] * image.shape[1]), int(hipR[1] * image.shape[0])), 8, (255, 0, 0), -1)
        cv2.circle(image, (int(kneeR[0] * image.shape[1]), int(kneeR[1] * image.shape[0])), 8, (255, 0, 0), -1)

        # to check if is it pushup or not
        colorLine = [255 , 0]
        if bodyLanguageClass == 'NOT PUSH-UP':
            colorLine = [0 , 255]

        # Draw lines connecting elbow, shoulder, wrist, and ankle
        cv2.line(image, (int(elbowR[0] * image.shape[1]), int(elbowR[1] * image.shape[0])),
                (int(shoulderR[0] * image.shape[1]), int(shoulderR[1] * image.shape[0])), (0 , colorLine[0] , colorLine[1]), 2)
        cv2.line(image, (int(elbowR[0] * image.shape[1]), int(elbowR[1] * image.shape[0])),
                (int(wristR[0] * image.shape[1]), int(wristR[1] * image.shape[0])), (0 , colorLine[0] , colorLine[1]), 2)
        cv2.line(image, (int(shoulderR[0] * image.shape[1]), int(shoulderR[1] * image.shape[0])),
                (int(hipR[0] * image.shape[1]), int(hipR[1] * image.shape[0])), (0 , colorLine[0] , colorLine[1]), 2)
        cv2.line(image, (int(hipR[0] * image.shape[1]), int(hipR[1] * image.shape[0])),
                (int(kneeR[0] * image.shape[1]), int(kneeR[1] * image.shape[0])), (0 , colorLine[0] , colorLine[1]), 2)
        cv2.line(image, (int(kneeR[0] * image.shape[1]), int(kneeR[1] * image.shape[0])),
                (int(ankleR[0] * image.shape[1]), int(ankleR[1] * image.shape[0])), (0 , colorLine[0] , colorLine[1]), 2)
        
        # Determine which lines to draw in red based on the value of x
        if bodyLanguageClass == 'STRAIGHT_YOUR_SHOULDER':
            # Draw line from ankle to elbow to shoulder in red
            cv2.line(image, (int(wristR[0] * image.shape[1]), int(wristR[1] * image.shape[0])),
                    (int(elbowR[0] * image.shape[1]), int(elbowR[1] * image.shape[0])), (0, 0, 255), 2)
            cv2.line(image, (int(elbowR[0] * image.shape[1]), int(elbowR[1] * image.shape[0])),
                    (int(shoulderR[0] * image.shape[1]), int(shoulderR[1] * image.shape[0])), (0, 0, 255), 2)
        elif bodyLanguageClass == 'STRAIGHT_YOUR_BACK':
            # Draw line from shoulder to hip to knee in red
            cv2.line(image, (int(shoulderR[0] * image.shape[1]), int(shoulderR[1] * image.shape[0])),
                    (int(hipR[0] * image.shape[1]), int(hipR[1] * image.shape[0])), (0, 0, 255), 2)
            cv2.line(image, (int(hipR[0] * image.shape[1]), int(hipR[1] * image.shape[0])),
                    (int(kneeR[0] * image.shape[1]), int(kneeR[1] * image.shape[0])), (0, 0, 255), 2)
        elif bodyLanguageClass == 'STRAIGHT_YOUR_LEG':
            # Draw line from hip to knee to ankle in red
            cv2.line(image, (int(hipR[0] * image.shape[1]), int(hipR[1] * image.shape[0])),
                    (int(kneeR[0] * image.shape[1]), int(kneeR[1] * image.shape[0])), (0, 0, 255), 2)
            cv2.line(image, (int(kneeR[0] * image.shape[1]), int(kneeR[1] * image.shape[0])),
                    (int(ankleR[0] * image.shape[1]), int(ankleR[1] * image.shape[0])), (0, 0, 255), 2)
        
        # Get status box
        cv2.rectangle(image, (0,0), (450, 50), (255, 255, 255), -1)    

        # Draw rectangle for CLASS and bodyLanguageClass at the top left corner
        cv2.rectangle(image, (0, 0), (370, 50), (255, 255, 255), -1)
        cv2.putText(image, 'CLASS', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, bodyLanguageClass, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw rectangle for ACCURACY at the bottom left corner
        height = image.shape[0]
        cv2.rectangle(image, (0, height - 50), (100, height), (255, 255, 255), -1)
        cv2.putText(image, 'ACCURACY', (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, accuracy, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # to named all the columns of our data
    landmarks = ['class','shoulderAngleR', 'hibAngleR', 'kneeAngleR'] 

    # must there exist pickle file
    # open the trained model. "pickle file"
    with open('packageA\dataset\coordsPushUps.pkl', 'rb') as f:
        model = pickle.load(f)

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(videoFile)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Change the path to save the output video
    output_path = os.path.join('packageA', 'output.mp4')

    # Change the codec to 'H264'
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

    ## Initiate holistic model
    with mp_pose.Pose(
        static_image_mode = False,
        model_complexity = 1,
        smooth_landmarks = True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
        ) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()

            # frame = cv2.flip(frame, 1)
            # frame = cv2.flip(frame, 0)

            # Check if frame is empty (end of video)
            if not ret:
                print("End of video.")
                break
        
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)           
            
            try: 

                angles, wristR, elbowR, shoulderR, hipR, kneeR, ankleR = getCoor(results)

                # if condition is it pushup form or not
                if angles[0] != 0.123: 
                    X = pd.DataFrame([angles], columns = landmarks[1:])
                    bodyLanguageClass = model.predict(X)[0]
                    accuracy = model.predict_proba(X)[0]
                
                    draw_visual(
                        image,
                        str(round(accuracy[np.argmax(accuracy)], 2)),
                        bodyLanguageClass.split(' ')[0], 
                        wristR, elbowR, shoulderR, hipR, kneeR, ankleR
                    )
                else:
                    draw_visual(
                        image,
                        '1',
                        'NOT PUSH-UP', 
                        wristR, elbowR, shoulderR, hipR, kneeR, ankleR
                    )

            except Exception as e: 
                print(e)
                pass

            out.write(image)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        return 'output.mp4'