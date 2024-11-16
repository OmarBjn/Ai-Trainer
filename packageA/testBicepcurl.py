import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd 
import pickle

def processVideo(videoFile):
    mp_pose = mp.solutions.pose

    def getCoor(results):
        landmarksResults = results.pose_landmarks.landmark
        landmarks = []
            
        # shoulderR 0, elbowR 1, wristR 2, hipR 3, kneeR 4,
        #Get Coordinates           
        landmarks.append([landmarksResults[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarksResults[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        landmarks.append([landmarksResults[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarksResults[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
        landmarks.append([landmarksResults[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarksResults[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])
        landmarks.append([landmarksResults[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarksResults[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
        landmarks.append([landmarksResults[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarksResults[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])

        # Get landmark coordinates for shoulder, elbow, and wrist
        wristR = landmarks[2]
        elbowR = landmarks[1]
        shoulderR = landmarks[0]
        hipR = landmarks[3]
        kneeR = landmarks[4]

        distanceShoulderHipX = abs(shoulderR[0] - hipR[0])
        distanceShoulderElbowX = abs(shoulderR[0] - elbowR[0])
        shoulderGreaterThanElbowX = shoulderR[0] > elbowR[0]

        return wristR, elbowR, shoulderR, hipR, kneeR, shoulderGreaterThanElbowX, distanceShoulderElbowX, distanceShoulderHipX


    def draw_visual(image, accuracy, bodyLanguageClass, wristR, elbowR, shoulderR, hipR, kneeR, distanceShoulderElbowX):

        # Draw circles at elbow, shoulder, wrist, and ankle positions
        cv2.circle(image, (int(elbowR[0] * image.shape[1]), int(elbowR[1] * image.shape[0])), 8, (255, 0, 0), -1)
        cv2.circle(image, (int(shoulderR[0] * image.shape[1]), int(shoulderR[1] * image.shape[0])), 8, (255, 0, 0), -1)
        cv2.circle(image, (int(wristR[0] * image.shape[1]), int(wristR[1] * image.shape[0])), 8, (255, 0, 0), -1)
        cv2.circle(image, (int(hipR[0] * image.shape[1]), int(hipR[1] * image.shape[0])), 8, (255, 0, 0), -1)
        cv2.circle(image, (int(kneeR[0] * image.shape[1]), int(kneeR[1] * image.shape[0])), 8, (255, 0, 0), -1)

        # Draw lines connecting elbow, shoulder, wrist, and ankle
        cv2.line(image, (int(elbowR[0] * image.shape[1]), int(elbowR[1] * image.shape[0])),
                (int(wristR[0] * image.shape[1]), int(wristR[1] * image.shape[0])), (0 , 255 , 0), 2)
        cv2.line(image, (int(elbowR[0] * image.shape[1]), int(elbowR[1] * image.shape[0])),
                (int(shoulderR[0] * image.shape[1]), int(shoulderR[1] * image.shape[0])), (0 , 255 , 0), 2)
        cv2.line(image, (int(shoulderR[0] * image.shape[1]), int(shoulderR[1] * image.shape[0])),
                (int(hipR[0] * image.shape[1]), int(hipR[1] * image.shape[0])), (0 , 255 , 0), 2)
        cv2.line(image, (int(hipR[0] * image.shape[1]), int(hipR[1] * image.shape[0])),
                (int(kneeR[0] * image.shape[1]), int(kneeR[1] * image.shape[0])), (0 , 255 , 0), 2)
        
        # Determine which lines to draw in red based on the value of x
        if bodyLanguageClass == 'STAND_UP_STRAIGHT':
            # Draw line from shoulder to hip to knee in red
            cv2.line(image, (int(shoulderR[0] * image.shape[1]), int(shoulderR[1] * image.shape[0])),
                    (int(hipR[0] * image.shape[1]), int(hipR[1] * image.shape[0])), (0, 0, 255), 2)
            cv2.line(image, (int(hipR[0] * image.shape[1]), int(hipR[1] * image.shape[0])),
                    (int(kneeR[0] * image.shape[1]), int(kneeR[1] * image.shape[0])), (0, 0, 255), 2)
            if distanceShoulderElbowX > 0.08:
                cv2.line(image, (int(elbowR[0] * image.shape[1]), int(elbowR[1] * image.shape[0])),
                (int(wristR[0] * image.shape[1]), int(wristR[1] * image.shape[0])), (0 , 0 , 255), 2)
                cv2.line(image, (int(elbowR[0] * image.shape[1]), int(elbowR[1] * image.shape[0])),
                (int(shoulderR[0] * image.shape[1]), int(shoulderR[1] * image.shape[0])), (0 , 0 , 255), 2)
        elif bodyLanguageClass == 'MOVE_YOUR_ELBOW_FORWARD' or bodyLanguageClass == 'MOVE_YOUR_ELBOW_BACKWARD':
            cv2.line(image, (int(elbowR[0] * image.shape[1]), int(elbowR[1] * image.shape[0])),
                (int(wristR[0] * image.shape[1]), int(wristR[1] * image.shape[0])), (0 , 0 , 255), 2)
            cv2.line(image, (int(elbowR[0] * image.shape[1]), int(elbowR[1] * image.shape[0])),
                (int(shoulderR[0] * image.shape[1]), int(shoulderR[1] * image.shape[0])), (0 , 0 , 255), 2)
          
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
    landmarks = ['class', 'shoulderGreaterThanElbowX', 'distanceShoulderHipX', 'distanceShoulderElbowX'] 

    # must there exist pickle file
    # open the trained model. "pickle file"
    with open('packageA\dataset\coordsBicepCurl.pkl', 'rb') as f:
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

            # Check if frame is empty (end of video)
            if not ret:
                print("End of video.")
                break

            # frame = cv2.flip(frame, 1)
            # frame = cv2.flip(frame, 0)
        
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)           
            
            try: 
                wristR, elbowR, shoulderR, hipR, kneeR, shoulderGreaterThanElbowX, distanceShoulderElbowX, distanceShoulderHipX = getCoor(results)

                data = [shoulderGreaterThanElbowX, distanceShoulderHipX, distanceShoulderElbowX]

                X = pd.DataFrame([data], columns = landmarks[1:])
                bodyLanguageClass = model.predict(X)[0]
                accuracy = model.predict_proba(X)[0]

                # elbow, shoulder, wrist, hip, knee,
                draw_visual(
                    image,
                    str(round(accuracy[np.argmax(accuracy)], 2)),
                    bodyLanguageClass.split(' ')[0], 
                    wristR, elbowR, shoulderR, hipR, kneeR, distanceShoulderElbowX
                )

            except Exception as e: 
                print(e)
                pass

            out.write(image)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        return 'output.mp4'