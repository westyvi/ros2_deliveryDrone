import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from hand_detection.mpDraw import draw_landmarks_on_image as MPDraw
import numpy as np

# define paths and plotting variables
imgPath = "/home/westyvi/Documents/school/personal_projects/palmDetection/palm1.jpg"
model_path = '/home/westyvi/Documents/school/personal_projects/palmDetection/hand_landmarker.task'
showAnnotation = True

# Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Load the input image.
image = mp.Image.create_from_file(imgPath)

# Detect hand landmarks from the input image.
detection_result = detector.detect(image)

# Process the classification result. In this case, visualize it.
annotated_image = MPDraw(image.numpy_view(), detection_result)
if showAnnotation:
    cv2.imshow('detected image', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    
# insert keypoints into numpy array for processing
keypoints = np.zeros((3,len(detection_result.hand_landmarks[0])))
for i, keypoint in enumerate(detection_result.hand_landmarks[0]):
    keypoints[:,i] = np.array([keypoint.x, keypoint.y, keypoint.z])

# see if fingertips extend some distance beyond palm
palm_open = True
palm_indices = [5, 9, 13, 17] # indices of bases of fingers
finger_indices = [8, 12, 16, 20] # indices of fingertip keypoints
finger_extended_cutoff = 0.5
for i, fidx in enumerate(finger_indices): # fidx = fingertip index
    pidx = palm_indices[i] # palm/finger base index
    finger_keypoint = keypoints[:,fidx]
    palm_keypoint = keypoints[:,pidx]
    finger_vector = (palm_keypoint - finger_keypoint)
    wrist2palm_vector = (keypoints[:,0] - palm_keypoint)
    
    # see what results this dot product gives to determine finger cutoff:
    val = np.inner(finger_vector, wrist2palm_vector)/np.inner(wrist2palm_vector, wrist2palm_vector)
    print('finger ', i, ': ', val)
    if np.inner(finger_vector, wrist2palm_vector)/np.inner(wrist2palm_vector, wrist2palm_vector) < finger_extended_cutoff:
        palm_open = False

print(palm_open)
    
    






